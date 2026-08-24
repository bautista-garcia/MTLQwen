#include "qwen35.hpp"
#include <algorithm>

namespace infeng::qwen35 {
namespace {
constexpr uint64_t halfBytes = 2;
constexpr uint32_t attentionConcurrency = 8;
MTL::Size size(uint64_t x, uint64_t y = 1, uint64_t z = 1) { return MTL::Size(x, y, z); }
uint64_t roundUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

Tensor linear(CommandBuffer& commands, const Tensor& x, const Linear& weight, uint32_t rows, Tensor output,
              Scratch& scratch, bool synchronize = true) {
    if (weight.type == QuantType::F16) {
        if (synchronize) commands.dispatch(weight.decode, size(1024, rows), size(32), {output, x, weight.weight}, rows);
        else commands.dispatchConcurrent(weight.decode, size(1024, rows), size(32), {output, x, weight.weight}, rows);
        return output;
    }
    if (scratch.decodeMode) {
        MTL::Size threads = size((weight.n + weight.outputsPerGroup - 1) / weight.outputsPerGroup *
                                 weight.decodeThreads, rows);
        if (synchronize)
            commands.dispatch(weight.decode, threads, size(weight.decodeThreads), {output, x, weight.weight});
        else commands.dispatchConcurrent(weight.decode, threads, size(weight.decodeThreads),
                                         {output, x, weight.weight});
        return output;
    }
    uint32_t padded = roundUp(rows, 8); Tensor source = x;
    if (padded != rows) {
        source = scratch.padInput.view(0, uint64_t(padded) * weight.k * halfBytes);
        commands.dispatch(scratch.padRows, size(roundUp(uint64_t(padded) * weight.k, 256)), size(256), {source, x},
                          rows, padded, weight.k);
    }
    MTL::Size threads = size(512 * (weight.n / 32), (padded + 127) / 128);
    if (synchronize) commands.dispatch(weight.prefill, threads, size(512), {output, source, weight.weight}, int64_t(padded));
    else commands.dispatchConcurrent(weight.prefill, threads, size(512), {output, source, weight.weight}, int64_t(padded));
    return output.view(0, uint64_t(rows) * weight.n * halfBytes);
}

Tensor rms(CommandBuffer& commands, Model& model, const Tensor& x, const Tensor& weight, Tensor output, uint32_t rows,
           uint32_t dim) {
    commands.dispatch(model.kernels.rms, size(256, rows), size(256), {output, x, weight}, rows, dim, 1e-6f); return output;
}
Tensor add(CommandBuffer& commands, Model& model, const Tensor& a, const Tensor& b, Tensor output, uint64_t elements) {
    commands.dispatch(model.kernels.add, size(roundUp(elements, 256)), size(256), {output, a, b}, uint32_t(elements));
    return output;
}
Tensor linearAdd(CommandBuffer& commands, Model& model, const Tensor& x, const Linear& weight,
                 const Tensor& residual, Tensor output, Scratch& scratch, uint32_t rows) {
    if (scratch.decodeMode) {
        MTL::Size threads = size((weight.n + weight.outputsPerGroup - 1) / weight.outputsPerGroup *
                                 weight.decodeThreads, rows);
        commands.dispatch(weight.decodeAdd, threads, size(weight.decodeThreads), {output, x, weight.weight, residual});
        return output;
    }
    Tensor projected = linear(commands, x, weight, rows, scratch.projected, scratch);
    return add(commands, model, residual, projected, output, uint64_t(rows) * weight.n);
}
Tensor mlp(CommandBuffer& commands, Model& model, const Tensor& x, const MlpWeights& weights,
           const Tensor& residual, Tensor output, Scratch& scratch, uint32_t rows) {
    if (scratch.decodeMode)
        commands.dispatch(weights.fusedDecode, size(12288 / weights.outputsPerGroup * weights.decodeThreads, rows),
                          size(weights.decodeThreads), {scratch.mlpMixed, x, weights.gate.weight, weights.up.weight},
                          int64_t(rows));
    else {
        Tensor gate = linear(commands, x, weights.gate, rows, scratch.mlpGate, scratch, false);
        Tensor up = linear(commands, x, weights.up, rows, scratch.mlpUp, scratch);
        uint64_t elements = uint64_t(rows) * 12288;
        commands.dispatch(model.kernels.siluMul, size(roundUp(elements, 256)), size(256),
                          {scratch.mlpMixed, gate, up}, uint32_t(elements));
    }
    return linearAdd(commands, model, scratch.mlpMixed, weights.down, residual, output, scratch, rows);
}

Tensor attention(CommandBuffer& commands, Model& model, const Layer& layer, const Tensor& x, const Tensor& residual,
                 Tensor output, Scratch& scratch, uint32_t batch, uint32_t rows, uint32_t kvLayer,
                 const Tensor& positions) {
    const AttentionWeights& weight = layer.attention;
    Tensor qg = linear(commands, x, weight.q, rows, scratch.attnQG, scratch, false);
    Tensor k = linear(commands, x, weight.k, rows, scratch.attnK, scratch, false);
    Tensor v = linear(commands, x, weight.v, rows, scratch.attnV, scratch);
    commands.dispatch(model.kernels.attentionPrepare, size(256, uint64_t(rows) * 20), size(256),
                      {scratch.attnQRope, scratch.attnKRope, qg, k, weight.qNorm, weight.kNorm, model.rope,
                       positions, model.queryStartLoc}, batch);
    uint32_t splits = std::max(1u, attentionConcurrency / rows);
    commands.dispatch(model.kernels.attentionScan, size(uint64_t(rows) * 16 * splits * 128), size(128),
                      {scratch.attnPartials, scratch.attnQRope, scratch.attnKRope, v, model.kv->key(kvLayer),
                       model.kv->value(kvLayer), model.sequenceSlots, positions, model.queryStartLoc},
                      batch, rows, model.maxLogicalBlocks * blockTokens, splits);
    commands.dispatch(model.kernels.attentionReduce, size(uint64_t(rows) * 16 * 128), size(128),
                      {scratch.attnGated, scratch.attnPartials, qg}, rows, splits);
    return linearAdd(commands, model, scratch.attnGated, weight.out, residual, output, scratch, rows);
}

Tensor gdn(CommandBuffer& commands, Model& model, const Layer& layer, const Tensor& x, const Tensor& residual,
           Tensor output, Scratch& scratch, LayerState& state, uint32_t rows, const Batch& layout) {
    const GdnWeights& weight = layer.gdn;
    Tensor mixed = linear(commands, x, weight.qkv, rows, scratch.gdnMixed, scratch, false);
    Tensor z = linear(commands, x, weight.z, rows, scratch.gdnZ, scratch, false);
    commands.dispatch(model.kernels.gdnBaPrepare, size(32 * 64, rows), size(64),
                      {scratch.gdnB, scratch.gdnG, x, weight.b.weight, weight.a.weight, weight.A, weight.dt}, rows);
    for (uint32_t row = 0; row < layout.size; ++row) {
        uint32_t start = layout.starts[row], length = layout.starts[row + 1] - start;
        uint64_t offset = uint64_t(start) * 8192 * 2;
        Tensor slots = model.sequenceSlots.view(uint64_t(row) * 4, 4), banks = model.stateBanks.view(uint64_t(row) * 4, 4);
        Tensor valid = model.batchKvValid.view(uint64_t(row) * 4, 4);
        Tensor destination = scratch.gdnConvolved.view(offset, uint64_t(length) * 8192 * 2);
        Tensor source = mixed.view(offset, uint64_t(length) * 8192 * 2);
        if (layout.queries[row].draft)
            commands.dispatch(model.kernels.gdnConvCandidates, size(uint64_t(length) * 8192), size(256),
                              {destination, state.conv[0], state.conv[1],
                               state.candidateConv.view(uint64_t(layout.candidates[row]) * convStateBytes,
                                                        uint64_t(length) * convStateBytes),
                               source, weight.conv, slots, banks, valid}, int64_t(1), int64_t(length));
        else commands.dispatch(model.kernels.gdnConv, size(uint64_t(8192) * std::max(length, 4u)), size(256),
                               {destination, state.conv[0], state.conv[1], source, weight.conv, slots, banks, valid},
                               int64_t(1), int64_t(length));
    }
    uint64_t elements = uint64_t(rows) * 4096;
    commands.dispatch(model.kernels.splitQk, size(roundUp(elements, 256)), size(256),
                      {scratch.gdnQ, scratch.gdnK, scratch.gdnV, scratch.gdnConvolved}, rows);
    for (uint32_t row = 0; row < layout.size; ++row) {
        uint32_t start = layout.starts[row], length = layout.starts[row + 1] - start;
        uint64_t offset = uint64_t(start) * 4096 * 2;
        Tensor slots = model.sequenceSlots.view(uint64_t(row) * 4, 4), banks = model.stateBanks.view(uint64_t(row) * 4, 4);
        Tensor valid = model.batchKvValid.view(uint64_t(row) * 4, 4);
        Tensor destination = scratch.gdnDelta.view(offset, uint64_t(length) * 4096 * 2);
        Tensor q = scratch.gdnQ.view(offset, uint64_t(length) * 4096 * 2);
        Tensor k = scratch.gdnK.view(offset, uint64_t(length) * 4096 * 2);
        Tensor v = scratch.gdnV.view(offset, uint64_t(length) * 4096 * 2);
        Tensor g = scratch.gdnG.view(uint64_t(start) * 32 * 4, uint64_t(length) * 32 * 4);
        Tensor beta = scratch.gdnB.view(uint64_t(start) * 32 * 2, uint64_t(length) * 32 * 2);
        if (layout.queries[row].draft)
            commands.dispatch(model.kernels.deltaCandidates, size(32 * 512), size(512),
                              {destination, state.recurrent[0], state.recurrent[1],
                               state.candidateRecurrent.view(uint64_t(layout.candidates[row]) * recurrentStateBytes,
                                                             uint64_t(length) * recurrentStateBytes),
                               slots, banks, valid, q, k, v, g, beta}, int64_t(1), int64_t(length), int64_t(32),
                              int64_t(length) * 4096, int64_t(4096), int64_t(128), int64_t(1));
        else if (length == 1)
            commands.dispatch(model.kernels.deltaDecode, size(32 * 512), size(512),
                              {destination, state.recurrent[0], state.recurrent[1], slots, banks, valid, q, k, v, g, beta},
                              int64_t(1), int64_t(1), int64_t(32), int64_t(4096), int64_t(4096), int64_t(128), int64_t(1));
        else commands.dispatch(model.kernels.deltaPrefill, size(128, 32), size(128),
                               {destination, state.recurrent[0], state.recurrent[1], slots, banks, valid, q, k, v, g, beta},
                               int64_t(1), int64_t(length), int64_t(32), int64_t(length) * 4096, int64_t(4096),
                               int64_t(128), int64_t(1));
    }
    commands.dispatch(model.kernels.gdnNorm, size(128, uint64_t(rows) * 32), size(128),
                      {scratch.gdnNormed, scratch.gdnDelta, z, weight.norm}, 1e-6f);
    return linearAdd(commands, model, scratch.gdnNormed, weight.out, residual, output, scratch, rows);
}

Tensor decoderLayer(CommandBuffer& commands, Model& model, const Layer& layer, LayerState& state, const Tensor& hidden,
                    Tensor output, Scratch& scratch, uint32_t rows, uint32_t kvLayer, const Tensor& positions,
                    const Batch& layout) {
    Tensor x = rms(commands, model, hidden, layer.inputNorm, scratch.inputNorm, rows, 4096);
    Tensor mid = layer.fullAttention
        ? attention(commands, model, layer, x, hidden, scratch.mid, scratch, layout.size, rows, kvLayer, positions)
        : gdn(commands, model, layer, x, hidden, scratch.mid, scratch, state, rows, layout);
    x = rms(commands, model, mid, layer.postNorm, scratch.postNorm, rows, 4096);
    return mlp(commands, model, x, layer.mlp, mid, output, scratch, rows);
}

void sample(CommandBuffer& commands, Model& model, const Session& session, const Tensor& logits, const Tensor& token,
            float temperature, float topP, int32_t topK) {
    if (temperature <= 0)
        commands.dispatch(model.kernels.argmax, size(256), size(256), {token, logits}, uint32_t(248320));
    else commands.dispatch(model.kernels.sample, size(1), size(1),
                           {token, model.rng.view(uint64_t(session.slot) * 8, 8), logits},
                           uint32_t(248320), temperature, topP, topK);
}

struct Metadata {
    std::array<uint32_t, maxBatchSequences> valid{}, slots{}, banks{};
    uint32_t rows = 0;
    void add(const Session& session) {
        valid[rows] = session.kvValid; slots[rows] = session.slot; banks[rows++] = session.bank;
    }
    void upload(Model& model) const {
        uint64_t bytes = uint64_t(rows) * 4;
        model.device.write(model.batchKvValid, valid.data(), bytes);
        model.device.write(model.sequenceSlots, slots.data(), bytes); model.device.write(model.stateBanks, banks.data(), bytes);
    }
};

void writeMetadata(Model& model, const Batch& batch) {
    Metadata metadata;
    for (uint32_t row = 0; row < batch.size; ++row)
        metadata.add(*batch.queries[row].session);
    metadata.upload(model);
    model.device.write(model.queryStartLoc, batch.starts.data(), uint64_t(batch.size + 1) * 4);
}

void encodeMtp(CommandBuffer& commands, Model& model, const Batch& batch, uint32_t rows, Scratch& scratch) {
    if (!model.hasMtp) return;
    commands.dispatch(model.kernels.embed, size(4096, rows), size(256),
                      {scratch.postNorm, model.inputIds, model.embedding}, int64_t(rows), int64_t(4096));
    commands.dispatch(model.kernels.mtpFuse, size(256, rows), size(256),
                      {scratch.mtpFused, scratch.postNorm, scratch.targetHidden, model.mtpSeeds,
                       model.sequenceSlots, model.stateBanks, model.batchKvValid, model.queryStartLoc,
                       model.mtp.embeddingNorm, model.mtp.hiddenNorm}, batch.size, rows);
    Tensor hidden = linear(commands, scratch.mtpFused, model.mtp.fusion, rows, scratch.hidden[0], scratch);
    Tensor normalized = rms(commands, model, hidden, model.mtp.layer.inputNorm, scratch.inputNorm, rows, 4096);
    Tensor k = linear(commands, normalized, model.mtp.layer.attention.k, rows, scratch.attnK, scratch, false);
    Tensor v = linear(commands, normalized, model.mtp.layer.attention.v, rows, scratch.attnV, scratch);
    commands.dispatch(model.kernels.mtpStore, size(uint64_t(rows) * 4 * 256), size(256),
                      {model.kv->key(targetKvLayers), model.kv->value(targetKvLayers), k, v,
                       model.sequenceSlots, model.batchKvValid, model.queryStartLoc,
                       model.mtp.layer.attention.kNorm, model.rope}, batch.size, rows,
                      model.maxLogicalBlocks * blockTokens);
    for (uint32_t row = 0; row < batch.size; ++row) if (!batch.queries[row].draft)
        commands.copy(scratch.targetHidden.view(uint64_t(batch.starts[row + 1] - 1) * 4096 * 2, 4096 * 2),
                      mtpSeed(model, *batch.queries[row].session, 1 - batch.queries[row].session->bank), 4096 * 2);
}
}  // namespace

void Scratch::ensure(Device& device, uint32_t requested, bool mtp) {
    // One grow-only private arena backs every activation, with aliases only where lifetimes do not overlap.
    uint32_t target = roundUp(requested, 32); if (uint64_t(target) * 4096 * 2 <= hidden[0].bytes) return;
    uint64_t rows = target, hiddenBytes = rows * 4096 * 2, mlpBytes = rows * 12288 * 2;
    uint64_t padBytes = rows * 12288 * 2;
    Tensor shared;
    std::vector<std::pair<Tensor*, uint64_t>> slots = {
        {&shared, padBytes + 2 * mlpBytes},
        {&hidden[0], hiddenBytes}, {&hidden[1], hiddenBytes}, {&inputNorm, hiddenBytes}, {&postNorm, hiddenBytes},
        {&mlpMixed, mlpBytes}, {&attnQG, rows * 8192 * 2}, {&attnK, rows * 1024 * 2},
        {&attnV, rows * 1024 * 2}, {&attnQRope, hiddenBytes}, {&attnKRope, rows * 1024 * 2},
        {&attnOut, hiddenBytes}, {&attnGated, hiddenBytes},
        {&attnPartials, std::max<uint64_t>(rows, attentionConcurrency) * 16 * 258 * 4},
        {&gdnMixed, rows * 8192 * 2}, {&gdnZ, hiddenBytes},
        {&gdnB, rows * 32 * 2}, {&gdnA, rows * 32 * 2}, {&gdnG, rows * 32 * 4},
        {&gdnConvolved, rows * 8192 * 2}, {&gdnQ, hiddenBytes}, {&gdnK, hiddenBytes}, {&gdnV, hiddenBytes},
        {&gdnDelta, hiddenBytes}, {&gdnNormed, hiddenBytes}, {&mid, hiddenBytes}, {&projected, hiddenBytes},
        {&mtpFused, mtp ? rows * 8192 * 2 : 0},
        {&targetHidden, mtp ? hiddenBytes : 0},
        {&targetLogits, std::min<uint64_t>(rows, maxLogitRows) * 248320 * 2}};
    uint64_t bytes = 0; for (auto [_, count] : slots) bytes = roundUp(bytes, 256) + count;
    Tensor storage = device.empty(bytes); uint64_t offset = 0;
    for (auto [tensor, count] : slots) { offset = roundUp(offset, 256); *tensor = storage.view(offset, count); offset += count; }
    padInput = shared.view(0, padBytes); mlpGate = shared.view(padBytes, mlpBytes);
    mlpUp = shared.view(padBytes + mlpBytes, mlpBytes);
}

// The MTP drafter advances a compact subset of sessions; target verification remains the only commit authority.
void draft(Model& model, Batch& batch) {
    Metadata metadata; std::array<int32_t, maxBatchSequences> anchors{};
    std::array<uint32_t, maxBatchSequences * maxDraftTokens> positions{};
    for (uint32_t i = 0; i < batch.size; ++i) if (batch.queries[i].draft) {
        Query& query = batch.queries[i]; Session& session = *query.session;
        uint32_t row = batch.candidates[i] / query.count;
        metadata.add(session); anchors[row] = session.request[session.kvValid];
        for (uint32_t step = 0; step < batch.drafts; ++step)
            positions[step * maxBatchSequences + row] = session.kvValid + step;
    }
    uint32_t rows = metadata.rows; metadata.upload(model); Batch layout; layout.size = rows;
    for (uint32_t row = 0; row < rows; ++row) layout.starts[row + 1] = row + 1;
    model.device.write(model.queryStartLoc, layout.starts.data(), uint64_t(rows + 1) * 4);
    model.device.write(model.draftTokens, anchors.data(), uint64_t(rows) * 4);
    model.device.write(model.draftPositions, positions.data(), sizeof(positions));
    Scratch& scratch = model.scratch(1, rows); CommandBuffer commands(model.device, 1 << 18, true, 128, "mtp_draft");
    for (uint32_t step = 0; step < batch.drafts; ++step) {
        Tensor ids = model.draftTokens.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4);
        Tensor stepPositions = model.draftPositions.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4);
        commands.dispatch(model.kernels.embed, size(4096, rows), size(256),
                          {scratch.postNorm, ids, model.embedding}, int64_t(rows), int64_t(4096));
        commands.dispatch(model.kernels.mtpDraftFuse, size(256, rows), size(256),
                          {scratch.mtpFused, scratch.postNorm, scratch.inputNorm, model.mtpSeeds,
                           model.sequenceSlots, model.stateBanks, model.batchKvValid, model.mtp.embeddingNorm,
                           model.mtp.hiddenNorm}, rows, uint32_t(1), step ? uint32_t(0) : uint32_t(2));
        Tensor hidden = linear(commands, scratch.mtpFused, model.mtp.fusion, rows, scratch.hidden[0], scratch);
        hidden = decoderLayer(commands, model, model.mtp.layer, model.states[0], hidden, scratch.hidden[1], scratch,
                              rows, targetKvLayers, stepPositions, layout);
        Tensor normalized = rms(commands, model, hidden, model.mtp.outputNorm, scratch.inputNorm, rows, 4096);
        Tensor logits = linear(commands, normalized, model.head, rows, scratch.targetLogits, scratch);
        for (uint32_t row = 0; row < rows; ++row)
            commands.dispatch(model.kernels.argmax, size(256), size(256),
                              {model.draftTokens.view((uint64_t(step + 1) * maxBatchSequences + row) * 4, 4),
                               logits.view(uint64_t(row) * 248320 * 2, 248320 * 2)}, uint32_t(248320));
    }
    commands.commit(); auto* proposed = static_cast<int32_t*>(model.draftTokens.buffer->metalBuffer->contents());
    for (uint32_t row = 0; row < batch.size; ++row) if (batch.queries[row].draft)
        for (uint32_t step = 0; step < batch.drafts; ++step)
            batch.queries[row].session->request.push_back(proposed[(step + 1) * maxBatchSequences +
                                                                   batch.candidates[row] / batch.queries[row].count]);
}

// Every target pass consumes one packed variable-length layout and samples only requested terminal rows.
void forward(Model& model, Batch& batch, float temperature, float topP, int32_t topK) {
    uint32_t rows = batch.starts[batch.size], maxQuery = 0, logitCount = 0;
    std::array<int32_t, maxBatchTokens> tokens{}; std::array<uint32_t, maxLogitRows> logitRows{};
    for (uint32_t row = 0; row < batch.size; ++row) {
        Query& query = batch.queries[row]; Session& session = *query.session;
        std::copy_n(session.request.begin() + session.kvValid, query.count, tokens.begin() + batch.starts[row]);
        maxQuery = std::max(maxQuery, query.count); if (!query.sample) continue;
        query.logit = logitCount; uint32_t count = query.draft ? query.count : 1;
        for (uint32_t i = 0; i < count; ++i)
            logitRows[logitCount++] = batch.starts[row] + (query.draft ? i : query.count - 1);
    }
    model.device.write(model.inputIds, tokens.data(), uint64_t(rows) * 4);
    if (logitCount) model.device.write(model.logitRows, logitRows.data(), uint64_t(logitCount) * 4);
    writeMetadata(model, batch); model.ensureCandidates(batch.candidates[batch.size]);
    Scratch& scratch = model.scratch(maxQuery, rows);
    const char* phase = maxQuery == 1 ? "decode" : batch.drafts ? "verify" : "prefill";
    CommandBuffer commands(model.device, uint64_t(512 + 48 * batch.size) * 128, true, 1600 + 24 * batch.size, phase);
    commands.dispatch(model.kernels.embed, size(4096, rows), size(256),
                      {scratch.hidden[0], model.inputIds, model.embedding}, int64_t(rows), int64_t(4096));
    Tensor hidden = scratch.hidden[0];
    for (uint32_t i = 0; i < model.layers.size(); ++i)
        hidden = decoderLayer(commands, model, model.layers[i], model.states[i], hidden, scratch.hidden[(i + 1) & 1],
                              scratch, rows, model.layers[i].kvIndex, model.batchKvValid, batch);
    if (model.hasMtp) commands.copy(hidden, scratch.targetHidden.view(0, uint64_t(rows) * 4096 * 2),
                                    uint64_t(rows) * 4096 * 2);
    if (logitCount) {
        Tensor input = hidden;
        if (logitCount != rows) {
            commands.dispatch(model.kernels.gatherRows, size(roundUp(uint64_t(logitCount) * 4096, 256)), size(256),
                              {scratch.postNorm, hidden, model.logitRows}, logitCount, uint32_t(4096));
            input = scratch.postNorm;
        }
        Tensor normalized = rms(commands, model, input, model.norm, scratch.inputNorm, logitCount, 4096);
        Tensor logits = linear(commands, normalized, model.head, logitCount, scratch.targetLogits, scratch);
        for (uint32_t row = 0; row < batch.size; ++row) if (batch.queries[row].sample) {
            Query& query = batch.queries[row];
            uint32_t count = query.draft ? query.count : 1;
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t result = query.logit + i;
                sample(commands, model, *query.session, logits.view(uint64_t(result) * 248320 * 2, 248320 * 2),
                       model.outputTokens.view(uint64_t(result) * 4, 4), temperature, topP, topK);
            }
        }
    }
    encodeMtp(commands, model, batch, rows, scratch); commands.commit();
}

}  // namespace infeng::qwen35
