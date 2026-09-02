#include "qwen35.hpp"
#include <algorithm>

namespace infeng::qwen35 {
namespace {
constexpr uint32_t attentionConcurrency = 8;
MTL::Size size(uint64_t x, uint64_t y = 1, uint64_t z = 1) {
  return MTL::Size(x, y, z);
}
uint64_t roundUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct Ops {
  CommandBuffer& commands;
  Model& model;
  Scratch& scratch;

  Tensor linear(const Tensor& x, const Linear& weight, uint32_t rows, Tensor output, const Tensor& residual = {}) {
    if (residual.buffer && !scratch.decodeMode) {
      Tensor projected = linear(x, weight, rows, scratch.projected);
      uint32_t elements = rows * weight.n;
      commands.dispatch(output.buffer->device->pipeline("add_half"), size(roundUp(elements, 256)), size(256), {output, residual, projected},
                        elements);
      return output;
    }
    if (scratch.decodeMode) {
      uint32_t group = weight.pipeline[linearDecode]->maxTotalThreadsPerThreadgroup();
      MTL::Size threads = size((weight.n + weight.outputsPerGroup - 1) / weight.outputsPerGroup * group, rows);
      if (residual.buffer)
        commands.dispatch(weight.pipeline[linearDecodeAdd], threads, size(group), {output, x, weight.weight, residual});
      else
        commands.dispatch(weight.pipeline[linearDecode], threads, size(group), {output, x, weight.weight});
      return output;
    }
    uint32_t padded = roundUp(rows, 8);
    Tensor source = x;
    if (padded != rows) {
      source = scratch.padInput.view(0, uint64_t(padded) * weight.k * 2);
      commands.dispatch(scratch.padRows, size(roundUp(uint64_t(padded) * weight.k, 256)), size(256), {source, x}, rows, padded, weight.k);
    }
    bool small = padded <= 8;
    Pipeline* pipeline = weight.pipeline[small ? linearPrefillSmall : linearPrefill];
    MTL::Size group = size(small ? 64 : 512);
    MTL::Size threads = size(group.width * (weight.n / (small ? 8 : 32)), (padded + 127) / 128);
    commands.dispatch(pipeline, threads, group, {output, source, weight.weight}, int64_t(padded));
    return output.view(0, uint64_t(rows) * weight.n * 2);
  }

  Tensor rms(const Tensor& x, const Tensor& weight, Tensor output, uint32_t rows, uint32_t dim = 4096) {
    commands.dispatch(model.device.pipeline("rmsnorm"), size(256, rows), size(256), {output, x, weight}, rows, dim, 1e-6f);
    return output;
  }

  Tensor mlp(const Tensor& x, const MlpWeights& weights, const Tensor& residual, Tensor output, uint32_t rows) {
    if (scratch.decodeMode)
      commands.dispatch(weights.fusedDecode, size(12288 / weights.outputsPerGroup * weights.fusedDecode->maxTotalThreadsPerThreadgroup(), rows),
                        size(weights.fusedDecode->maxTotalThreadsPerThreadgroup()), {scratch.mlpMixed, x, weights.gate.weight, weights.up.weight},
                        int64_t(rows));
    else {
      Tensor gate = linear(x, weights.gate, rows, scratch.mlpGate);
      Tensor up = linear(x, weights.up, rows, scratch.mlpUp);
      uint64_t elements = uint64_t(rows) * 12288;
      commands.dispatch(output.buffer->device->pipeline("silu_mul"), size(roundUp(elements, 256)), size(256), {scratch.mlpMixed, gate, up},
                        uint32_t(elements));
    }
    return linear(scratch.mlpMixed, weights.down, rows, output, residual);
  }

  Tensor attention(const Layer& layer, const Tensor& x, const Tensor& residual, Tensor output, uint32_t batch, uint32_t rows, uint32_t kvLayer,
                   const Tensor& positions, uint32_t dflash = unbound) {
    const AttentionWeights& weight = layer.attention;
    Tensor qg = linear(x, weight.q, rows, scratch.attnQG);
    Tensor k = linear(x, weight.k, rows, scratch.attnK);
    Tensor v = linear(x, weight.v, rows, scratch.attnV);
    bool draft = dflash != unbound;
    uint32_t heads = draft ? 32 : 16, group = draft ? 128 : 256, kvLayerIndex = draft ? targetKvLayers + dflash : kvLayer;
    Tensor rope = draft ? model.dflashRope : model.rope, result = draft ? scratch.attnOut : scratch.attnGated;
    commands.dispatch(model.device.pipeline(draft ? "dflash_attention_prepare" : "attention_prepare"),
                      size(group, uint64_t(rows) * (draft ? 40 : 20)), size(group),
                      {scratch.attnQRope, scratch.attnKRope, qg, k, weight.qNorm, weight.kNorm, rope, positions, model.queryStartLoc}, batch);
    uint32_t splits = std::max(1u, attentionConcurrency / rows);
    commands.dispatch(model.device.pipeline(draft ? "dflash_attention_scan" : "attention_scan"), size(uint64_t(rows) * heads * splits * 128),
                      size(128),
                      {scratch.attnPartials, scratch.attnQRope, scratch.attnKRope, v, model.kv->key(kvLayerIndex), model.kv->value(kvLayerIndex),
                       model.sequenceSlots, positions, model.queryStartLoc},
                      batch, rows, model.maxLogicalBlocks * blockTokens, splits, uint32_t(draft && dflash + 1 < dflashLayers));
    commands.dispatch(model.device.pipeline(draft ? "dflash_attention_reduce" : "attention_reduce"), size(uint64_t(rows) * heads * 128), size(128),
                      {result, scratch.attnPartials, qg}, rows, splits);
    return linear(result, weight.out, rows, output, residual);
  }

  Tensor gdn(const Layer& layer, const Tensor& x, const Tensor& residual, Tensor output, LayerState& state, uint32_t rows, const Batch& layout) {
    const GdnWeights& weight = layer.gdn;
    Tensor mixed = linear(x, weight.qkv, rows, scratch.gdnMixed);
    Tensor z = linear(x, weight.z, rows, scratch.gdnZ);
    commands.dispatch(model.device.pipeline("gdn_ba_prepare_4096x32"), size(32 * 64, rows), size(64),
                      {scratch.gdnB, scratch.gdnG, x, weight.b, weight.a, weight.A, weight.dt}, rows, weight.baF32);
    for (uint32_t row = 0; row < layout.size; ++row) {
      uint32_t start = layout.query_start_loc[row], length = layout.query_start_loc[row + 1] - start;
      uint64_t offset = uint64_t(start) * 8192 * 2;
      Tensor slots = model.sequenceSlots.view(uint64_t(row) * 4, 4), banks = model.stateBanks.view(uint64_t(row) * 4, 4);
      Tensor valid = model.batchKvValid.view(uint64_t(row) * 4, 4);
      Tensor destination = scratch.gdnConvolved.view(offset, uint64_t(length) * 8192 * 2);
      Tensor source = mixed.view(offset, uint64_t(length) * 8192 * 2);
      if (layout.queries[row].draft)
        commands.dispatch(model.device.pipeline("gdn_causal_conv_candidates"), size(uint64_t(length) * 8192), size(256),
                          {destination, state.conv[0], state.conv[1],
                           state.candidateConv.view(uint64_t(layout.state_start_loc[row]) * convStateBytes,
                                                    uint64_t(length) * convStateBytes), source,
                           weight.conv, slots, banks, valid},
                          int64_t(1), int64_t(length));
      else
        commands.dispatch(model.device.pipeline("gdn_causal_conv_silu"), size(uint64_t(8192) * std::max(length, 4u)), size(256),
                          {destination, state.conv[0], state.conv[1], source, weight.conv, slots, banks, valid}, int64_t(1), int64_t(length));
    }
    uint64_t elements = uint64_t(rows) * 4096;
    commands.dispatch(model.device.pipeline("split_repeat_qk"), size(roundUp(elements, 256)), size(256),
                      {scratch.gdnQ, scratch.gdnK, scratch.gdnV, scratch.gdnConvolved}, rows);
    for (uint32_t row = 0; row < layout.size; ++row) {
      uint32_t start = layout.query_start_loc[row], length = layout.query_start_loc[row + 1] - start;
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
        commands.dispatch(
            model.device.pipeline("delta_rule_candidates"), size(32 * 512), size(512),
            {destination, state.recurrent[0], state.recurrent[1],
             state.candidateRecurrent.view(uint64_t(layout.state_start_loc[row]) * recurrentStateBytes,
                                           uint64_t(length) * recurrentStateBytes), slots,
             banks, valid, q, k, v, g, beta},
            int64_t(1), int64_t(length), int64_t(32), int64_t(length) * 4096, int64_t(4096), int64_t(128), int64_t(1));
      else if (length == 1)
        commands.dispatch(model.device.pipeline("delta_rule_decode"), size(32 * 512), size(512),
                          {destination, state.recurrent[0], state.recurrent[1], slots, banks, valid, q, k, v, g, beta}, int64_t(1), int64_t(1),
                          int64_t(32), int64_t(4096), int64_t(4096), int64_t(128), int64_t(1));
      else
        commands.dispatch(model.device.pipeline("delta_rule_prefill"), size(128, 32), size(128),
                          {destination, state.recurrent[0], state.recurrent[1], slots, banks, valid, q, k, v, g, beta}, int64_t(1), int64_t(length),
                          int64_t(32), int64_t(length) * 4096, int64_t(4096), int64_t(128), int64_t(1));
    }
    commands.dispatch(model.device.pipeline("rmsnorm_gated_128"), size(128, uint64_t(rows) * 32), size(128),
                      {scratch.gdnNormed, scratch.gdnDelta, z, weight.norm}, 1e-6f);
    return linear(scratch.gdnNormed, weight.out, rows, output, residual);
  }

  Tensor decoder(const Layer& layer, LayerState& state, const Tensor& hidden, Tensor output, uint32_t rows, uint32_t kvLayer, const Tensor& positions,
                 const Batch& layout, uint32_t dflash = unbound) {
    Tensor x = rms(hidden, layer.inputNorm, scratch.inputNorm, rows);
    Tensor mid = dflash != unbound || layer.fullAttention ? attention(layer, x, hidden, scratch.mid, layout.size, rows, kvLayer, positions, dflash)
                                                          : gdn(layer, x, hidden, scratch.mid, state, rows, layout);
    x = rms(mid, layer.postNorm, dflash == unbound ? scratch.postNorm : scratch.inputNorm, rows);
    return mlp(x, layer.mlp, mid, output, rows);
  }

  Tensor mtpInput(const Tensor& ids, const Tensor& hidden, uint32_t rows, uint32_t batch, uint32_t mode) {
    commands.dispatch(model.device.pipeline("q4_k_embed"), size(4096, rows), size(256), {scratch.postNorm, ids, model.embedding}, int64_t(rows),
                      int64_t(4096));
    commands.dispatch(model.device.pipeline("mtp_fuse"), size(256, rows), size(256),
                      {scratch.mtpFused, scratch.postNorm, hidden, model.mtpSeeds, model.sequenceSlots, model.stateBanks, model.batchKvValid,
                       model.queryStartLoc, model.mtp.embeddingNorm, model.mtp.hiddenNorm},
                      batch, rows, mode);
    return linear(scratch.mtpFused, model.mtp.fusion, rows, scratch.hidden[0]);
  }

  void argmax(const Tensor& token, const Tensor& logits, uint32_t rows = 1, uint32_t group = 1, uint32_t stride = 1) {
    commands.dispatch(model.device.pipeline("argmax_logits"), size(256, rows), size(256), {token, logits}, vocabSize, group, stride);
  }

  void sample(const Sequence& sequence, const Tensor& logits, const Tensor& token, float temperature, float topP, int32_t topK) {
    if (temperature <= 0)
      argmax(token, logits);
    else
      commands.dispatch(model.device.pipeline("sample_logits"), size(1), size(1), {token, model.rng.view(uint64_t(sequence.slot) * 8, 8), logits},
                        vocabSize, temperature, topP, topK);
  }
};

uint32_t writeMetadata(Model& model, const Batch& batch, bool drafts = false) {
  std::array<uint32_t, maxBatchSequences> valid{}, slots{}, banks{};
  uint32_t rows = 0;
  for (uint32_t row = 0; row < batch.size; ++row)
    if (!drafts || batch.queries[row].draft) {
      const Query& query = batch.queries[row];
      valid[rows] = query.valid;
      slots[rows] = query.sequence->slot;
      banks[rows++] = query.sequence->bank;
    }
  uint64_t bytes = uint64_t(rows) * 4;
  model.device.write(model.batchKvValid, valid.data(), bytes);
  model.device.write(model.sequenceSlots, slots.data(), bytes);
  model.device.write(model.stateBanks, banks.data(), bytes);
  return rows;
}

void encodeMtp(Ops& ops, const Batch& batch, uint32_t rows) {
  auto& [commands, model, scratch] = ops;
  Tensor hidden = ops.mtpInput(model.inputIds, scratch.targetHidden, rows, batch.size, 1);
  Tensor normalized = ops.rms(hidden, model.mtp.layer.inputNorm, scratch.inputNorm, rows);
  Tensor k = ops.linear(normalized, model.mtp.layer.attention.k, rows, scratch.attnK);
  Tensor v = ops.linear(normalized, model.mtp.layer.attention.v, rows, scratch.attnV);
  commands.dispatch(model.device.pipeline("mtp_store_kv"), size(uint64_t(rows) * 4 * 256), size(256),
                    {model.kv->key(targetKvLayers), model.kv->value(targetKvLayers), k, v, model.sequenceSlots, model.batchKvValid,
                     model.queryStartLoc, model.mtp.layer.attention.kNorm, model.rope},
                    batch.size, rows, model.maxLogicalBlocks * blockTokens);
  for (uint32_t row = 0; row < batch.size; ++row)
    if (!batch.queries[row].draft)
      commands.copy(scratch.targetHidden.view(uint64_t(batch.query_start_loc[row + 1] - 1) * 4096 * 2, 4096 * 2),
                    mtpSeed(model, *batch.queries[row].sequence, 1 - batch.queries[row].sequence->bank), 4096 * 2);
}

void encodeDflash(Ops& ops, const Batch& batch, uint32_t rows) {
  auto& [commands, model, scratch] = ops;
  Tensor fused = ops.linear(scratch.dflashFeatures, model.dflash.fusion, rows, scratch.dflashContext);
  Tensor context = ops.rms(fused, model.dflash.hiddenNorm, scratch.postNorm, rows);
  for (uint32_t index = 0; index < dflashLayers; ++index) {
    const AttentionWeights& attention = model.dflash.layers[index].attention;
    Tensor k = ops.linear(context, attention.k, rows, scratch.attnK);
    Tensor v = ops.linear(context, attention.v, rows, scratch.attnV);
    commands.dispatch(model.device.pipeline("dflash_store_kv"), size(uint64_t(rows) * 8 * 128), size(128),
                      {model.kv->key(targetKvLayers + index), model.kv->value(targetKvLayers + index), k, v, model.sequenceSlots, model.batchKvValid,
                       model.queryStartLoc, attention.kNorm, model.dflashRope},
                      batch.size, rows, model.maxLogicalBlocks * blockTokens);
  }
}
} // namespace

void Scratch::ensure(Device& device, uint32_t requested, Drafter drafter) {
  // One grow-only private arena backs every activation, with aliases only where lifetimes do not overlap.
  if (!padRows)
    padRows = device.pipeline("pad_rows");
  uint32_t target = roundUp(requested, 32);
  if (uint64_t(target) * 4096 * 2 <= hidden[0].bytes)
    return;
  uint64_t rows = target, hiddenBytes = rows * 4096 * 2, mlpBytes = rows * 12288 * 2;
  uint64_t padBytes = rows * (drafter == Drafter::dflash ? 32768 : 12288) * 2;
  Tensor shared;
  std::vector<std::pair<Tensor*, uint64_t>> slots = {{&shared, padBytes + 2 * mlpBytes},
                                                     {&hidden[0], hiddenBytes},
                                                     {&hidden[1], hiddenBytes},
                                                     {&inputNorm, hiddenBytes},
                                                     {&postNorm, hiddenBytes},
                                                     {&mlpMixed, mlpBytes},
                                                     {&attnQG, rows * 8192 * 2},
                                                     {&attnK, rows * 1024 * 2},
                                                     {&attnV, rows * 1024 * 2},
                                                     {&attnQRope, hiddenBytes},
                                                     {&attnKRope, rows * 1024 * 2},
                                                     {&attnOut, hiddenBytes},
                                                     {&attnGated, hiddenBytes},
                                                     {&attnPartials, std::max<uint64_t>(rows, attentionConcurrency) * 32 * 130 * 4},
                                                     {&gdnMixed, rows * 8192 * 2},
                                                     {&gdnZ, hiddenBytes},
                                                     {&gdnB, rows * 32 * 2},
                                                     {&gdnG, rows * 32 * 4},
                                                     {&gdnConvolved, rows * 8192 * 2},
                                                     {&gdnQ, hiddenBytes},
                                                     {&gdnK, hiddenBytes},
                                                     {&gdnV, hiddenBytes},
                                                     {&gdnDelta, hiddenBytes},
                                                     {&gdnNormed, hiddenBytes},
                                                     {&mid, hiddenBytes},
                                                     {&projected, hiddenBytes},
                                                     {&mtpFused, drafter == Drafter::mtp ? rows * 8192 * 2 : 0},
                                                     {&targetHidden, drafter == Drafter::mtp ? hiddenBytes : 0},
                                                     {&dflashFeatures, drafter == Drafter::dflash ? rows * 32768 * 2 : 0},
                                                     {&dflashContext, drafter == Drafter::dflash ? hiddenBytes : 0},
                                                     {&targetLogits, std::min<uint64_t>(rows, maxLogitRows) * vocabSize * 2}};
  uint64_t bytes = 0;
  for (auto [_, count] : slots)
    bytes = roundUp(bytes, 256) + count;
  Tensor storage = device.empty(bytes);
  uint64_t offset = 0;
  for (auto [tensor, count] : slots) {
    offset = roundUp(offset, 256);
    *tensor = storage.view(offset, count);
    offset += count;
  }
  padInput = shared.view(0, padBytes);
  mlpGate = shared.view(padBytes, mlpBytes);
  mlpUp = shared.view(padBytes + mlpBytes, mlpBytes);
}

void draftDflash(Model& model, Batch& batch) {
  Batch layout;
  std::array<int32_t, maxBatchTokens> inputs{};
  std::array<uint32_t, maxLogitRows> logitRows{};
  uint32_t packed = 0, proposalRows = 0, active = 0;
  for (uint32_t i = 0; i < batch.size; ++i)
    if (batch.queries[i].draft) {
      Query& query = batch.queries[i];
      layout.query_start_loc[active] = packed;
      inputs[packed++] = query.tokens[0];
      for (uint32_t step = 0; step < batch.drafts; ++step) {
        inputs[packed] = dflashMaskToken;
        logitRows[proposalRows++] = packed++;
      }
      layout.query_start_loc[++active] = packed;
    }
  layout.size = writeMetadata(model, batch, true);
  model.device.write(model.queryStartLoc, layout.query_start_loc.data(), uint64_t(layout.size + 1) * 4);
  model.device.write(model.inputIds, inputs.data(), uint64_t(packed) * 4);
  model.device.write(model.logitRows, logitRows.data(), uint64_t(proposalRows) * 4);
  Scratch& scratch = model.scratch(batch.drafts + 1, packed);
  CommandBuffer commands(model.device, 1 << 18, true, 160, "dflash_draft");
  Ops ops{commands, model, scratch};
  commands.dispatch(model.device.pipeline("q4_k_embed"), size(4096, packed), size(256), {scratch.hidden[0], model.inputIds, model.embedding},
                    int64_t(packed), int64_t(4096));
  Tensor hidden = scratch.hidden[0];
  for (uint32_t index = 0; index < dflashLayers; ++index)
    hidden = ops.decoder(model.dflash.layers[index], model.states[0], hidden, scratch.hidden[(index + 1) & 1], packed, targetKvLayers + index,
                         model.batchKvValid, layout, index);
  commands.dispatch(model.device.pipeline("gather_rows"), size(roundUp(uint64_t(proposalRows) * 4096, 256)), size(256),
                    {scratch.postNorm, hidden, model.logitRows}, proposalRows, uint32_t(4096));
  Tensor normalized = ops.rms(scratch.postNorm, model.dflash.outputNorm, scratch.inputNorm, proposalRows);
  Tensor logits = ops.linear(normalized, model.head, proposalRows, scratch.targetLogits);
  ops.argmax(model.draftTokens.view(uint64_t(maxBatchSequences) * 4, model.draftTokens.bytes - uint64_t(maxBatchSequences) * 4), logits, proposalRows,
             batch.drafts, maxBatchSequences);
  commands.commit();
}

// The selected drafter proposes tokens; target verification remains the only commit authority.
void draft(Model& model, Batch& batch) {
  if (model.hasDflash()) {
    draftDflash(model, batch);
    return;
  }
  std::array<int32_t, maxBatchSequences> anchors{};
  std::array<uint32_t, maxBatchSequences * maxDraftTokens> positions{};
  for (uint32_t i = 0; i < batch.size; ++i)
    if (batch.queries[i].draft) {
      Query& query = batch.queries[i];
      uint32_t row = batch.state_start_loc[i] / query.count;
      anchors[row] = query.tokens[0];
      for (uint32_t step = 0; step < batch.drafts; ++step)
        positions[step * maxBatchSequences + row] = query.valid + step;
    }
  uint32_t rows = writeMetadata(model, batch, true);
  Batch layout;
  layout.size = rows;
  for (uint32_t row = 0; row < rows; ++row)
    layout.query_start_loc[row + 1] = row + 1;
  model.device.write(model.queryStartLoc, layout.query_start_loc.data(), uint64_t(rows + 1) * 4);
  model.device.write(model.draftTokens, anchors.data(), uint64_t(rows) * 4);
  model.device.write(model.draftPositions, positions.data(), sizeof(positions));
  Scratch& scratch = model.scratch(1, rows);
  CommandBuffer commands(model.device, 1 << 18, true, 128, "mtp_draft");
  Ops ops{commands, model, scratch};
  for (uint32_t step = 0; step < batch.drafts; ++step) {
    Tensor ids = model.draftTokens.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4);
    Tensor stepPositions = model.draftPositions.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4);
    Tensor hidden = ops.mtpInput(ids, scratch.inputNorm, rows, rows, step ? 0 : 2);
    hidden = ops.decoder(model.mtp.layer, model.states[0], hidden, scratch.hidden[1], rows, targetKvLayers, stepPositions, layout);
    Tensor normalized = ops.rms(hidden, model.mtp.outputNorm, scratch.inputNorm, rows);
    Tensor logits = ops.linear(normalized, model.head, rows, scratch.targetLogits);
    ops.argmax(model.draftTokens.view(uint64_t(step + 1) * maxBatchSequences * 4, uint64_t(rows) * 4), logits, rows);
  }
  commands.commit();
}

// Every target pass consumes one packed variable-length layout and samples only requested terminal rows.
void forward(Model& model, Batch& batch, float temperature, float topP, int32_t topK) {
  uint32_t rows = batch.query_start_loc[batch.size], maxQuery = 0, logitCount = 0;
  std::array<int32_t, maxBatchTokens> tokens{};
  std::array<uint32_t, maxLogitRows> logitRows{};
  auto* proposed = model.draftTokens.contents<int32_t>();
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    if (query.draft) {
      tokens[batch.query_start_loc[row]] = query.tokens[0];
      uint32_t draftRow = batch.state_start_loc[row] / query.count;
      for (uint32_t step = 0; step < batch.drafts; ++step)
        tokens[batch.query_start_loc[row] + step + 1] = proposed[(step + 1) * maxBatchSequences + draftRow];
    } else
      std::copy_n(query.tokens, query.count, tokens.begin() + batch.query_start_loc[row]);
    maxQuery = std::max(maxQuery, query.count);
    if (!query.sample)
      continue;
    query.logit = logitCount;
    uint32_t count = query.draft ? query.count : 1;
    for (uint32_t i = 0; i < count; ++i)
      logitRows[logitCount++] = batch.query_start_loc[row] + (query.draft ? i : query.count - 1);
  }
  model.device.write(model.inputIds, tokens.data(), uint64_t(rows) * 4);
  if (logitCount)
    model.device.write(model.logitRows, logitRows.data(), uint64_t(logitCount) * 4);
  writeMetadata(model, batch);
  model.device.write(model.queryStartLoc, batch.query_start_loc.data(), uint64_t(batch.size + 1) * 4);
  model.ensureCandidates(batch.state_start_loc[batch.size]);
  Scratch& scratch = model.scratch(maxQuery, rows);
  const char* phase = maxQuery == 1 ? "decode" : batch.drafts ? "verify" : "prefill";
  CommandBuffer commands(model.device, uint64_t(512 + 48 * batch.size) * 128, true, 1600 + 24 * batch.size, phase);
  Ops ops{commands, model, scratch};
  commands.dispatch(model.device.pipeline("q4_k_embed"), size(4096, rows), size(256), {scratch.hidden[0], model.inputIds, model.embedding},
                    int64_t(rows), int64_t(4096));
  Tensor hidden = scratch.hidden[0];
  for (uint32_t i = 0; i < model.layers.size(); ++i) {
    hidden =
        ops.decoder(model.layers[i], model.states[i], hidden, scratch.hidden[(i + 1) & 1], rows, model.layers[i].kvIndex, model.batchKvValid, batch);
    if (model.hasDflash() && i % 4 == 1)
      commands.dispatch(model.device.pipeline("capture_hidden"), size(roundUp(uint64_t(rows) * 4096, 256)), size(256),
                        {scratch.dflashFeatures, hidden}, (i - 1) / 4, rows);
  }
  if (model.hasMtp())
    commands.copy(hidden, scratch.targetHidden.view(0, uint64_t(rows) * 4096 * 2), uint64_t(rows) * 4096 * 2);
  if (logitCount) {
    Tensor input = hidden;
    if (logitCount != rows) {
      commands.dispatch(model.device.pipeline("gather_rows"), size(roundUp(uint64_t(logitCount) * 4096, 256)), size(256),
                        {scratch.postNorm, hidden, model.logitRows}, logitCount, uint32_t(4096));
      input = scratch.postNorm;
    }
    Tensor normalized = ops.rms(input, model.norm, scratch.inputNorm, logitCount);
    Tensor logits = ops.linear(normalized, model.head, logitCount, scratch.targetLogits);
    for (uint32_t row = 0; row < batch.size; ++row)
      if (batch.queries[row].sample) {
        Query& query = batch.queries[row];
        uint32_t count = query.draft ? query.count : 1;
        for (uint32_t i = 0; i < count; ++i) {
          uint32_t result = query.logit + i;
          ops.sample(*query.sequence, logits.view(uint64_t(result) * vocabSize * 2, uint64_t(vocabSize) * 2),
                     model.outputTokens.view(uint64_t(result) * 4, 4), temperature, topP, topK);
        }
      }
  }
  if (model.hasMtp())
    encodeMtp(ops, batch, rows);
  else if (model.hasDflash())
    encodeDflash(ops, batch, rows);
  commands.commit();
}

} // namespace infeng::qwen35
