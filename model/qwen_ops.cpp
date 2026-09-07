#include "qwen35.hpp"
#include <algorithm>
#include <numeric>
namespace infeng::qwen35 {
namespace {
constexpr uint32_t attentionConcurrency = 8;
MTL::Size size(uint64_t x, uint64_t y = 1, uint64_t z = 1) {
  return MTL::Size(x, y, z);
}
struct Ops {
  Device& commands;
  Engine& model;
  Scratch& scratch;
  Tensor linear(const Tensor& x, const Linear& weight, uint32_t rows, Tensor output, const Tensor& residual = {}) {
    if (scratch.decodeMode) {
      uint32_t group = weight.pipeline[linearDecode]->maxTotalThreadsPerThreadgroup();
      MTL::Size threads = size((weight.n + weight.outputsPerGroup - 1) / weight.outputsPerGroup * group, rows);
      if (residual.buffer)
        commands.dispatch(weight.pipeline[linearDecodeAdd], threads, size(group), {output, x, weight.weight, residual});
      else
        commands.dispatch(weight.pipeline[linearDecode], threads, size(group), {output, x, weight.weight});
      return output;
    }
    bool small = rows <= 8;
    bool add = bool(residual.buffer);
    Pipeline* pipeline = weight.pipeline[small ? linearPrefillSmall : linearPrefill];
    MTL::Size group = size(small ? 64 : 512);
    MTL::Size threads = size(group.width * (weight.n / (small ? 8 : 32)), (rows + 127) / 128);
    commands.dispatch(pipeline, threads, group, {output, x, weight.weight, add ? residual : output}, int64_t(rows), uint32_t(add));
    return output.view(0, uint64_t(rows) * weight.n * 2);
  }
  Tensor rms(const Tensor& x, const Tensor& weight, Tensor output, uint32_t rows) {
    commands.dispatch("rmsnorm", size(256, rows), size(256), {output, x, weight});
    return output;
  }
  Tensor embed(const Tensor& ids, Tensor output, uint32_t rows) {
    commands.dispatch("q4_k_embed", size(4096, rows), size(256), {output, ids, model.embedding});
    return output;
  }
  Tensor logits(const Tensor& x, const Tensor& norm, uint32_t rows) {
    return linear(rms(x, norm, scratch.inputNorm, rows), model.head, rows, scratch.targetLogits);
  }
  Tensor mlp(const Tensor& x, const MlpWeights& weights, const Tensor& residual, Tensor output, uint32_t rows) {
    if (scratch.decodeMode)
      commands.dispatch(weights.fusedDecode, size(12288 / weights.outputsPerGroup * weights.fusedDecode->maxTotalThreadsPerThreadgroup(), rows),
                        size(weights.fusedDecode->maxTotalThreadsPerThreadgroup()), {scratch.mlpGate, x, weights.gate.weight, weights.up.weight},
                        int64_t(rows));
    else {
      Tensor gate = linear(x, weights.gate, rows, scratch.mlpGate), up = linear(x, weights.up, rows, scratch.mlpUp);
      commands.dispatch("silu_mul", size(uint64_t(rows) * 12288), size(256), {scratch.mlpGate, gate, up});
    }
    return linear(scratch.mlpGate, weights.down, rows, output, residual);
  }
  Tensor attention(const Layer& layer, const Tensor& x, const Tensor& residual, Tensor output, uint32_t batch, uint32_t rows, uint32_t kvLayer,
                   const Tensor& positions, uint32_t dflash = unbound) {
    const AttentionWeights& weight = layer.attention;
    Tensor qg = linear(x, weight.q, rows, scratch.attnQG), k = linear(x, weight.k, rows, scratch.attnK);
    Tensor v = linear(x, weight.v, rows, scratch.attnV);
    bool draft = dflash != unbound;
    uint32_t heads = draft ? 32 : 16, group = draft ? 128 : 256, kvLayerIndex = draft ? targetKvLayers + dflash : kvLayer;
    Tensor rope = draft ? model.dflashRope : model.rope;
    commands.dispatch(draft ? "dflash_attention_prepare" : "attention_prepare", size(group, uint64_t(rows) * (draft ? 40 : 20)), size(group),
                      {scratch.attnQRope, scratch.attnKRope, qg, k, weight.qNorm, weight.kNorm, rope, positions, model.queryStartLoc}, batch);
    uint32_t splits = std::max(1u, attentionConcurrency / rows);
    commands.dispatch(draft ? "dflash_attention_scan" : "attention_scan", size(uint64_t(rows) * heads * splits * 128), size(128),
                      {scratch.attnPartials, scratch.attnQRope, scratch.attnKRope, v, model.kv->key(kvLayerIndex), model.kv->value(kvLayerIndex),
                       model.sequenceSlots, positions, model.queryStartLoc},
                      batch, rows, uint32_t(model.blocks.size()) * blockTokens, splits, uint32_t(draft && dflash + 1 < dflashLayers));
    commands.dispatch(draft ? "dflash_attention_reduce" : "attention_reduce", size(uint64_t(rows) * heads * 128), size(128),
                      {scratch.attnOut, scratch.attnPartials, qg}, rows, splits);
    return linear(scratch.attnOut, weight.out, rows, output, residual);
  }
  Tensor gdn(const Layer& layer, const Tensor& x, const Tensor& residual, Tensor output, uint32_t rows, const Batch& layout) {
    const GdnWeights& weight = layer.gdn;
    uint32_t layerIndex = &layer - model.layers.data();
    GdnState bank0 = gdnState(model.gdnStates[0], maxBatchSequences, layerIndex);
    GdnState bank1 = gdnState(model.gdnStates[1], maxBatchSequences, layerIndex);
    GdnState candidates = gdnState(model.candidateStates, model.candidateStates.bytes / gdnCheckpointBytes, layerIndex);
    auto* starts = model.queryStartLoc.contents<uint32_t>();
    Tensor mixed = linear(x, weight.qkv, rows, scratch.gdnMixed), z = linear(x, weight.z, rows, scratch.gdnZ);
    commands.dispatch("gdn_ba_prepare_4096x32", size(32 * 64, rows), size(64),
                      {scratch.gdnB, scratch.gdnG, x, weight.b, weight.a, weight.A, weight.dt}, weight.b.bytes == uint64_t(32) * 4096 * 4);
    for (uint32_t row = 0; row < layout.size; ++row) {
      const Query& query = layout.queries[row];
      uint32_t start = starts[row], length = starts[row + 1] - start;
      uint64_t offset = uint64_t(start) * 8192 * 2;
      const Sequence& sequence = *query.sequence;
      Tensor destination = scratch.gdnConvolved.view(offset, uint64_t(length) * 8192 * 2), source = mixed.view(offset, destination.bytes);
      bool draft = query.state != unbound;
      Tensor candidate = draft ? candidates.conv.view(uint64_t(query.state) * convStateBytes, uint64_t(length) * convStateBytes) : destination;
      commands.dispatch(draft ? "gdn_causal_conv_candidates" : "gdn_causal_conv_silu", size(uint64_t(8192) * (draft ? length : std::max(length, 4u))),
                        size(256), {destination, bank0.conv, bank1.conv, candidate, source, weight.conv}, sequence.slot, sequence.bank,
                        uint32_t(sequence.kvValid != 0), int64_t(length));
    }
    commands.dispatch("split_repeat_qk", size(uint64_t(rows) * 4096), size(256), {scratch.gdnQ, scratch.gdnK, scratch.gdnV, scratch.gdnConvolved});
    for (uint32_t row = 0; row < layout.size; ++row) {
      const Query& query = layout.queries[row];
      uint32_t start = starts[row], length = starts[row + 1] - start;
      uint64_t offset = uint64_t(start) * 4096 * 2;
      const Sequence& sequence = *query.sequence;
      Tensor destination = scratch.gdnMixed.view(offset, uint64_t(length) * 4096 * 2);
      Tensor q = scratch.gdnQ.view(offset, uint64_t(length) * 4096 * 2), k = scratch.gdnK.view(offset, uint64_t(length) * 4096 * 2),
             v = scratch.gdnV.view(offset, uint64_t(length) * 4096 * 2);
      Tensor g = scratch.gdnG.view(start * 128ull, length * 128ull), beta = scratch.gdnB.view(start * 64ull, length * 64ull);
      bool draft = query.state != unbound;
      Tensor candidate =
          draft ? candidates.recurrent.view(uint64_t(query.state) * recurrentStateBytes, uint64_t(length) * recurrentStateBytes) : destination;
      const char* kernel = draft ? "delta_rule_candidates" : length == 1 ? "delta_rule_decode" : "delta_rule_prefill";
      commands.dispatch(kernel, draft || length < 2 ? size(32 * 512) : size(128, 32), size(draft || length < 2 ? 512 : 128),
                        {destination, bank0.recurrent, bank1.recurrent, candidate, q, k, v, g, beta}, sequence.slot, sequence.bank,
                        uint32_t(sequence.kvValid != 0), int64_t(length));
    }
    commands.dispatch("rmsnorm_gated_128", size(128, uint64_t(rows) * 32), size(128), {scratch.gdnQ, scratch.gdnMixed, z, weight.norm});
    return linear(scratch.gdnQ, weight.out, rows, output, residual);
  }
  Tensor decoder(const Layer& layer, const Tensor& hidden, Tensor output, uint32_t rows, uint32_t kvLayer, const Tensor& positions,
                 const Batch* layout = nullptr, uint32_t batch = 0, uint32_t dflash = unbound) {
    Tensor x = rms(hidden, layer.inputNorm, scratch.inputNorm, rows);
    uint32_t sequences = layout ? layout->size : batch ? batch : rows;
    Tensor mid = dflash != unbound || layer.attention.q.weight.buffer
                     ? attention(layer, x, hidden, scratch.mid, sequences, rows, kvLayer, positions, dflash)
                     : gdn(layer, x, hidden, scratch.mid, rows, *layout);
    x = rms(mid, layer.postNorm, dflash == unbound ? scratch.postNorm : scratch.inputNorm, rows);
    return mlp(x, layer.mlp, mid, output, rows);
  }
  Tensor mtpInput(const Tensor& ids, const Tensor& hidden, uint32_t rows, uint32_t batch, uint32_t mode) {
    embed(ids, scratch.postNorm, rows);
    commands.dispatch("mtp_fuse", size(256, rows), size(256),
                      {scratch.gdnMixed, scratch.postNorm, hidden, model.mtpSeeds, model.sequenceSlots, model.stateBanks, model.batchKvValid,
                       model.queryStartLoc, model.draftModel.embeddingNorm, model.draftModel.hiddenNorm},
                      batch, rows, mode);
    return linear(scratch.gdnMixed, model.draftModel.fusion, rows, scratch.hidden[0]);
  }
  void argmax(const Tensor& token, const Tensor& logits, uint32_t rows = 1, uint32_t group = 1, uint32_t stride = 1) {
    commands.dispatch("argmax_logits", size(256, rows), size(256), {token, logits}, group, stride);
  }
  void sample(const Sequence& sequence, const Tensor& logits, const Tensor& tokens, uint32_t count) {
    if (sequence.temperature <= 0)
      argmax(tokens, logits, count);
    else
      commands.dispatch("sample_logits", size(count), size(1),
                        {tokens, model.rng.view(uint64_t(sequence.slot) * 8, 8),
                         model.sampledRng.view((tokens.offset - model.outputTokens.offset) * 2, uint64_t(count) * 8), logits},
                        sequence.temperature, sequence.topP, sequence.topK);
  }
};
uint32_t writeMetadata(Engine& model, const Batch& batch, bool drafts = false) {
  auto *valid = model.batchKvValid.contents<uint32_t>(), *slots = model.sequenceSlots.contents<uint32_t>(),
       *banks = model.stateBanks.contents<uint32_t>();
  uint32_t rows = 0;
  for (uint32_t row = 0; row < batch.size; ++row)
    if (!drafts || batch.queries[row].state != unbound) {
      const Query& query = batch.queries[row];
      valid[rows] = query.sequence->kvValid;
      slots[rows] = query.sequence->slot;
      banks[rows++] = query.sequence->bank;
    }
  return rows;
}
void encodeDrafter(Ops& ops, const Batch& batch, uint32_t rows) {
  auto& [commands, model, scratch] = ops;
  if (model.drafter == Drafter::mtp) {
    Tensor hidden = ops.mtpInput(model.inputIds, scratch.targetHidden, rows, batch.size, 1);
    Tensor normalized = ops.rms(hidden, model.draftModel.layers[0].inputNorm, scratch.inputNorm, rows);
    Tensor k = ops.linear(normalized, model.draftModel.layers[0].attention.k, rows, scratch.attnK),
           v = ops.linear(normalized, model.draftModel.layers[0].attention.v, rows, scratch.attnV);
    commands.dispatch("mtp_store_kv", size(uint64_t(rows) * 1024), size(256),
                      {model.kv->key(targetKvLayers), model.kv->value(targetKvLayers), k, v, model.sequenceSlots, model.batchKvValid,
                       model.queryStartLoc, model.draftModel.layers[0].attention.kNorm, model.rope},
                      batch.size, rows, uint32_t(model.blocks.size()) * blockTokens);
    for (uint32_t row = 0; row < batch.size; ++row)
      if (batch.queries[row].state == unbound)
        commands.copy(scratch.targetHidden.view(uint64_t(model.queryStartLoc.contents<uint32_t>()[row + 1] - 1) * 8192, 8192),
                      mtpSeed(model, *batch.queries[row].sequence, 1 - batch.queries[row].sequence->bank));
  } else {
    Tensor context = ops.rms(ops.linear(scratch.dflashFeatures, model.draftModel.fusion, rows, scratch.hidden[0]), model.draftModel.hiddenNorm,
                             scratch.postNorm, rows);
    for (uint32_t index = 0; index < dflashLayers; ++index) {
      const AttentionWeights& attention = model.draftModel.layers[index].attention;
      Tensor k = ops.linear(context, attention.k, rows, scratch.attnK), v = ops.linear(context, attention.v, rows, scratch.attnV);
      commands.dispatch("dflash_store_kv", size(uint64_t(rows) * 1024), size(128),
                        {model.kv->key(targetKvLayers + index), model.kv->value(targetKvLayers + index), k, v, model.sequenceSlots,
                         model.batchKvValid, model.queryStartLoc, attention.kNorm, model.dflashRope},
                        batch.size, rows, uint32_t(model.blocks.size()) * blockTokens);
    }
  }
}
} // namespace
void Scratch::ensure(Device& device, uint32_t requested, Drafter drafter) {
  uint32_t target = (requested + 31) / 32 * 32;
  if (uint64_t(target) * 4096 * 2 <= hidden[0].bytes)
    return;
  uint64_t rows = target, hiddenBytes = rows * 4096 * 2, mlpBytes = rows * 12288 * 2;
  uint64_t kvBytes = rows * 1024 * 2;
  uint64_t partialBytes = std::max<uint64_t>(rows, attentionConcurrency) * 32 * 130 * 4;
  uint64_t extraBytes = drafter == Drafter::mtp ? hiddenBytes : drafter == Drafter::dflash ? 8 * hiddenBytes : 0;
  uint64_t logitBytes = std::min<uint64_t>(rows, maxLogitRows) * vocabSize * 2;
  Tensor storage = device.empty(2 * mlpBytes + 17 * hiddenBytes + 3 * kvBytes + rows * 32 * 6 + partialBytes + extraBytes + logitBytes);
  uint64_t offset = 0;
  auto take = [&](uint64_t bytes, auto&... tensors) { ((tensors = storage.view(offset, bytes), offset += bytes), ...); };
  take(mlpBytes, mlpGate, mlpUp);
  take(hiddenBytes, hidden[0], hidden[1], inputNorm, postNorm);
  take(2 * hiddenBytes, attnQG);
  take(kvBytes, attnK, attnV);
  take(hiddenBytes, attnQRope);
  take(kvBytes, attnKRope);
  take(hiddenBytes, attnOut);
  take(partialBytes, attnPartials);
  take(2 * hiddenBytes, gdnMixed);
  take(hiddenBytes, gdnZ);
  take(rows * 32 * 2, gdnB);
  take(rows * 32 * 4, gdnG);
  take(2 * hiddenBytes, gdnConvolved);
  take(hiddenBytes, gdnQ, gdnK, gdnV, mid);
  take(drafter == Drafter::mtp ? hiddenBytes : 0, targetHidden);
  take(drafter == Drafter::dflash ? 8 * hiddenBytes : 0, dflashFeatures);
  take(logitBytes, targetLogits);
}
void draft(Engine& model, Batch& batch, uint32_t drafts) {
  bool dflash = model.drafter == Drafter::dflash;
  uint32_t rows = writeMetadata(model, batch, true), packed = rows * (dflash ? drafts + 1 : 1);
  auto* starts = model.queryStartLoc.contents<uint32_t>();
  for (uint32_t i = 0; i < batch.size; ++i)
    if (batch.queries[i].state != unbound) {
      Query& query = batch.queries[i];
      uint32_t row = query.state / query.count;
      if (dflash) {
        uint32_t start = row * (drafts + 1);
        model.inputIds.contents<int32_t>()[start] = query.sequence->request[query.sequence->kvValid];
        std::fill_n(model.inputIds.contents<int32_t>() + start + 1, drafts, dflashMaskToken);
        std::iota(model.logitRows.contents<uint32_t>() + row * drafts, model.logitRows.contents<uint32_t>() + (row + 1) * drafts, start + 1);
      } else {
        model.draftTokens.contents<int32_t>()[row] = query.sequence->request[query.sequence->kvValid];
        for (uint32_t step = 0; step < drafts; ++step)
          model.draftPositions.contents<uint32_t>()[step * maxBatchSequences + row] = query.sequence->kvValid + step;
      }
    }
  for (uint32_t row = 0; row <= rows; ++row)
    starts[row] = row * (dflash ? drafts + 1 : 1);
  Scratch& scratch = model.scratch(dflash ? drafts + 1 : 1, packed);
  Device& commands = model.device.command();
  Ops ops{commands, model, scratch};
  if (dflash) {
    Tensor hidden = ops.embed(model.inputIds, scratch.hidden[0], packed);
    for (uint32_t index = 0; index < dflashLayers; ++index)
      hidden = ops.decoder(model.draftModel.layers[index], hidden, scratch.hidden[(index + 1) & 1], packed, targetKvLayers + index,
                           model.batchKvValid, nullptr, rows, index);
    uint32_t proposals = rows * drafts;
    commands.dispatch("gather_rows", size(uint64_t(proposals) * 4096), size(256), {scratch.postNorm, hidden, model.logitRows});
    Tensor logits = ops.logits(scratch.postNorm, model.draftModel.outputNorm, proposals);
    ops.argmax(model.draftTokens.view(uint64_t(maxBatchSequences) * 4, model.draftTokens.bytes - uint64_t(maxBatchSequences) * 4), logits, proposals,
               drafts, maxBatchSequences);
  } else {
    for (uint32_t step = 0; step < drafts; ++step) {
      Tensor ids = model.draftTokens.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4);
      Tensor positions = model.draftPositions.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4);
      Tensor hidden = ops.mtpInput(ids, scratch.inputNorm, rows, rows, step ? 0 : 2);
      hidden = ops.decoder(model.draftModel.layers[0], hidden, scratch.hidden[1], rows, targetKvLayers, positions);
      Tensor logits = ops.logits(hidden, model.draftModel.outputNorm, rows);
      ops.argmax(model.draftTokens.view(uint64_t(step + 1) * maxBatchSequences * 4, uint64_t(rows) * 4), logits, rows);
    }
  }
  commands.commit();
}
void forward(Engine& model, Batch& batch, uint32_t drafts, uint32_t stateRows) {
  auto* starts = model.queryStartLoc.contents<uint32_t>();
  starts[0] = 0;
  for (uint32_t row = 0; row < batch.size; ++row)
    starts[row + 1] = starts[row] + batch.queries[row].count;
  uint32_t rows = starts[batch.size], maxQuery = 0, logitCount = 0;
  auto *tokens = model.inputIds.contents<int32_t>(), *proposed = model.draftTokens.contents<int32_t>();
  auto* logitRows = model.logitRows.contents<uint32_t>();
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    if (query.state != unbound) {
      tokens[starts[row]] = query.sequence->request[query.sequence->kvValid];
      uint32_t draftRow = query.state / query.count;
      for (uint32_t step = 0; step < drafts; ++step)
        tokens[starts[row] + step + 1] = proposed[(step + 1) * maxBatchSequences + draftRow];
    } else
      std::copy_n(query.sequence->request.data() + query.sequence->kvValid, query.count, tokens + starts[row]);
    maxQuery = std::max(maxQuery, query.count);
    if (query.logit == unbound)
      continue;
    query.logit = logitCount;
    uint32_t count = query.state != unbound ? query.count : 1;
    for (uint32_t i = 0; i < count; ++i)
      logitRows[logitCount++] = starts[row] + (query.state != unbound ? i : query.count - 1);
  }
  writeMetadata(model, batch);
  if (gdnCheckpointBytes * stateRows > model.candidateStates.bytes)
    model.candidateStates = model.device.empty(gdnCheckpointBytes * stateRows);
  Scratch& scratch = model.scratch(maxQuery, rows);
  Device& commands = model.device.command();
  Ops ops{commands, model, scratch};
  Tensor hidden = ops.embed(model.inputIds, scratch.hidden[0], rows);
  for (uint32_t i = 0; i < model.layers.size(); ++i) {
    hidden = ops.decoder(model.layers[i], hidden, scratch.hidden[(i + 1) & 1], rows, i / fullAttentionInterval, model.batchKvValid, &batch);
    if (model.drafter == Drafter::dflash && i % 4 == 1)
      commands.dispatch("capture_hidden", size(uint64_t(rows) * 4096), size(256), {scratch.dflashFeatures, hidden}, (i - 1) / 4);
  }
  if (model.drafter == Drafter::mtp)
    commands.copy(hidden, scratch.targetHidden.view(0, uint64_t(rows) * 4096 * 2));
  if (logitCount) {
    Tensor input = hidden;
    if (logitCount != rows) {
      commands.dispatch("gather_rows", size(uint64_t(logitCount) * 4096), size(256), {scratch.postNorm, hidden, model.logitRows});
      input = scratch.postNorm;
    }
    Tensor logits = ops.logits(input, model.norm, logitCount);
    for (uint32_t row = 0; row < batch.size; ++row)
      if (batch.queries[row].logit != unbound) {
        Query& query = batch.queries[row];
        uint32_t count = query.state != unbound ? query.count : 1;
        ops.sample(*query.sequence, logits.view(uint64_t(query.logit) * vocabSize * 2, uint64_t(count) * vocabSize * 2),
                   model.outputTokens.view(uint64_t(query.logit) * 4, uint64_t(count) * 4), count);
      }
  }
  if (model.drafter != Drafter::none)
    encodeDrafter(ops, batch, rows);
  commands.commit();
}
} // namespace infeng::qwen35
