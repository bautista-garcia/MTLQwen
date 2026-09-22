#include "model/qwen35/qwen35.hpp"
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
  Engine& engine;
  Scratch& scratch;
  bool decode;

  Tensor linear(const Tensor& x, const Linear& weight, uint32_t rows, Tensor output, const Tensor& residual = {}) {
    if (decode) {
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
    return output;
  }

  Tensor logits(const Tensor& x, const Tensor& norm, uint32_t rows) {
    commands.dispatch("rmsnorm", size(256, rows), size(256), {scratch.temporary, x, norm});
    return linear(scratch.temporary, engine.head, rows, scratch.targetLogits);
  }

  Tensor mlp(const Tensor& x, const MlpWeights& weights, const Tensor& residual, Tensor output, uint32_t rows) {
    if (decode)
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
    Tensor qg = linear(x, weight.q, rows, scratch.mixed), k = linear(x, weight.k, rows, scratch.k);
    Tensor v = linear(x, weight.v, rows, scratch.v);
    bool draft = dflash != unbound;
    uint32_t heads = draft ? 32 : 16, group = draft ? 128 : 256;
    const Tensor& rope = draft ? engine.dflashRope : engine.rope;
    commands.dispatch(draft ? "dflash_attention_prepare" : "attention_prepare", size(group, uint64_t(rows) * (draft ? 40 : 20)), size(group),
                      {scratch.attnQRope, scratch.attnKRope, qg, k, weight.qNorm, weight.kNorm, rope, positions, engine.queryStartLoc}, batch);
    uint32_t splits = std::max(1u, attentionConcurrency / rows);
    commands.dispatch(draft ? "dflash_attention_scan" : "attention_scan", size(uint64_t(rows) * heads * splits * 128), size(128),
                      {scratch.attnPartials, scratch.attnQRope, scratch.attnKRope, v, engine.kv->key(kvLayer), engine.kv->value(kvLayer),
                       engine.sequenceSlots, positions, engine.queryStartLoc},
                      batch, rows, uint32_t(engine.blocks.size()) * blockTokens, splits, uint32_t(draft && dflash + 1 < dflashLayers));
    commands.dispatch(draft ? "dflash_attention_reduce" : "attention_reduce", size(uint64_t(rows) * heads * 128), size(128),
                      {scratch.temporary, scratch.attnPartials, qg}, rows, splits);
    return linear(scratch.temporary, weight.out, rows, output, residual);
  }

  Tensor gdn(const Layer& layer, const Tensor& x, const Tensor& residual, Tensor output, uint32_t rows, const Batch& layout) {
    const GdnWeights& weight = layer.gdn;
    uint32_t layerIndex = &layer - engine.layers.data();
    GdnState bank0 = gdnState(engine.gdnStates[0], maxBatchSequences, layerIndex);
    GdnState bank1 = gdnState(engine.gdnStates[1], maxBatchSequences, layerIndex);
    GdnState candidates = gdnState(engine.candidateStates, engine.candidateStates.bytes / gdnCheckpointBytes, layerIndex);
    Tensor mixed = linear(x, weight.qkv, rows, scratch.mixed), z = linear(x, weight.z, rows, scratch.temporary);
    commands.dispatch("gdn_ba_prepare_4096x32", size(32 * 64, rows), size(64),
                      {scratch.gdnB, scratch.gdnG, x, weight.b, weight.a, weight.A, weight.dt}, weight.b.bytes == uint64_t(32) * 4096 * 4);
    for (uint32_t row = 0; row < layout.size; ++row) {
      const Query& query = layout.queries[row];
      uint64_t offset = uint64_t(query.start) * 8192 * 2;
      const Sequence& sequence = *query.sequence;
      Tensor destination = scratch.gdnConvolved.view(offset, uint64_t(query.count) * 8192 * 2), source = mixed.view(offset, destination.bytes);
      bool draft = query.state != unbound;
      Tensor candidate = draft ? candidates.conv.view(uint64_t(query.state) * convStateBytes, uint64_t(query.count) * convStateBytes) : destination;
      commands.dispatch(draft ? "gdn_causal_conv_candidates" : "gdn_causal_conv_silu",
                        size(uint64_t(8192) * (draft ? query.count : std::max(query.count, 4u))), size(256),
                        {destination, bank0.conv, bank1.conv, candidate, source, weight.conv}, sequence.slot, sequence.bank,
                        uint32_t(sequence.kvValid != 0), int64_t(query.count));
    }
    commands.dispatch("split_repeat_qk", size(uint64_t(rows) * 4096), size(256), {scratch.q, scratch.k, scratch.v, scratch.gdnConvolved});
    for (uint32_t row = 0; row < layout.size; ++row) {
      const Query& query = layout.queries[row];
      uint64_t offset = uint64_t(query.start) * 4096 * 2;
      const Sequence& sequence = *query.sequence;
      Tensor destination = scratch.mixed.view(offset, uint64_t(query.count) * 4096 * 2);
      Tensor q = scratch.q.view(offset, uint64_t(query.count) * 4096 * 2), k = scratch.k.view(offset, uint64_t(query.count) * 4096 * 2),
             v = scratch.v.view(offset, uint64_t(query.count) * 4096 * 2);
      Tensor g = scratch.gdnG.view(query.start * 128ull, query.count * 128ull), beta = scratch.gdnB.view(query.start * 64ull, query.count * 64ull);
      bool draft = query.state != unbound;
      Tensor candidate =
          draft ? candidates.recurrent.view(uint64_t(query.state) * recurrentStateBytes, uint64_t(query.count) * recurrentStateBytes) : destination;
      const char* kernel = draft ? "delta_rule_candidates" : query.count == 1 ? "delta_rule_decode" : "delta_rule_prefill";
      commands.dispatch(kernel, draft || query.count == 1 ? size(32 * 512) : size(128, 32), size(draft || query.count == 1 ? 512 : 128),
                        {destination, bank0.recurrent, bank1.recurrent, candidate, q, k, v, g, beta}, sequence.slot, sequence.bank,
                        uint32_t(sequence.kvValid != 0), int64_t(query.count));
    }
    commands.dispatch("rmsnorm_gated_128", size(128, uint64_t(rows) * 32), size(128), {scratch.q, scratch.mixed, z, weight.norm});
    return linear(scratch.q, weight.out, rows, output, residual);
  }

  Tensor decoder(const Layer& layer, const Tensor& hidden, Tensor output, uint32_t rows, uint32_t batch, uint32_t kvLayer, const Tensor& positions,
                 const Batch* layout = nullptr, uint32_t dflash = unbound) {
    commands.dispatch("rmsnorm", size(256, rows), size(256), {scratch.norm, hidden, layer.inputNorm});
    Tensor mid = layer.attention.q.weight.buffer ? attention(layer, scratch.norm, hidden, scratch.mid, batch, rows, kvLayer, positions, dflash)
                                                 : gdn(layer, scratch.norm, hidden, scratch.mid, rows, *layout);
    commands.dispatch("rmsnorm", size(256, rows), size(256), {scratch.norm, mid, layer.postNorm});
    return mlp(scratch.norm, layer.mlp, mid, output, rows);
  }

  Tensor mtpInput(const Tensor& ids, const Tensor& hidden, uint32_t rows, uint32_t batch, uint32_t mode) {
    commands.dispatch("q4_k_embed", size(4096, rows), size(256), {scratch.norm, ids, engine.embedding});
    commands.dispatch("mtp_fuse", size(256, rows), size(256),
                      {scratch.mixed, scratch.norm, hidden, engine.mtpSeeds, engine.sequenceSlots, engine.stateBanks, engine.batchKvValid,
                       engine.queryStartLoc, engine.draftModel.embeddingNorm, engine.draftModel.hiddenNorm},
                      batch, rows, mode);
    return linear(scratch.mixed, engine.draftModel.fusion, rows, scratch.hidden[0]);
  }

  void argmax(const Tensor& token, const Tensor& logits, uint32_t rows = 1, uint32_t group = 1, uint32_t stride = 1) {
    commands.dispatch("argmax_logits", size(256, rows), size(256), {token, logits}, group, stride);
  }

  void sample(const Sequence& sequence, const Tensor& logits, const Tensor& tokens, uint32_t count) {
    if (sequence.temperature <= 0)
      argmax(tokens, logits, count);
    else
      commands.dispatch("sample_logits", size(count), size(1),
                        {tokens, engine.rng.view(uint64_t(sequence.slot) * 8, 8),
                         engine.sampledRng.view((tokens.offset - engine.outputTokens.offset) * 2, uint64_t(count) * 8), logits},
                        sequence.temperature, sequence.topP, sequence.topK);
  }
};

void writeMetadata(Engine& engine, const Sequence& sequence, uint32_t row) {
  engine.batchKvValid.contents<uint32_t>()[row] = sequence.kvValid;
  engine.sequenceSlots.contents<uint32_t>()[row] = sequence.slot;
  engine.stateBanks.contents<uint32_t>()[row] = sequence.bank;
}

void encodeDrafter(Ops& ops, const Batch& batch, uint32_t rows) {
  auto& [commands, engine, scratch, decode] = ops;
  bool mtp = engine.drafter == Drafter::mtp;
  Tensor context = mtp ? ops.mtpInput(engine.inputIds, scratch.targetHidden, rows, batch.size, 1)
                       : ops.linear(scratch.dflashFeatures, engine.draftModel.fusion, rows, scratch.hidden[0]);
  commands.dispatch("rmsnorm", size(256, rows), size(256),
                    {scratch.norm, context, mtp ? engine.draftModel.layers[0].inputNorm : engine.draftModel.hiddenNorm});
  context = scratch.norm;
  for (uint32_t index = 0; index < (mtp ? 1 : dflashLayers); ++index) {
    const AttentionWeights& attention = engine.draftModel.layers[index].attention;
    Tensor k = ops.linear(context, attention.k, rows, scratch.k), v = ops.linear(context, attention.v, rows, scratch.v);
    commands.dispatch(mtp ? "mtp_store_kv" : "dflash_store_kv", size(uint64_t(rows) * 1024), size(mtp ? 256 : 128),
                      {engine.kv->key(targetKvLayers + index), engine.kv->value(targetKvLayers + index), k, v, engine.sequenceSlots,
                       engine.batchKvValid, engine.queryStartLoc, attention.kNorm, mtp ? engine.rope : engine.dflashRope},
                      batch.size, rows, uint32_t(engine.blocks.size()) * blockTokens);
  }
  if (mtp)
    for (uint32_t row = 0; row < batch.size; ++row)
      if (batch.queries[row].state == unbound)
        commands.copy(scratch.targetHidden.view(uint64_t(batch.queries[row].start + batch.queries[row].count - 1) * 8192, 8192),
                      mtpSeed(engine, *batch.queries[row].sequence, 1 - batch.queries[row].sequence->bank));
}
} // namespace

void Scratch::allocate(Device& device, Drafter drafter) {
  uint64_t rows = maxBatchTokens, hiddenBytes = rows * 4096 * 2, mlpBytes = rows * 12288 * 2;
  uint64_t kvBytes = rows * 1024 * 2;
  uint64_t partialBytes = rows * 32 * 130 * 4;
  uint64_t extraBytes = drafter == Drafter::mtp ? hiddenBytes : drafter == Drafter::dflash ? 8 * hiddenBytes : 0;
  uint64_t logitBytes = uint64_t(maxLogitRows) * vocabSize * 2;
  Tensor storage = device.empty(2 * mlpBytes + 13 * hiddenBytes + kvBytes + rows * 32 * 6 + partialBytes + extraBytes + logitBytes);
  uint64_t offset = 0;
  auto take = [&](uint64_t bytes, auto&... tensors) { ((tensors = storage.view(offset, bytes), offset += bytes), ...); };
  take(mlpBytes, mlpGate, mlpUp);
  take(hiddenBytes, hidden[0], hidden[1], norm, temporary);
  take(2 * hiddenBytes, mixed);
  take(hiddenBytes, q, k, v, attnQRope);
  take(kvBytes, attnKRope);
  take(partialBytes, attnPartials);
  take(rows * 32 * 2, gdnB);
  take(rows * 32 * 4, gdnG);
  take(2 * hiddenBytes, gdnConvolved);
  take(hiddenBytes, mid);
  take(drafter == Drafter::mtp ? hiddenBytes : 0, targetHidden);
  take(drafter == Drafter::dflash ? 8 * hiddenBytes : 0, dflashFeatures);
  take(logitBytes, targetLogits);
}

void draft(Engine& engine, Batch& batch) {
  // drafter batch building
  bool dflash = engine.drafter == Drafter::dflash;
  uint32_t draftWidth = engine.draftWidth, draftSequences = batch.candidateRows / (draftWidth + 1);
  uint32_t rowsPerSequence = dflash ? draftWidth + 1 : 1;
  auto *queryStarts = engine.queryStartLoc.contents<uint32_t>(), *draftPositions = engine.draftPositions.contents<uint32_t>(),
       *logitRows = engine.logitRows.contents<uint32_t>();
  auto* draftTokens = engine.draftTokens.contents<int32_t>();
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    if (query.state == unbound)
      continue;
    const Sequence& sequence = *query.sequence;
    uint32_t draftIndex = query.state / query.count, inputStart = draftIndex * rowsPerSequence;
    queryStarts[draftIndex] = inputStart;
    writeMetadata(engine, sequence, draftIndex);
    draftTokens[inputStart] = sequence.request[sequence.kvValid];
    if (dflash) {
      std::fill_n(draftTokens + inputStart + 1, draftWidth, dflashMaskToken);
      std::iota(logitRows + draftIndex * draftWidth, logitRows + (draftIndex + 1) * draftWidth, inputStart + 1);
    } else
      for (uint32_t step = 0; step < draftWidth; ++step)
        draftPositions[step * maxBatchSequences + draftIndex] = sequence.kvValid + step;
  }
  uint32_t packedRows = draftSequences * rowsPerSequence;
  queryStarts[draftSequences] = packedRows;
  // drafter forward pass
  Scratch& scratch = engine.workspace;
  Device& commands = engine.device.command();
  Ops ops{commands, engine, scratch, !dflash};
  if (dflash) {
    Tensor hidden = scratch.hidden[0];
    commands.dispatch("q4_k_embed", size(4096, packedRows), size(256), {hidden, engine.draftTokens, engine.embedding});
    for (uint32_t index = 0; index < dflashLayers; ++index)
      hidden = ops.decoder(engine.draftModel.layers[index], hidden, scratch.hidden[(index + 1) & 1], packedRows, draftSequences,
                           targetKvLayers + index, engine.batchKvValid, nullptr, index);
    uint32_t proposals = draftSequences * draftWidth;
    commands.dispatch("gather_rows", size(uint64_t(proposals) * 4096), size(256), {scratch.norm, hidden, engine.logitRows});
    Tensor logits = ops.logits(scratch.norm, engine.draftModel.outputNorm, proposals);
    ops.argmax(engine.draftTokens.view(uint64_t(maxBatchSequences) * 4, engine.draftTokens.bytes - uint64_t(maxBatchSequences) * 4), logits,
               proposals, draftWidth, maxBatchSequences);
  } else {
    for (uint32_t step = 0; step < draftWidth; ++step) {
      Tensor ids = engine.draftTokens.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(draftSequences) * 4);
      Tensor positions = engine.draftPositions.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(draftSequences) * 4);
      Tensor hidden = ops.mtpInput(ids, scratch.norm, draftSequences, draftSequences, step ? 0 : 2);
      hidden = ops.decoder(engine.draftModel.layers[0], hidden, scratch.hidden[1], draftSequences, draftSequences, targetKvLayers, positions);
      Tensor logits = ops.logits(hidden, engine.draftModel.outputNorm, draftSequences);
      ops.argmax(engine.draftTokens.view(uint64_t(step + 1) * maxBatchSequences * 4, uint64_t(draftSequences) * 4), logits, draftSequences);
    }
  }
  commands.commit();
}

void forward(Engine& engine, Batch& batch) {
  auto* starts = engine.queryStartLoc.contents<uint32_t>();
  auto *tokens = engine.inputIds.contents<int32_t>(), *proposed = engine.draftTokens.contents<int32_t>();
  auto* logitRows = engine.logitRows.contents<uint32_t>();
  uint32_t rows = 0, maxQuery = 0, logitCount = 0, draftRow = 0;
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    query.start = rows;
    starts[row] = query.start;
    writeMetadata(engine, sequence, row);
    bool speculative = query.state != unbound;
    if (speculative) {
      tokens[query.start] = sequence.request[sequence.kvValid];
      for (uint32_t step = 0; step < engine.draftWidth; ++step)
        tokens[query.start + step + 1] = proposed[(step + 1) * maxBatchSequences + draftRow];
      ++draftRow;
    } else
      std::copy_n(sequence.request.data() + sequence.kvValid, query.count, tokens + query.start);
    rows += query.count;
    maxQuery = std::max(maxQuery, query.count);
    if (query.logit == unbound)
      continue;
    query.logit = logitCount;
    uint32_t count = speculative ? query.count : 1;
    for (uint32_t i = 0; i < count; ++i)
      logitRows[logitCount++] = query.start + (speculative ? i : query.count - 1);
  }
  starts[batch.size] = rows;
  if (gdnCheckpointBytes * batch.candidateRows > engine.candidateStates.bytes)
    engine.candidateStates = engine.device.empty(gdnCheckpointBytes * batch.candidateRows);
  Scratch& scratch = engine.workspace;
  Device& commands = engine.device.command();
  Ops ops{commands, engine, scratch, maxQuery <= maxDecodeRows};
  Tensor hidden = scratch.hidden[0];
  commands.dispatch("q4_k_embed", size(4096, rows), size(256), {hidden, engine.inputIds, engine.embedding});
  for (uint32_t i = 0; i < engine.layers.size(); ++i) {
    hidden =
        ops.decoder(engine.layers[i], hidden, scratch.hidden[(i + 1) & 1], rows, batch.size, i / fullAttentionInterval, engine.batchKvValid, &batch);
    if (engine.drafter == Drafter::dflash && i % 4 == 1)
      commands.dispatch("capture_hidden", size(uint64_t(rows) * 4096), size(256), {scratch.dflashFeatures, hidden}, (i - 1) / 4);
  }
  if (engine.drafter == Drafter::mtp)
    commands.copy(hidden, scratch.targetHidden.view(0, uint64_t(rows) * 4096 * 2));
  if (logitCount) {
    if (logitCount != rows)
      commands.dispatch("gather_rows", size(uint64_t(logitCount) * 4096), size(256), {scratch.norm, hidden, engine.logitRows});
    Tensor logits = ops.logits(logitCount == rows ? hidden : scratch.norm, engine.norm, logitCount);
    for (uint32_t row = 0; row < batch.size; ++row)
      if (batch.queries[row].logit != unbound) {
        Query& query = batch.queries[row];
        uint32_t count = query.state != unbound ? query.count : 1;
        ops.sample(*query.sequence, logits.view(uint64_t(query.logit) * vocabSize * 2, uint64_t(count) * vocabSize * 2),
                   engine.outputTokens.view(uint64_t(query.logit) * 4, uint64_t(count) * 4), count);
      }
  }
  if (engine.drafter != Drafter::none)
    encodeDrafter(ops, batch, rows);
  commands.commit();
}
} // namespace infeng::qwen35
