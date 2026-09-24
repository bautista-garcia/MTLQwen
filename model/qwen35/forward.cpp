#include "model/qwen35/qwen35.hpp"
#include <algorithm>
#include <gguf.h>

namespace infeng::qwen35 {
namespace {

MTL::Size size(uint64_t x, uint64_t y = 1, uint64_t z = 1) {
  return MTL::Size(x, y, z);
}

void loadWeights(Engine& engine, const std::filesystem::path& path, Weights& weights) {
  Tensor file = engine.device.mapped(path);
  std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_from_buffer(file.contents<uint8_t>(), file.bytes, {true, nullptr}), gguf_free);
  uint64_t data = gguf_get_data_offset(g.get()), count = gguf_get_n_tensors(g.get());
  engine.modelBytes += file.bytes - data;
  engine.parameterCount += count == 69 ? 1291904512 : count == 442 ? 9197093888 : 8953803264;
  for (uint64_t i = 0; i < count; ++i) {
    const int64_t* shape = gguf_get_tensor_ne(g.get(), i);
    weights.emplace(gguf_get_tensor_name(g.get(), i), Weight{file.view(data + gguf_get_tensor_offset(g.get(), i), gguf_get_tensor_size(g.get(), i)),
                                                             uint32_t(shape[1]), uint32_t(shape[0]), uint32_t(gguf_get_tensor_type(g.get(), i))});
  }
}

struct Ops {
  Engine& engine;
  const Batch& batch;
  uint32_t rows = 0, source = 0;
  bool preparing = rows != 0, decode = rows <= maxDecodeRows;
  bool sequential = false, prefill = false;
  Scratch& scratch = engine.workspace;
  std::string prefix;
  Tensor queries;

  Weight& weight(const std::string& name, uint32_t from = unbound) {
    std::string key = from == unbound ? prefix + name : name;
    auto& weights = engine.weights[from == unbound ? source : from];
    auto found = weights.find(key);
    if (found == weights.end())
      throw std::runtime_error("missing weight " + key);
    return found->second;
  }

  template <class K, class... Args>
  void dispatch(K kernel, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors, const Args&... args) {
    if (!preparing)
      engine.device.dispatch(kernel, threads, group, tensors, args...);
  }

  // A gated projection uses up weights in decode, or precomputed gate activations in prefill.
  Tensor linear(const Tensor& x, Weight& w, uint32_t count, Tensor output, const Tensor& auxiliary = {}, bool gated = false) {
    uint32_t phase = decode ? (gated ? 3 : 0) : count <= 8 ? 2 : 1;
    Kernel& k = w.kernels[phase];
    if (!k.pipeline) {
      uint32_t quant = w.type == GGML_TYPE_Q8_0 ? 0 : w.type == GGML_TYPE_IQ4_XS ? 4 : w.type - 11;
      const char* names[]{"q8_0", "q4_k", "q5_k", "q6_k", "iq4_xs"};
      uint32_t outputs[]{2, w.k == 32768 || (w.k == 4096 && w.n == 4096) ? 2u : 4u, 4, 8, 4};
      const char* suffix[]{"_decode", "_prefill", "_prefill_small"};
      std::string name = phase == 3 ? "mlp_gate_up_" + std::string(names[quant]) + "_decode"
                                    : "linear_" + std::string(names[quant]) + "_k" + std::to_string(w.k) + "_n" + std::to_string(w.n) +
                                          suffix[phase] + (phase == 0 && auxiliary.buffer ? "_add" : "");
      Pipeline* pipeline = engine.device.pipeline(name);
      uint32_t group = pipeline->maxTotalThreadsPerThreadgroup();
      uint32_t width = phase == 3 ? (w.type == GGML_TYPE_Q5_K ? 4 : 8) : phase == 0 ? outputs[quant] : phase == 1 ? 32 : 8;
      k = {pipeline, w.n / width * group, group};
    }
    dispatch(k.pipeline, size(k.threads, decode ? count : 1), size(k.group), {output, x, w, auxiliary.buffer ? auxiliary : output}, int64_t(count),
             gated ? 2u : uint32_t(bool(auxiliary.buffer)));
    return output;
  }

  Tensor logits(const Tensor& x, const Tensor& norm, uint32_t count) {
    dispatch(count == rows ? "rms_norm" : "rms_norm_gather", size(256, count), size(256), {scratch.temporary, x, norm, queries});
    return linear(scratch.temporary, weight("output.weight", 0), count, scratch.targetLogits);
  }

  void decoder(uint32_t layer, const Tensor& hidden) {
    source = layer >= targetLayers && engine.drafter == Drafter::dflash;
    prefix = "blk." + std::to_string(source ? layer - targetLayers : layer) + ".";
    bool fullAttention = layer >= targetLayers || (layer + 1) % fullAttentionInterval == 0;
    const Tensor& x = scratch.norm;
    dispatch("rms_norm", size(256, rows), size(256), {x, hidden, weight("attn_norm.weight")});
    if (fullAttention) {
      bool draft = source == 1;
      uint32_t kvLayer = layer < targetLayers ? layer / fullAttentionInterval : targetKvLayers + layer - targetLayers;
      Tensor qg = linear(x, weight("attn_q.weight"), rows, scratch.mixed), k = linear(x, weight("attn_k.weight"), rows, scratch.k);
      Tensor v = linear(x, weight("attn_v.weight"), rows, scratch.v);
      uint32_t heads = draft ? 32 : 16, group = draft ? 128 : 256;
      const Tensor& rope = draft ? engine.dflashRope : engine.rope;
      dispatch(draft ? "dflash_attention_prepare" : "attention_prepare", size(group, uint64_t(rows) * (draft ? 40 : 20)), size(group),
               {scratch.attnQRope, scratch.attnKRope, qg, k, weight("attn_q_norm.weight"), weight("attn_k_norm.weight"), rope, queries});
      uint32_t splits = std::max(1u, 8 / rows);
      dispatch(draft ? "dflash_attention_scan" : "attention_scan", size(uint64_t(rows) * heads * splits * 128), size(128),
               {scratch.attnPartials, scratch.attnQRope, scratch.attnKRope, v, engine.kv->key(kvLayer), engine.kv->value(kvLayer), queries}, rows,
               uint32_t(engine.blocks.size()) * blockTokens, splits, uint32_t(draft && layer + 1 < targetLayers + dflashLayers));
      dispatch(draft ? "dflash_attention_reduce" : "attention_reduce", size(uint64_t(rows) * heads * 128), size(128),
               {scratch.temporary, scratch.attnPartials, qg}, rows, splits);
    } else {
      Tensor mixed = linear(x, weight("attn_qkv.weight"), rows, scratch.mixed), z = linear(x, weight("attn_gate.weight"), rows, scratch.temporary);
      dispatch("gdn_ba_prepare_4096x32", size(32 * 64, rows), size(64),
               {scratch.gdnB, scratch.gdnG, x, weight("ssm_beta.weight"), weight("ssm_alpha.weight"), weight("ssm_a"), weight("ssm_dt.bias")},
               weight("ssm_beta.weight").type == GGML_TYPE_F32);
      uint64_t offset = (layer - layer / fullAttentionInterval) * (convStateBytes + recurrentStateBytes);
      dispatch("gdn_causal_conv_silu", size(uint64_t(8192) * rows), size(256),
               {scratch.q, scratch.k, scratch.v, mixed, weight("ssm_conv1d.weight"), queries}, offset);
      for (uint32_t phase = 0; phase < 2; ++phase)
        if (phase ? prefill : sequential)
          dispatch(phase ? "delta_rule_prefill" : "delta_rule_decode", phase ? size(128, 32, batch.size) : size(32 * 512, batch.size),
                   size(phase ? 128 : 512), {scratch.mixed, scratch.q, scratch.k, scratch.v, scratch.gdnG, scratch.gdnB, queries},
                   offset + convStateBytes);
      dispatch("gated_rms_norm_128", size(128, uint64_t(rows) * 32), size(128), {scratch.q, scratch.mixed, z, weight("ssm_norm.weight")});
    }
    linear(fullAttention ? scratch.temporary : scratch.q, weight(fullAttention ? "attn_output.weight" : "ssm_out.weight"), rows, hidden, hidden);
    dispatch("rms_norm", size(256, rows), size(256), {x, hidden, weight(source ? "ffn_norm.weight" : "post_attention_norm.weight")});
    Weight &gate = weight("ffn_gate.weight"), &up = weight("ffn_up.weight");
    if (!decode)
      linear(x, gate, rows, scratch.mlpGate);
    linear(x, decode ? gate : up, rows, scratch.mlpGate, decode ? up : scratch.mlpGate, true);
    linear(scratch.mlpGate, weight("ffn_down.weight"), rows, hidden, hidden);
  }

  Tensor mtpInput(const Tensor& proposals, const Tensor& hidden, uint32_t mode) {
    source = 0;
    prefix = "blk.32.nextn.";
    dispatch("embedding_q4_k", size(4096, rows), size(256), {scratch.norm, engine.requestData, weight("token_embd.weight", 0), queries, proposals},
             mode ? 0u : 2u, engine.maxContext + 1);
    dispatch("mtp_fuse", size(256, rows), size(256), {scratch.mixed, scratch.norm, hidden, queries, weight("enorm.weight"), weight("hnorm.weight")},
             rows, mode);
    return linear(scratch.mixed, weight("eh_proj.weight"), rows, scratch.hidden);
  }
};

} // namespace

void Engine::loadModel(const std::filesystem::path& path, const std::filesystem::path& draftPath) {
  if (!draftPath.empty())
    loadWeights(*this, draftPath, weights[1]);
  drafter = weights[1].empty() ? Drafter::none : weights[1].count("blk.32.attn_norm.weight") ? Drafter::mtp : Drafter::dflash;
  draftWidth = drafter == Drafter::mtp ? 2 : drafter == Drafter::dflash ? maxDraftTokens : 0;
  if (drafter == Drafter::mtp)
    weights[0] = std::move(weights[1]);
  else
    loadWeights(*this, path, weights[0]);
  Scratch& s = workspace;
  uint64_t h = uint64_t(maxBatchTokens) * 8192;
  for (Tensor* tensor : {&s.hidden, &s.norm, &s.temporary, &s.q, &s.k, &s.v, &s.attnQRope})
    *tensor = device.empty(h);
  s.mlpGate = device.empty(3 * h);
  s.mixed = device.empty(2 * h);
  s.attnKRope = device.empty(h / 4);
  s.attnPartials = device.empty(uint64_t(maxBatchTokens) * 32 * 130 * 4);
  s.gdnB = device.empty(maxBatchTokens * 64);
  s.gdnG = device.empty(maxBatchTokens * 128);
  if (drafter == Drafter::dflash)
    s.draftContext = device.empty(8 * h);
  s.targetLogits = device.empty(uint64_t(maxLogitRows) * vocabSize * 2);
  device.command();
  for (uint32_t pairs = 32; pairs <= (drafter == Drafter::dflash ? 64 : 32); pairs *= 2) {
    Tensor& table = pairs == 32 ? rope : dflashRope;
    table = device.empty(uint64_t(maxContext) * pairs * 4);
    device.dispatch("init_rope", size(uint64_t(maxContext) * pairs), size(256), {table}, 10000000.0f, pairs);
  }
  device.commit();
}

void forward(Engine& engine, const Batch& batch, bool drafting, uint32_t prepareRows) {
  bool dflash = engine.drafter == Drafter::dflash;
  Ops ops{engine, batch, prepareRows};
  uint32_t count = std::min(prepareRows, maxLogitRows);
  if (!ops.preparing) {
    for (uint32_t row = 0; row < batch.size; ++row) {
      GpuQuery query = batch.queries[row];
      bool speculative = query.state != unbound;
      if (drafting && !speculative)
        continue;
      uint32_t packed = drafting ? query.state / query.count : row;
      const Sequence& sequence = *engine.sequences[query.slot];
      const auto& [state, nextState] = engine.statePool[query.slot];
      query.start = ops.rows;
      query.count = drafting && !dflash ? 1 : query.count;
      query.samples = drafting ? (dflash ? engine.draftWidth : 1) : speculative ? query.count : query.position + query.count == sequence.requested;
      ops.sequential |= speculative || query.count == 1;
      ops.prefill |= !speculative && query.count > 1;
      query.previous = query.position ? state.address() : 0;
      query.next = speculative ? engine.candidateStates.address() + query.state * stateBytes : nextState.address();
      query.logit = count;
      batch.queries[drafting * maxBatchSequences + packed] = query;
      count += query.samples;
      ops.rows += query.count;
      ops.decode &= query.count <= maxDecodeRows;
    }
  }
  uint32_t rows = ops.rows, width = engine.draftWidth;
  Scratch& scratch = engine.workspace;
  ops.queries = engine.queryData.view(drafting, 1, maxBatchSequences * sizeof(GpuQuery));
  const Tensor& hidden = scratch.hidden;
  for (uint32_t step = 0; step < (drafting && !dflash ? width : 1); ++step) {
    if (drafting && !dflash)
      ops.mtpInput(engine.draftTokens.view(step * maxBatchSequences, rows, 4), scratch.norm, step ? 0 : 2);
    else
      ops.dispatch("embedding_q4_k", size(4096, rows), size(256),
                   {hidden, engine.requestData, ops.weight("token_embd.weight", 0), ops.queries, engine.draftTokens}, uint32_t(drafting),
                   engine.maxContext + 1);
    for (uint32_t i = 0; i < (drafting ? (dflash ? dflashLayers : mtpLayers) : targetLayers); ++i) {
      ops.decoder((drafting ? targetLayers : 0) + i, hidden);
      if (!drafting && dflash && i % 4 == 1)
        ops.dispatch("capture_hidden", size(uint64_t(rows) * 4096), size(256), {scratch.draftContext, hidden}, (i - 1) / 4);
    }
    if (count) {
      Tensor norm = ops.weight(drafting && !dflash ? "blk.32.nextn.shared_head_norm.weight" : "output_norm.weight", drafting && dflash);
      Tensor logits = ops.logits(hidden, norm, count);
      Tensor output =
          drafting ? engine.draftTokens.view((step + 1) * maxBatchSequences, dflash ? width * maxBatchSequences : rows, 4) : engine.outputTokens;
      ops.dispatch("sample_logits", size(256, count), size(256), {output, engine.rng, engine.sampledRng, logits, ops.queries},
                   uint32_t(drafting ? engine.drafter : Drafter::none));
    }
  }
  if (!drafting && engine.drafter != Drafter::none) {
    Tensor context =
        dflash ? ops.linear(scratch.draftContext, ops.weight("fc.weight", 1), rows, hidden) : ops.mtpInput(engine.draftTokens, hidden, 1);
    ops.dispatch("rms_norm", size(256, rows), size(256),
                 {scratch.norm, context, ops.weight(dflash ? "enc.output_norm.weight" : "blk.32.attn_norm.weight", dflash)});
    ops.source = dflash;
    for (uint32_t index = 0; index < (dflash ? dflashLayers : mtpLayers); ++index) {
      ops.prefix = "blk." + std::to_string(dflash ? index : targetLayers) + ".";
      Tensor k = ops.linear(scratch.norm, ops.weight("attn_k.weight"), rows, scratch.k),
             v = ops.linear(scratch.norm, ops.weight("attn_v.weight"), rows, scratch.v);
      ops.dispatch(dflash ? "dflash_store_kv" : "mtp_store_kv", size(uint64_t(rows) * 1024), size(dflash ? 128 : 256),
                   {engine.kv->key(targetKvLayers + index), engine.kv->value(targetKvLayers + index), k, v, ops.queries,
                    ops.weight("attn_k_norm.weight"), dflash ? engine.dflashRope : engine.rope},
                   uint32_t(engine.blocks.size()) * blockTokens);
    }
  }
}
} // namespace infeng::qwen35
