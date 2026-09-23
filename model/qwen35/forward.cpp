#include "model/qwen35/qwen35.hpp"
#include <algorithm>
#include <gguf.h>

namespace infeng::qwen35 {
namespace {

MTL::Size size(uint64_t x, uint64_t y = 1) {
  return MTL::Size(x, y, 1);
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
  Batch& batch;
  bool preparing = false;
  Device& commands = engine.device;
  Scratch& scratch = engine.workspace;
  uint32_t rows = batch.rows, source = 0;
  bool decode = batch.maxQuery <= maxDecodeRows;
  std::string prefix;
  Tensor positions = engine.batchKvValid;

  Weight& weight(const std::string& name, uint32_t from = unbound) {
    std::string key = from == unbound ? prefix + name : name;
    auto& weights = engine.weights[from == unbound ? source : from];
    auto found = weights.find(key);
    if (found == weights.end())
      throw std::runtime_error("missing weight " + key);
    return found->second;
  }

  Tensor t(const char* name) {
    return weight(name).tensor;
  }

  template <class K, class... Args>
  void dispatch(K kernel, MTL::Size threads, MTL::Size group, std::initializer_list<Tensor> tensors, const Args&... args) {
    if (!preparing)
      commands.dispatch(kernel, threads, group, tensors, args...);
  }

  Kernel& kernel(Weight& w, uint32_t phase, bool add = false) {
    Kernel& result = w.kernels[phase];
    if (!result.pipeline) {
      uint32_t quant = w.type == GGML_TYPE_Q8_0 ? 0 : w.type == GGML_TYPE_IQ4_XS ? 4 : w.type - 11;
      const char* names[]{"q8_0", "q4_k", "q5_k", "q6_k", "iq4_xs"};
      uint32_t outputs[]{2, w.k == 32768 || (w.k == 4096 && w.n == 4096) ? 2u : 4u, 4, 8, 4};
      const char* suffix[]{"_decode", "_prefill", "_prefill_small"};
      std::string name = phase == 3 ? "mlp_gate_up_" + std::string(names[quant]) + "_decode"
                                    : "linear_" + std::string(names[quant]) + "_k" + std::to_string(w.k) + "_n" + std::to_string(w.n) +
                                          suffix[phase] + (phase == 0 && add ? "_add" : "");
      Pipeline* pipeline = commands.pipeline(name);
      uint32_t group = pipeline->maxTotalThreadsPerThreadgroup();
      uint32_t width = phase == 3 ? (w.type == GGML_TYPE_Q5_K ? 4 : 8) : phase == 0 ? outputs[quant] : phase == 1 ? 32 : 8;
      result = {pipeline, w.n / width * group, group};
    }
    return result;
  }

  Tensor linear(const Tensor& x, Weight& weight, uint32_t count, Tensor output, const Tensor& residual = {}) {
    const Kernel& k = kernel(weight, decode ? 0 : count <= 8 ? 2 : 1, bool(residual.buffer));
    auto buffers = {output, x, weight.tensor, residual.buffer ? residual : output};
    if (decode)
      dispatch(k.pipeline, size(k.threads, count), size(k.group), buffers);
    else
      dispatch(k.pipeline, size(k.threads), size(k.group), buffers, int64_t(count), uint32_t(bool(residual.buffer)));
    return output;
  }

  Tensor logits(const Tensor& x, const Tensor& norm, uint32_t rows) {
    dispatch("rms_norm", size(256, rows), size(256), {scratch.temporary, x, norm});
    return linear(scratch.temporary, weight("output.weight", 0), rows, scratch.targetLogits);
  }

  Tensor attention(uint32_t layer, const Tensor& x, const Tensor& residual) {
    bool draft = source == 1;
    uint32_t kvLayer = layer < targetLayers ? layer / fullAttentionInterval : targetKvLayers + layer - targetLayers;
    Tensor qg = linear(x, weight("attn_q.weight"), rows, scratch.mixed), k = linear(x, weight("attn_k.weight"), rows, scratch.k);
    Tensor v = linear(x, weight("attn_v.weight"), rows, scratch.v);
    uint32_t heads = draft ? 32 : 16, group = draft ? 128 : 256;
    const Tensor& rope = draft ? engine.dflashRope : engine.rope;
    dispatch(draft ? "dflash_attention_prepare" : "attention_prepare", size(group, uint64_t(rows) * (draft ? 40 : 20)), size(group),
             {scratch.attnQRope, scratch.attnKRope, qg, k, t("attn_q_norm.weight"), t("attn_k_norm.weight"), rope, positions, engine.queryStartLoc},
             batch.packed);
    uint32_t splits = std::max(1u, 8 / rows);
    dispatch(draft ? "dflash_attention_scan" : "attention_scan", size(uint64_t(rows) * heads * splits * 128), size(128),
             {scratch.attnPartials, scratch.attnQRope, scratch.attnKRope, v, engine.kv->key(kvLayer), engine.kv->value(kvLayer), engine.sequenceSlots,
              positions, engine.queryStartLoc},
             batch.packed, rows, uint32_t(engine.blocks.size()) * blockTokens, splits, uint32_t(draft && layer + 1 < targetLayers + dflashLayers));
    dispatch(draft ? "dflash_attention_reduce" : "attention_reduce", size(uint64_t(rows) * heads * 128), size(128),
             {scratch.temporary, scratch.attnPartials, qg}, rows, splits);
    return linear(scratch.temporary, weight("attn_output.weight"), rows, scratch.mid, residual);
  }

  Tensor gdn(uint32_t layerIndex, const Tensor& x, const Tensor& residual) {
    Tensor conv = t("ssm_conv1d.weight");
    GdnState bank0 = gdnState(engine.gdnStates[0], maxBatchSequences, layerIndex);
    GdnState bank1 = gdnState(engine.gdnStates[1], maxBatchSequences, layerIndex);
    GdnState candidates = gdnState(engine.candidateStates, engine.candidateStates.bytes / gdnCheckpointBytes, layerIndex);
    Tensor mixed = linear(x, weight("attn_qkv.weight"), rows, scratch.mixed), z = linear(x, weight("attn_gate.weight"), rows, scratch.temporary);
    dispatch("gdn_ba_prepare_4096x32", size(32 * 64, rows), size(64),
             {scratch.gdnB, scratch.gdnG, x, t("ssm_beta.weight"), t("ssm_alpha.weight"), t("ssm_a"), t("ssm_dt.bias")},
             t("ssm_beta.weight").bytes == uint64_t(32) * 4096 * 4);
    for (uint32_t row = 0; row < batch.size; ++row) {
      const Query& query = batch.queries[row];
      uint64_t offset = uint64_t(query.start) * 8192 * 2;
      const Sequence& sequence = *query.sequence;
      Tensor destination = scratch.gdnConvolved.view(offset, uint64_t(query.count) * 8192 * 2), source = mixed.view(offset, destination.bytes);
      bool draft = query.state != unbound;
      Tensor candidate = draft ? candidates.conv.view(uint64_t(query.state) * convStateBytes, uint64_t(query.count) * convStateBytes) : destination;
      dispatch(draft ? "gdn_causal_conv_candidates" : "gdn_causal_conv_silu",
               size(uint64_t(8192) * (draft ? query.count : std::max(query.count, 4u))), size(256),
               {destination, bank0.conv, bank1.conv, candidate, source, conv}, sequence.slot, sequence.bank, uint32_t(sequence.kvValid != 0),
               int64_t(query.count));
    }
    dispatch("split_repeat_qk", size(uint64_t(rows) * 4096), size(256), {scratch.q, scratch.k, scratch.v, scratch.gdnConvolved});
    for (uint32_t row = 0; row < batch.size; ++row) {
      const Query& query = batch.queries[row];
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
      dispatch(kernel, draft || query.count == 1 ? size(32 * 512) : size(128, 32), size(draft || query.count == 1 ? 512 : 128),
               {destination, bank0.recurrent, bank1.recurrent, candidate, q, k, v, g, beta}, sequence.slot, sequence.bank,
               uint32_t(sequence.kvValid != 0), int64_t(query.count));
    }
    dispatch("gated_rms_norm_128", size(128, uint64_t(rows) * 32), size(128), {scratch.q, scratch.mixed, z, t("ssm_norm.weight")});
    return linear(scratch.q, weight("ssm_out.weight"), rows, scratch.mid, residual);
  }

  Tensor decoder(uint32_t layer, const Tensor& hidden, Tensor output) {
    source = layer >= targetLayers && engine.drafter == Drafter::dflash;
    prefix = "blk." + std::to_string(source ? layer - targetLayers : layer) + ".";
    dispatch("rms_norm", size(256, rows), size(256), {scratch.norm, hidden, t("attn_norm.weight")});
    Tensor mid =
        layer >= targetLayers || (layer + 1) % fullAttentionInterval == 0 ? attention(layer, scratch.norm, hidden) : gdn(layer, scratch.norm, hidden);
    dispatch("rms_norm", size(256, rows), size(256), {scratch.norm, mid, t(source ? "ffn_norm.weight" : "post_attention_norm.weight")});
    Kernel& fused = kernel(weight("ffn_gate.weight"), 3);
    if (decode)
      dispatch(fused.pipeline, size(fused.threads, rows), size(fused.group),
               {scratch.mlpGate, scratch.norm, t("ffn_gate.weight"), t("ffn_up.weight")}, int64_t(rows));
    else {
      Tensor gate = linear(scratch.norm, weight("ffn_gate.weight"), rows, scratch.mlpGate),
             up = linear(scratch.norm, weight("ffn_up.weight"), rows, scratch.mlpUp);
      dispatch("silu_and_mul", size(uint64_t(rows) * 12288), size(256), {scratch.mlpGate, gate, up});
    }
    return linear(scratch.mlpGate, weight("ffn_down.weight"), rows, output, mid);
  }

  Tensor mtpInput(const Tensor& ids, const Tensor& hidden, uint32_t mode) {
    source = 0;
    prefix = "blk.32.nextn.";
    dispatch("embedding_q4_k", size(4096, rows), size(256), {scratch.norm, ids, weight("token_embd.weight", 0).tensor});
    dispatch("mtp_fuse", size(256, rows), size(256),
             {scratch.mixed, scratch.norm, hidden, engine.mtpSeeds, engine.sequenceSlots, engine.stateBanks, engine.batchKvValid,
              engine.queryStartLoc, t("enorm.weight"), t("hnorm.weight")},
             batch.packed, rows, mode);
    return linear(scratch.mixed, weight("eh_proj.weight"), rows, scratch.hidden[0]);
  }

  void encodeDrafter() {
    bool mtp = engine.drafter == Drafter::mtp;
    source = !mtp;
    prefix.clear();
    Tensor context =
        mtp ? mtpInput(engine.inputIds, scratch.draftContext, 1) : linear(scratch.draftContext, weight("fc.weight"), rows, scratch.hidden[0]);
    dispatch("rms_norm", size(256, rows), size(256),
             {scratch.norm, context, mtp ? weight("blk.32.attn_norm.weight", 0).tensor : t("enc.output_norm.weight")});
    for (uint32_t index = 0; index < (mtp ? 1 : dflashLayers); ++index) {
      prefix = "blk." + std::to_string(mtp ? 32 : index) + ".";
      Tensor k = linear(scratch.norm, weight("attn_k.weight"), rows, scratch.k), v = linear(scratch.norm, weight("attn_v.weight"), rows, scratch.v);
      dispatch(mtp ? "mtp_store_kv" : "dflash_store_kv", size(uint64_t(rows) * 1024), size(mtp ? 256 : 128),
               {engine.kv->key(targetKvLayers + index), engine.kv->value(targetKvLayers + index), k, v, engine.sequenceSlots, engine.batchKvValid,
                engine.queryStartLoc, t("attn_k_norm.weight"), mtp ? engine.rope : engine.dflashRope},
               batch.size, rows, uint32_t(engine.blocks.size()) * blockTokens);
    }
    if (mtp && !preparing)
      for (uint32_t row = 0; row < batch.size; ++row)
        if (batch.queries[row].state == unbound)
          commands.copy(scratch.draftContext.view(uint64_t(batch.queries[row].start + batch.queries[row].count - 1) * 8192, 8192),
                        mtpSeed(engine, *batch.queries[row].sequence, 1 - batch.queries[row].sequence->bank));
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
  workspace.allocate(device, drafter);
  device.command();
  for (uint32_t pairs = 32; pairs <= (drafter == Drafter::dflash ? 64 : 32); pairs *= 2) {
    Tensor& table = pairs == 32 ? rope : dflashRope;
    table = device.empty(uint64_t(maxContext) * pairs * 4);
    device.dispatch("init_rope", size(uint64_t(maxContext) * pairs), size(256), {table}, 10000000.0f, pairs);
  }
  device.commit();
}

void Scratch::allocate(Device& device, Drafter drafter) {
  uint64_t h = uint64_t(maxBatchTokens) * 8192;
  auto allocate = [&](uint64_t bytes, std::initializer_list<Tensor*> buffers) {
    for (Tensor* tensor : buffers)
      *tensor = device.empty(bytes);
  };
  allocate(h, {&hidden[0], &hidden[1], &norm, &temporary, &q, &k, &v, &attnQRope, &mid});
  allocate(3 * h, {&mlpGate, &mlpUp});
  allocate(2 * h, {&mixed, &gdnConvolved});
  attnKRope = device.empty(h / 4);
  attnPartials = device.empty(uint64_t(maxBatchTokens) * 32 * 130 * 4);
  gdnB = device.empty(maxBatchTokens * 64);
  gdnG = device.empty(maxBatchTokens * 128);
  if (drafter != Drafter::none)
    draftContext = device.empty(h * (drafter == Drafter::mtp ? 1 : 8));
  targetLogits = device.empty(uint64_t(maxLogitRows) * vocabSize * 2);
}

void forward(Engine& engine, Batch& batch, bool drafting, bool prepare) {
  bool dflash = engine.drafter == Drafter::dflash;
  uint32_t rows = batch.rows, width = engine.draftWidth, count = drafting ? batch.packed * (dflash ? width : 1) : batch.logits;
  Scratch& scratch = engine.workspace;
  if (!prepare)
    engine.device.command();
  Ops ops{engine, batch, prepare};
  for (uint32_t step = 0; step < (drafting && !dflash ? width : 1); ++step) {
    Tensor hidden = scratch.hidden[0];
    if (drafting && !dflash) {
      ops.positions = engine.draftPositions.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4);
      hidden = ops.mtpInput(engine.draftTokens.view(uint64_t(step) * maxBatchSequences * 4, uint64_t(rows) * 4), scratch.norm, step ? 0 : 2);
    } else
      ops.dispatch("embedding_q4_k", size(4096, rows), size(256),
                   {hidden, drafting ? engine.draftTokens : engine.inputIds, ops.weight("token_embd.weight", 0).tensor});
    for (uint32_t i = 0; i < (drafting ? (dflash ? dflashLayers : mtpLayers) : targetLayers); ++i) {
      hidden = ops.decoder((drafting ? targetLayers : 0) + i, hidden, scratch.hidden[(i + 1) & 1]);
      if (!drafting && dflash && i % 4 == 1)
        ops.dispatch("capture_hidden", size(uint64_t(rows) * 4096), size(256), {scratch.draftContext, hidden}, (i - 1) / 4);
    }
    if (!drafting && engine.drafter == Drafter::mtp && !prepare)
      engine.device.copy(hidden, scratch.draftContext.view(0, uint64_t(rows) * 4096 * 2));
    if (count) {
      if (count != rows)
        ops.dispatch("gather_rows", size(uint64_t(count) * 4096), size(256), {scratch.norm, hidden, engine.logitRows});
      Tensor norm = ops.weight(drafting && !dflash ? "blk.32.nextn.shared_head_norm.weight" : "output_norm.weight", drafting && dflash).tensor;
      Tensor logits = ops.logits(count == rows ? hidden : scratch.norm, norm, count);
      if (drafting)
        ops.dispatch(
            "argmax_logits", size(256, count), size(256),
            {engine.draftTokens.view(uint64_t(step + 1) * maxBatchSequences * 4, uint64_t(dflash ? width * maxBatchSequences : rows) * 4), logits},
            dflash ? width : 1, dflash ? maxBatchSequences : 1);
      else
        for (uint32_t row = 0; row < batch.size; ++row) {
          const Query& query = batch.queries[row];
          if (query.logit == unbound)
            continue;
          const Sequence& sequence = *query.sequence;
          uint32_t samples = query.state != unbound ? query.count : 1;
          Tensor scores = logits.view(uint64_t(query.logit) * vocabSize * 2, uint64_t(samples) * vocabSize * 2);
          Tensor output = engine.outputTokens.view(uint64_t(query.logit) * 4, uint64_t(samples) * 4);
          if (sequence.temperature <= 0)
            ops.dispatch("argmax_logits", size(256, samples), size(256), {output, scores}, 1u, 1u);
          else
            ops.dispatch("sample_logits", size(samples), size(1),
                         {output, engine.rng.view(uint64_t(sequence.slot) * 8, 8),
                          engine.sampledRng.view(uint64_t(query.logit) * 8, uint64_t(samples) * 8), scores},
                         sequence.temperature, sequence.topP, sequence.topK);
        }
    }
  }
  if (!drafting && engine.drafter != Drafter::none)
    ops.encodeDrafter();
  if (!prepare)
    engine.device.commit();
}
} // namespace infeng::qwen35
