#include "qwen35.hpp"
#include <chrono>
#include <gguf.h>
#include <tuple>
namespace infeng::qwen35 {
namespace {
constexpr const char* quantNames[]{"q8_0", "q4_k", "q5_k", "q6_k", "iq4_xs"};
constexpr uint8_t quantOutputs[]{2, 4, 4, 8, 4};
using Weight = std::tuple<Tensor, uint32_t, uint32_t, QuantType>;
struct GGUF {
  Tensor file;
  std::unique_ptr<gguf_context, decltype(&gguf_free)> context{nullptr, gguf_free};
  size_t data;
  GGUF(Model& model, const std::filesystem::path& path) : file(model.device.mapped(path)) {
    context.reset(gguf_init_from_buffer(file.contents<uint8_t>(), file.bytes, {true, nullptr}));
    data = gguf_get_data_offset(context.get());
    model.modelBytes += file.bytes - data;
    model.parameterCount += gguf_get_n_tensors(context.get()) == 69 ? 1291904512 : gguf_get_n_tensors(context.get()) == 442 ? 9197093888 : 8953803264;
  }
  Weight operator()(const std::string& name) {
    int64_t id = gguf_find_tensor(context.get(), name.c_str());
    if (id < 0)
      throw std::runtime_error("missing weight " + name);
    const int64_t* shape = gguf_get_tensor_ne(context.get(), id);
    uint64_t offset = data + gguf_get_tensor_offset(context.get(), id);
    return {file.view(offset, gguf_get_tensor_size(context.get(), id)), uint32_t(shape[1]), shape[1] == 1 ? 0 : uint32_t(shape[0]),
            QuantType(gguf_get_tensor_type(context.get(), id))};
  }
};
Linear linear(Device& device, Weight weight) {
  auto [data, n, k, type] = weight;
  uint32_t quant = type == QuantType::Q8_0 ? 0 : type == QuantType::IQ4_XS ? 4 : uint32_t(type) - 11;
  std::string root = std::string(quantNames[quant]) + "_k" + std::to_string(k) + "_n" + std::to_string(n);
  Pipeline* decode = device.pipeline(root + "_decode");
  uint8_t outputs = type == QuantType::Q4_K && ((k == 4096 && n == 4096) || k == 32768) ? 2 : quantOutputs[quant];
  return {data,
          {decode, n == 4096 ? device.pipeline(root + "_decode_add") : nullptr, device.pipeline(root + "_prefill"),
           device.pipeline(root + "_prefill_small")},
          k,
          n,
          outputs};
}
} // namespace
Model::Model(const std::filesystem::path& path, const std::filesystem::path& kernels, uint32_t context, bool profile, Drafter selected,
             const std::filesystem::path& draftPath)
    : device(kernels, profile), maxContext(context), blocks((context + blockTokens - 1) / blockTokens), maxLogicalBlocks(blocks.size()),
      drafter(selected) {
  GGUF target(*this, path);
  GGUF* g = &target;
  std::string r;
  auto w = [&](const char* name) { return (*g)(r + name); };
  auto t = [&](const char* name) { return std::get<0>(w(name)); };
  auto p = [&](const char* name) { return linear(device, w(name)); };
  auto build = [&](Layer& layer, bool full, bool draft) {
    Weight gate = w("ffn_gate.weight");
    QuantType type = std::get<3>(gate);
    uint32_t quant = type == QuantType::Q8_0 ? 0 : type == QuantType::IQ4_XS ? 4 : uint32_t(type) - 11;
    layer = {full,
             0,
             t("attn_norm.weight"),
             t(draft ? "ffn_norm.weight" : "post_attention_norm.weight"),
             {linear(device, gate), p("ffn_up.weight"), p("ffn_down.weight")}};
    layer.mlp.fusedDecode = device.pipeline("mlp_gate_up_" + std::string(quantNames[quant]) + "_decode");
    layer.mlp.outputsPerGroup = type == QuantType::Q5_K ? 4 : 8;
    if (full)
      layer.attention = {p("attn_q.weight"),      p("attn_k.weight"),      p("attn_v.weight"),
                         p("attn_output.weight"), t("attn_q_norm.weight"), t("attn_k_norm.weight")};
    else {
      Weight b = w("ssm_beta.weight"), a = w("ssm_alpha.weight");
      layer.gdn = {p("attn_qkv.weight"),   p("attn_gate.weight"), p("ssm_out.weight"), std::get<0>(b), std::get<0>(a),
                   t("ssm_conv1d.weight"), t("ssm_norm.weight"),  t("ssm_dt.bias"),    t("ssm_a"),     std::get<3>(b) == QuantType::F32};
    }
  };
  embedding = t("token_embd.weight");
  norm = t("output_norm.weight");
  head = p("output.weight");
  uint8_t kvIndex = 0;
  for (uint32_t i = 0; i < layers.size(); ++i) {
    r = "blk." + std::to_string(i) + ".";
    build(layers[i], (i + 1) % fullAttentionInterval == 0, false);
    if (layers[i].fullAttention)
      layers[i].kvIndex = kvIndex++;
  }
  if (hasMtp()) {
    r = "blk.32.";
    build(mtp.layer, true, false);
    r += "nextn.";
    mtp.embeddingNorm = t("enorm.weight");
    mtp.hiddenNorm = t("hnorm.weight");
    mtp.outputNorm = t("shared_head_norm.weight");
    mtp.fusion = p("eh_proj.weight");
  }
  if (hasDflash()) {
    GGUF draft(*this, draftPath);
    g = &draft;
    r.clear();
    dflash.fusion = p("fc.weight");
    dflash.hiddenNorm = t("enc.output_norm.weight");
    dflash.outputNorm = t("output_norm.weight");
    for (uint32_t i = 0; i < dflashLayers; ++i) {
      r = "blk." + std::to_string(i) + ".";
      build(dflash.layers[i], true, true);
    }
  }
  auto makeRope = [&](uint32_t pairs) {
    Tensor result = device.empty(uint64_t(maxContext) * pairs * 4);
    CommandBuffer commands(device, 64);
    commands.dispatch(device.pipeline("init_rope"), MTL::Size((uint64_t(maxContext) * pairs + 255) / 256 * 256, 1, 1), MTL::Size(256, 1, 1), {result},
                      maxContext, 10000000.0f, pairs);
    commands.commit();
    return result;
  };
  rope = makeRope(32);
  if (hasDflash())
    dflashRope = makeRope(64);
  Tensor* controls[]{&inputIds,    &batchKvValid, &queryStartLoc, &draftPositions, &sequenceSlots, &stateBanks,
                     &draftTokens, &outputTokens, &rng,           &logitRows};
  Tensor control = device.empty(sizeof(controls) / sizeof(*controls) * 512, true);
  for (uint32_t i = 0; i < sizeof(controls) / sizeof(*controls); ++i)
    *controls[i] = control.view(uint64_t(i) * 512, 512);
  auto* seeds = rng.contents<uint64_t>();
  uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (uint32_t i = 0; i < maxBatchSequences; ++i)
    seeds[i] = ++seed;
  if (hasMtp()) {
    std::array<_Float16, 2 * maxBatchSequences * 4096> zero{};
    mtpSeeds = device.upload(zero.data(), sizeof(zero));
  }
  Tensor stateArena = device.empty(2 * gdnCheckpointBytes * maxBatchSequences);
  for (uint32_t i = 0; i < layers.size(); ++i)
    if (!layers[i].fullAttention)
      for (uint32_t bank = 0; bank < 2; ++bank) {
        uint64_t offset = (bank * gdnCheckpointBytes + gdnOffset(i)) * maxBatchSequences;
        states[i].conv[bank] = stateArena.view(offset, convStateBytes * maxBatchSequences);
        states[i].recurrent[bank] = stateArena.view(offset + convStateBytes * maxBatchSequences, recurrentStateBytes * maxBatchSequences);
      }
  kv = std::make_unique<SparseKV>(device, maxBatchSequences * maxLogicalBlocks, blocks.size(), targetKvLayers,
                                  hasMtp()      ? mtpLayers
                                  : hasDflash() ? dflashLayers
                                                : 0);
}
} // namespace infeng::qwen35
