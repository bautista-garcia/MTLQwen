#include "qwen35.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
namespace infeng::qwen35 {
namespace {
struct Mapping {
    int fd = -1;
    uint8_t* data = nullptr;
    uint64_t bytes = 0;
    explicit Mapping(const std::filesystem::path& path) {
        fd = open(path.c_str(), O_RDONLY); if (fd < 0) throw std::runtime_error("failed to open GGUF");
        struct stat status{}; if (fstat(fd, &status)) throw std::runtime_error("failed to stat GGUF");
        bytes = status.st_size; data = static_cast<uint8_t*>(mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) throw std::runtime_error("failed to mmap GGUF");
    }
    ~Mapping() { if (data && data != MAP_FAILED) munmap(data, bytes); if (fd >= 0) close(fd); }
};
struct Reader {
    const uint8_t* data;
    uint64_t bytes, offset = 0;
    template <class T> T read() {
        if (offset + sizeof(T) > bytes) throw std::runtime_error("truncated GGUF");
        T value; std::memcpy(&value, data + offset, sizeof(T)); offset += sizeof(T); return value;
    }
    std::string string() {
        uint64_t count = read<uint64_t>(); if (offset + count > bytes) throw std::runtime_error("truncated GGUF string");
        std::string value(reinterpret_cast<const char*>(data + offset), count); offset += count; return value;
    }
    void skip(uint64_t count) { if (offset + count > bytes) throw std::runtime_error("truncated GGUF"); offset += count; }
};
double number(Reader& reader, uint32_t type) {
    switch (type) {
        case 0: return reader.read<uint8_t>(); case 1: return reader.read<int8_t>();
        case 2: return reader.read<uint16_t>(); case 3: return reader.read<int16_t>();
        case 4: return reader.read<uint32_t>(); case 5: return reader.read<int32_t>();
        case 6: return reader.read<float>(); case 7: return reader.read<uint8_t>() != 0;
        case 10: return reader.read<uint64_t>(); case 11: return reader.read<int64_t>(); case 12: return reader.read<double>();
        default: throw std::runtime_error("GGUF metadata is not numeric");
    }
}
void skipValue(Reader& reader, uint32_t type) {
    static constexpr uint8_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type < std::size(sizes) && sizes[type]) { reader.skip(sizes[type]); return; }
    if (type == 8) { reader.skip(reader.read<uint64_t>()); return; }
    if (type == 9) { uint32_t item = reader.read<uint32_t>(); uint64_t count = reader.read<uint64_t>();
        for (uint64_t i = 0; i < count; ++i) skipValue(reader, item); return; }
    throw std::runtime_error("unknown GGUF metadata type " + std::to_string(type));
}
struct Info { std::string name; std::vector<uint64_t> shape; QuantType type; uint64_t offset; };
struct Weight { Tensor data; std::vector<uint64_t> shape; QuantType type; };
struct Selected { const Info* info; std::string target, transform; uint64_t bytes, arenaOffset; };
uint64_t product(const std::vector<uint64_t>& shape) {
    uint64_t value = 1; for (uint64_t dim : shape) value *= dim; return value;
}
std::pair<uint32_t, uint32_t> block(QuantType type) {
    switch (type) {
        case QuantType::Q8_0: return {32, 34}; case QuantType::Q4_K: return {256, 144};
        case QuantType::Q5_K: return {256, 176}; case QuantType::Q6_K: return {256, 210};
        case QuantType::IQ4_XS: return {256, 136}; default: throw std::runtime_error("unsupported quantization type");
    }
}
std::pair<std::string, std::string> target(const std::string& name) {
    static const std::unordered_map<std::string, std::string> roots = {
        {"token_embd.weight", "model.embed_tokens.weight"}, {"output.weight", "lm_head.weight"},
        {"output_norm.weight", "model.norm.weight"}, {"model.output_norm.weight", "model.norm.weight"}};
    if (auto found = roots.find(name); found != roots.end()) return {found->second, {}};
    if (name.rfind("blk.", 0)) return {};
    size_t first = name.find('.', 4);
    if (first == std::string::npos) return {};
    std::string index = name.substr(4, first - 4), suffix = name.substr(first + 1);
    static const std::unordered_map<std::string, std::string> blocks = {
        {"attn_norm.weight", "input_layernorm.weight"}, {"post_attention_norm.weight", "post_attention_layernorm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"}, {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"}, {"ffn_down.weight", "mlp.down_proj.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"}, {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"}, {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_q_norm.weight", "self_attn.q_norm.weight"}, {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"attn_qkv.weight", "linear_attn.in_proj_qkv.weight"}, {"attn_gate.weight", "linear_attn.in_proj_z.weight"},
        {"ssm_beta.weight", "linear_attn.in_proj_b.weight"}, {"ssm_alpha.weight", "linear_attn.in_proj_a.weight"},
        {"ssm_conv1d.weight", "linear_attn.conv1d_weight"}, {"ssm_out.weight", "linear_attn.out_proj.weight"},
        {"ssm_norm.weight", "linear_attn.norm.weight"}, {"ssm_dt.bias", "linear_attn.dt_bias"},
        {"ssm_a", "linear_attn.A_log"}, {"nextn.eh_proj.weight", "nextn.fusion.weight"},
        {"nextn.enorm.weight", "nextn.embedding_norm.weight"},
        {"nextn.hnorm.weight", "nextn.hidden_norm.weight"},
        {"nextn.shared_head_norm.weight", "nextn.output_norm.weight"}};
    auto found = blocks.find(suffix); if (found == blocks.end()) return {};
    std::string transform = found->second == "linear_attn.conv1d_weight" ? "conv1d" :
                            found->second == "linear_attn.A_log" ? "neg_log" : "";
    return {"model.layers." + index + "." + found->second, transform};
}
Weight& require(std::unordered_map<std::string, Weight>& weights, const std::string& name) {
    auto found = weights.find(name); if (found == weights.end()) throw std::runtime_error("missing weight " + name);
    return found->second;
}
void shape(const Weight& weight, std::initializer_list<uint64_t> expected, const std::string& name) {
    if (weight.shape != std::vector<uint64_t>(expected)) throw std::runtime_error("unsupported shape for " + name);
}
struct LinearSpec { const char* decode; const char* prefill; uint16_t threads; uint8_t outputs; };
LinearSpec linearSpec(QuantType type, uint32_t k, uint32_t n) {
    if (type == QuantType::F16 && k == 4096 && n == 32) return {"dense_4096x32", "dense_4096x32", 32, 1};
    if (type == QuantType::Q4_K) {
        if (k == 4096 && n == 1024) return {"q4_k_k4096_n1024_decode", "q4_k_k4096_n1024_prefill", 64, 4};
        if (k == 4096 && n == 4096) return {"q4_k_k4096_n4096_decode", "q4_k_k4096_n4096_prefill", 64, 2};
        if (k == 4096 && n == 8192) return {"q4_k_k4096_n8192_decode", "q4_k_k4096_n8192_prefill", 64, 4};
        if (k == 4096 && n == 12288) return {"q4_k_k4096_n12288_decode", "q4_k_k4096_n12288_prefill", 64, 4};
        if (k == 12288 && n == 4096) return {"q4_k_k12288_n4096_decode", "q4_k_k12288_n4096_prefill", 64, 4};
    }
    if (type == QuantType::Q5_K) {
        if (k == 4096 && n == 1024) return {"q5_k_k4096_n1024_decode", "q5_k_k4096_n1024_prefill", 128, 4};
        if (k == 4096 && n == 4096) return {"q5_k_k4096_n4096_decode", "q5_k_k4096_n4096_prefill", 128, 4};
        if (k == 4096 && n == 8192) return {"q5_k_k4096_n8192_decode", "q5_k_k4096_n8192_prefill", 128, 4};
        if (k == 4096 && n == 12288) return {"q5_k_k4096_n12288_decode", "q5_k_k4096_n12288_prefill", 128, 4};
        if (k == 12288 && n == 4096) return {"q5_k_k12288_n4096_decode", "q5_k_k12288_n4096_prefill", 128, 4};
    }
    if (type == QuantType::Q6_K) {
        if (k == 4096 && n == 1024) return {"q6_k_k4096_n1024_decode", "q6_k_k4096_n1024_prefill", 128, 8};
        if (k == 4096 && n == 8192) return {"q6_k_k4096_n8192_decode", "q6_k_k4096_n8192_prefill", 128, 8};
        if (k == 12288 && n == 4096) return {"q6_k_k12288_n4096_decode", "q6_k_k12288_n4096_prefill", 128, 8};
        if (k == 4096 && n == 248320) return {"q6_k_k4096_n248320_decode", "q6_k_k4096_n248320_prefill", 128, 8};
    }
    if (type == QuantType::Q8_0 && k == 4096 && n == 4096)
        return {"q8_0_k4096_n4096_decode", "q8_0_k4096_n4096_prefill", 128, 2};
    if (type == QuantType::Q8_0 && k == 8192 && n == 4096)
        return {"q8_0_k8192_n4096_decode", "q8_0_k8192_n4096_prefill", 128, 2};
    if (type == QuantType::IQ4_XS && k == 4096 && n == 12288)
        return {"iq4_xs_k4096_n12288_decode", "iq4_xs_k4096_n12288_prefill", 64, 4};
    throw std::runtime_error("unsupported linear type/shape " + std::to_string(static_cast<uint32_t>(type)) + " " +
                             std::to_string(k) + "x" + std::to_string(n));
}
Linear makeLinear(Device& device, const Weight& weight) {
    if (weight.shape.size() != 2) throw std::runtime_error("linear weight is not a matrix");
    uint32_t n = weight.shape[0], k = weight.shape[1]; LinearSpec spec = linearSpec(weight.type, k, n);
    Pipeline* decodeAdd = weight.type != QuantType::F16 && n == 4096
        ? device.pipeline(std::string(spec.decode) + "_add") : nullptr;
    return {weight.data, device.pipeline(spec.decode), decodeAdd, device.pipeline(spec.prefill), k, n,
            spec.threads, spec.outputs, weight.type};
}
MlpWeights makeMlp(Device& device, std::unordered_map<std::string, Weight>& weights, const std::string& prefix) {
    MlpWeights result{makeLinear(device, require(weights, prefix + "gate_proj.weight")),
                      makeLinear(device, require(weights, prefix + "up_proj.weight")),
                      makeLinear(device, require(weights, prefix + "down_proj.weight"))};
    if (result.gate.type != result.up.type) throw std::runtime_error("mixed MLP gate/up quantization unsupported");
    if (result.gate.type == QuantType::Q4_K)
        result.fusedDecode = device.pipeline("mlp_gate_up_q4_k_decode"), result.decodeThreads = 128,
        result.outputsPerGroup = 8;
    else if (result.gate.type == QuantType::Q5_K)
        result.fusedDecode = device.pipeline("mlp_gate_up_q5_k_decode"), result.decodeThreads = 128,
        result.outputsPerGroup = 4;
    else if (result.gate.type == QuantType::IQ4_XS)
        result.fusedDecode = device.pipeline("mlp_gate_up_iq4_xs_decode"), result.decodeThreads = 128,
        result.outputsPerGroup = 8;
    else throw std::runtime_error("unsupported fused MLP quantization");
    return result;
}
}  // namespace
Model::Model(const std::filesystem::path& path, const std::filesystem::path& kernelPath,
             uint32_t context, bool profile)
    : device(kernelPath, profile), maxContext(context), blocks((context + blockTokens - 1) / blockTokens),
      maxLogicalBlocks(blocks.size()) {
    auto started = std::chrono::steady_clock::now();
    if (path.extension() != ".gguf") throw std::runtime_error("Qwen3.5 only accepts GGUF weights");
    if (!context || context > 65536) throw std::runtime_error("max_context must be between 1 and 65536");
    Mapping mapping(path); Reader reader{mapping.data, mapping.bytes};
    if (reader.read<uint32_t>() != 0x46554747) throw std::runtime_error("invalid GGUF magic");
    uint32_t version = reader.read<uint32_t>(); if (version != 2 && version != 3) throw std::runtime_error("unsupported GGUF version");
    uint64_t tensorCount = reader.read<uint64_t>(), metadataCount = reader.read<uint64_t>(), alignment = 32;
    std::unordered_map<std::string, double> metadata;
    for (uint64_t i = 0; i < metadataCount; ++i) {
        std::string key = reader.string(); uint32_t type = reader.read<uint32_t>();
        static const std::array<const char*, 14> wanted = {"general.alignment", "qwen35.block_count",
            "qwen35.embedding_length", "qwen35.feed_forward_length", "qwen35.context_length",
            "qwen35.attention.head_count", "qwen35.attention.head_count_kv", "qwen35.attention.key_length",
            "qwen35.full_attention_interval", "qwen35.ssm.conv_kernel", "qwen35.ssm.state_size",
            "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank", "qwen35.nextn_predict_layers"};
        bool keep = std::find_if(wanted.begin(), wanted.end(), [&](const char* value) { return key == value; }) != wanted.end();
        if (keep) metadata[key] = number(reader, type); else skipValue(reader, type);
        if (key == "general.alignment") alignment = metadata[key];
    }
    auto meta = [&](const char* key) -> uint64_t {
        auto found = metadata.find(key); if (found == metadata.end()) throw std::runtime_error(std::string("missing metadata ") + key);
        return found->second;
    };
    auto nextn = metadata.find("qwen35.nextn_predict_layers");
    uint64_t nextnLayers = nextn == metadata.end() ? 0 : nextn->second;
    if (nextnLayers > 1) throw std::runtime_error("only one Qwen3.5 MTP layer is supported");
    hasMtp = nextnLayers == 1;
    if (meta("qwen35.block_count") != layers.size() + nextnLayers || meta("qwen35.embedding_length") != 4096 ||
        meta("qwen35.feed_forward_length") != 12288 || meta("qwen35.context_length") < context ||
        meta("qwen35.attention.head_count") != 16 || meta("qwen35.attention.head_count_kv") != 4 ||
        meta("qwen35.attention.key_length") != 256 || meta("qwen35.full_attention_interval") != 4 ||
        meta("qwen35.ssm.conv_kernel") != 4 || meta("qwen35.ssm.state_size") != 128 ||
        meta("qwen35.ssm.group_count") != 16 || meta("qwen35.ssm.time_step_rank") != 32)
        throw std::runtime_error("unsupported Qwen3.5 architecture");
    std::vector<Info> infos; infos.reserve(tensorCount);
    for (uint64_t i = 0; i < tensorCount; ++i) {
        Info info; info.name = reader.string(); uint32_t dims = reader.read<uint32_t>(); info.shape.resize(dims);
        for (uint32_t dim = 0; dim < dims; ++dim) info.shape[dims - dim - 1] = reader.read<uint64_t>();
        info.type = static_cast<QuantType>(reader.read<uint32_t>()); info.offset = reader.read<uint64_t>(); infos.push_back(std::move(info));
    }
    uint64_t dataStart = (reader.offset + alignment - 1) / alignment * alignment, arenaBytes = 0;
    std::vector<Selected> selected; selected.reserve(infos.size()); std::unordered_set<std::string> targets;
    for (const Info& info : infos) {
        auto [name, transform] = target(info.name); if (name.empty() || !targets.insert(name).second) continue;
        if (info.type == QuantType::F32 && info.shape == std::vector<uint64_t>{32, 4096}) transform = "f16";
        uint64_t numel = product(info.shape), bytes;
        if (info.type == QuantType::F32 || info.type == QuantType::F16)
            bytes = numel * (transform == "conv1d" || transform == "f16" || info.type == QuantType::F16 ? 2 : 4);
        else { auto [blockSize, blockBytes] = block(info.type); bytes = numel / blockSize * blockBytes; }
        arenaBytes = (arenaBytes + 255) & ~255ull; selected.push_back({&info, name, transform, bytes, arenaBytes});
        arenaBytes += bytes; parameterCount += numel; modelBytes += bytes;
    }
    Tensor weightArena = device.empty(arenaBytes); std::unordered_map<std::string, Weight> weights;
    weights.reserve(selected.size());
    for (const Selected& item : selected) {
        const Info& info = *item.info; uint64_t source = dataStart + info.offset;
        if (source >= mapping.bytes) throw std::runtime_error("GGUF tensor offset outside file");
        if (item.transform == "neg_log") {
            std::vector<float> values(product(info.shape)); std::memcpy(values.data(), mapping.data + source, values.size() * 4);
            for (float& value : values) value = std::log(-value); device.write(weightArena.view(item.arenaOffset, item.bytes), values.data(), item.bytes);
        } else if (item.transform == "conv1d" || item.transform == "f16") {
            uint64_t count = product(info.shape); std::vector<_Float16> values(count); auto* sourceValues = reinterpret_cast<const float*>(mapping.data + source);
            for (uint64_t i = 0; i < count; ++i) values[i] = sourceValues[i];
            device.write(weightArena.view(item.arenaOffset, item.bytes), values.data(), item.bytes);
        } else device.write(weightArena.view(item.arenaOffset, item.bytes), mapping.data + source, item.bytes);
        QuantType type = item.transform == "f16" ? QuantType::F16 : info.type;
        weights.emplace(item.target, Weight{weightArena.view(item.arenaOffset, item.bytes), info.shape, type});
    }
    Weight& embed = require(weights, "model.embed_tokens.weight"); shape(embed, {248320, 4096}, "embedding"); embedding = embed.data;
    Weight& finalNorm = require(weights, "model.norm.weight"); shape(finalNorm, {4096}, "model norm"); norm = finalNorm.data;
    head = makeLinear(device, require(weights, "lm_head.weight")); if (head.n != 248320) throw std::runtime_error("unsupported vocabulary size");
    uint8_t kv = 0;
    for (uint32_t index = 0; index < layers.size(); ++index) {
        Layer& layer = layers[index]; std::string root = "model.layers." + std::to_string(index) + ".";
        layer.inputNorm = require(weights, root + "input_layernorm.weight").data;
        layer.postNorm = require(weights, root + "post_attention_layernorm.weight").data;
        layer.mlp = makeMlp(device, weights, root + "mlp."); layer.fullAttention = (index + 1) % 4 == 0;
        if (layer.fullAttention) {
            std::string prefix = root + "self_attn."; layer.kvIndex = kv++;
            layer.attention = {makeLinear(device, require(weights, prefix + "q_proj.weight")),
                               makeLinear(device, require(weights, prefix + "k_proj.weight")),
                               makeLinear(device, require(weights, prefix + "v_proj.weight")),
                               makeLinear(device, require(weights, prefix + "o_proj.weight")),
                               require(weights, prefix + "q_norm.weight").data,
                               require(weights, prefix + "k_norm.weight").data};
        } else {
            std::string prefix = root + "linear_attn.";
            layer.gdn = {makeLinear(device, require(weights, prefix + "in_proj_qkv.weight")),
                         makeLinear(device, require(weights, prefix + "in_proj_z.weight")),
                         makeLinear(device, require(weights, prefix + "in_proj_b.weight")),
                         makeLinear(device, require(weights, prefix + "in_proj_a.weight")),
                         makeLinear(device, require(weights, prefix + "out_proj.weight")),
                         require(weights, prefix + "conv1d_weight").data, require(weights, prefix + "norm.weight").data,
                         require(weights, prefix + "dt_bias").data, require(weights, prefix + "A_log").data};
        }
    }
    if (hasMtp) {
        std::string root = "model.layers.32.", attention = root + "self_attn.";
        mtp.embeddingNorm = require(weights, root + "nextn.embedding_norm.weight").data;
        mtp.hiddenNorm = require(weights, root + "nextn.hidden_norm.weight").data;
        mtp.outputNorm = require(weights, root + "nextn.output_norm.weight").data;
        mtp.fusion = makeLinear(device, require(weights, root + "nextn.fusion.weight"));
        mtp.layer.inputNorm = require(weights, root + "input_layernorm.weight").data;
        mtp.layer.postNorm = require(weights, root + "post_attention_layernorm.weight").data;
        mtp.layer.mlp = makeMlp(device, weights, root + "mlp."); mtp.layer.fullAttention = true;
        mtp.layer.attention = {makeLinear(device, require(weights, attention + "q_proj.weight")),
                               makeLinear(device, require(weights, attention + "k_proj.weight")),
                               makeLinear(device, require(weights, attention + "v_proj.weight")),
                               makeLinear(device, require(weights, attention + "o_proj.weight")),
                               require(weights, attention + "q_norm.weight").data,
                               require(weights, attention + "k_norm.weight").data};
    }
    kernels = {device.pipeline("q4_k_embed"), device.pipeline("rmsnorm"), device.pipeline("add_half"),
               device.pipeline("silu_mul"), device.pipeline("pad_rows"), device.pipeline("init_rope"),
               device.pipeline("argmax_logits"), device.pipeline("sample_logits"), device.pipeline("mtp_draft_fuse"),
               device.pipeline("attention_prepare"), device.pipeline("attention_scan"),
               device.pipeline("attention_reduce"), device.pipeline("mtp_store_kv"), device.pipeline("gdn_prepare"),
               device.pipeline("gdn_ba_prepare_4096x32"), device.pipeline("gdn_causal_conv_silu"),
               device.pipeline("gdn_causal_conv_candidates"),
               device.pipeline("split_repeat_qk"), device.pipeline("delta_rule_prefill"),
               device.pipeline("delta_rule_decode"), device.pipeline("delta_rule_candidates"),
               device.pipeline("rmsnorm_gated_128"), device.pipeline("mtp_fuse"),
               device.pipeline("gather_rows")};
    rope = device.empty(uint64_t(context) * 32 * 2 * 2);
    { CommandBuffer commands(device, 64); commands.dispatch(kernels.initRope, MTL::Size(((uint64_t(context) * 32 + 255) / 256) * 256, 1, 1),
                                      MTL::Size(256, 1, 1), {rope}, context, 10000000.0f); commands.commit(); }
    inputIds = device.empty(uint64_t(maxBatchTokens) * sizeof(int32_t), true);
    batchKvValid = device.empty(maxBatchSequences * sizeof(uint32_t), true);
    queryStartLoc = device.empty((maxBatchSequences + 1) * sizeof(uint32_t), true);
    sequenceSlots = device.empty(maxBatchSequences * sizeof(uint32_t), true);
    stateBanks = device.empty(maxBatchSequences * sizeof(uint32_t), true);
    outputTokens = device.empty(maxLogitRows * sizeof(int32_t), true);
    logitRows = device.empty(maxLogitRows * sizeof(uint32_t), true);
    std::array<uint64_t, maxBatchSequences> seeds{}; uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (uint64_t& value : seeds) value = ++seed; rng = device.empty(sizeof(seeds), true);
    std::memcpy(rng.buffer->metalBuffer->contents(), seeds.data(), sizeof(seeds));
    if (hasMtp) {
        draftPositions = device.empty(uint64_t(maxDraftTokens) * maxBatchSequences * sizeof(uint32_t), true);
        draftTokens = device.empty(uint64_t(maxDraftTokens + 1) * maxBatchSequences * sizeof(int32_t), true);
        std::array<_Float16, 2 * maxBatchSequences * 4096> zeroSeeds{};
        mtpSeeds = device.upload(zeroSeeds.data(), sizeof(zeroSeeds));
    }
    for (uint32_t i = 0; i < layers.size(); ++i) if (!layers[i].fullAttention) {
        gdnOffsets[i] = gdnBytes; gdnBytes += recurrentStateBytes + convStateBytes;
        for (uint32_t bank = 0; bank < 2; ++bank) {
            states[i].conv[bank] = device.empty(convStateBytes * maxBatchSequences);
            states[i].recurrent[bank] = device.empty(recurrentStateBytes * maxBatchSequences);
        }
    }
    cacheNamespace = std::hash<std::string>{}(std::filesystem::absolute(path).string());
    workspace.padRows = kernels.padRows; workspace.ensure(device, 1, hasMtp);
    this->kv = std::make_unique<SparseKV>(device, maxBatchSequences * maxLogicalBlocks, blocks.size(),
                                          targetKvLayers, hasMtp);
    std::printf("GGUF weights loaded in %.3fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
}
}  // namespace infeng::qwen35
