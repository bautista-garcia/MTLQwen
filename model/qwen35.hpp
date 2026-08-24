#pragma once
#include "backend/metal/device.hpp"
#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace infeng::qwen35 {
using metal::CommandBuffer;
using metal::Device;
using metal::Pipeline;
using metal::SparseBuffer;
using metal::Tensor;

inline constexpr uint32_t blockTokens = 128, targetKvPlanes = 16, mtpKvPlanes = 2;
inline constexpr uint32_t maxDraftTokens = 4, maxBatchSequences = 8, maxBatchTokens = 128;
inline constexpr uint32_t maxLogitRows = maxBatchSequences * (maxDraftTokens + 1), gdnCheckpointTokens = 512;
inline constexpr uint32_t unbound = UINT32_MAX;
inline constexpr uint64_t convStateBytes = 8192 * 4 * 2, recurrentStateBytes = uint64_t(32) * 128 * 128 * 4;

enum class QuantType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ4_XS = 23 };
struct Linear {
    Tensor weight; Pipeline *decode = nullptr, *prefill = nullptr;
    uint32_t k = 0, n = 0; uint16_t decodeThreads = 0; uint8_t outputsPerGroup = 0; QuantType type{};
};
struct MlpWeights {
    Linear gate, up, down; Pipeline* fusedDecode = nullptr;
    uint16_t decodeThreads = 0; uint8_t outputsPerGroup = 0;
};
struct AttentionWeights { Linear q, k, v, out; Tensor qNorm, kNorm; };
struct GdnWeights { Linear qkv, z, b, a, out; Tensor conv, norm, dt, A; };
struct Layer {
    bool fullAttention = false; uint8_t kvIndex = 0; Tensor inputNorm, postNorm;
    MlpWeights mlp; AttentionWeights attention; GdnWeights gdn;
};
struct MtpWeights { Linear fusion; Layer layer; Tensor embeddingNorm, hiddenNorm, outputNorm; };
struct Kernels {
    Pipeline *embed, *rms, *add, *siluMul, *padRows, *initRope, *argmax, *sample, *mtpDraftFuse;
    Pipeline *attention, *attentionGate, *unpackAttention, *ropeQk, *mtpStore;
    Pipeline *gdnPrepare, *gdnConv, *gdnConvCandidates, *splitQk, *deltaPrefill, *deltaDecode, *deltaCandidates,
             *gdnNorm, *mtpFuse, *gatherRows;
};
struct Scratch {
    bool decodeMode = false; Pipeline* padRows = nullptr;
    Tensor hidden[2], inputNorm, postNorm, padInput, mlpGate, mlpUp, mlpMixed;
    Tensor attnQG, attnK, attnV, attnQNorm, attnKNorm, attnQRope, attnKRope, attnOut, attnGated;
    Tensor gdnMixed, gdnZ, gdnB, gdnA, gdnG, gdnConvolved, gdnQ, gdnK, gdnV, gdnDelta, gdnNormed;
    Tensor mid, projected, mtpFused, targetHidden, targetLogits;
    void ensure(Device& device, uint32_t rows, bool mtp);
};
struct LayerState { Tensor conv[2], recurrent[2], candidateConv, candidateRecurrent; };
struct PhysicalBlock { uint32_t refs = 0; uint64_t hash = 0, touch = 0; };
struct HybridCheckpoint { Tensor arena; uint64_t touch = 0; };
struct Session;
struct Query {
    Session* session = nullptr; uint32_t count = 1, output = 0, logit = unbound, accepted = 0;
    bool draft = false, sample = false, terminal = false;
};
struct Batch {
    std::array<Query, maxBatchSequences> queries{}; std::array<uint32_t, maxBatchSequences + 1> starts{}, candidates{};
    uint32_t size = 0, drafts = 0;
};
struct Model {
    Device device; uint32_t maxContext; std::unique_ptr<SparseBuffer> kv;
    Tensor embedding, norm, rope; Linear head; std::array<Layer, 32> layers; MtpWeights mtp;
    std::array<LayerState, 32> states; Kernels kernels{}; Scratch workspace;
    Tensor inputIds, batchKvValid, queryStartLoc, draftPositions, sequenceSlots, stateBanks;
    Tensor draftTokens, outputTokens, rng, mtpSeeds, logitRows;
    std::vector<PhysicalBlock> blocks; std::unordered_map<uint64_t, uint32_t> prefixTable;
    std::unordered_map<uint64_t, HybridCheckpoint> checkpointCache;
    std::array<bool, maxBatchSequences> slots{}; std::mutex control;
    uint64_t nextSequenceId = 1, clock = 0, cacheNamespace = 0;
    uint32_t physicalBlocks = 0, maxLogicalBlocks, gdnBytes = 0, candidateCapacity = 0;
    std::array<uint32_t, 32> gdnOffsets{}; uint64_t parameterCount = 0, modelBytes = 0; bool hasMtp = false;
    Model(const std::filesystem::path& weights, const std::filesystem::path& kernels, uint32_t maxContext, bool profile);
    Scratch& scratch(uint32_t query, uint32_t rows) {
        workspace.decodeMode = query <= maxDraftTokens + 1; workspace.ensure(device, rows, hasMtp); return workspace;
    }
    uint8_t acquireSlot(); uint32_t acquireBlock(); void reserve(const Batch& batch); void ensureCandidates(uint32_t rows);
    void bind(Session& session, uint32_t logical, uint32_t physical); void unbind(Session& session, uint32_t logical);
    void lookupPrefix(Session& session); void publishPrefix(Session& session, uint32_t oldValid); void release(Session& session);
};
enum class SequencePhase : uint8_t { pp, tg, eos };
struct Session {
    Model& model; uint64_t sequenceId; std::vector<int32_t> request; std::vector<uint32_t> bindings;
    uint32_t kvValid = 0; SequencePhase phase = SequencePhase::pp; uint8_t slot, bank = 0;
    bool speculative = false; uint8_t draftCount = 2; std::deque<int32_t> ready; int32_t lastReturned = -1;
    std::vector<int32_t> stopTokens; uint64_t draftedTokens = 0, acceptedTokens = 0;
    Session(Model& model); ~Session();
};

Tensor mtpSeed(Model& model, const Session& session, uint32_t bank);
void draft(Model& model, Batch& batch);
void forward(Model& model, Batch& batch, float temperature, float topP, int32_t topK);
void step(Model& model, Session* const* sessions, const int32_t* ids, const uint32_t* inputStarts, uint32_t count,
          float temperature, float topP, int32_t topK, int32_t* output, uint8_t* ready);
}  // namespace infeng::qwen35
