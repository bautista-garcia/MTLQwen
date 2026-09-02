#pragma once
#include "backend/metal/device.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace infeng::qwen35 {
using metal::CommandBuffer;
using metal::Device;
using metal::Pipeline;
using metal::SparseKV;
using metal::Tensor;

inline constexpr uint32_t blockTokens = 128, targetLayers = 32, fullAttentionInterval = 4, targetKvLayers = 8;
inline constexpr uint32_t mtpLayers = 1, dflashLayers = 6;
inline constexpr uint32_t maxDraftTokens = 7, maxBatchSequences = 8, maxDecodeRows = 5, maxBatchTokens = blockTokens;
inline constexpr uint32_t maxLogitRows = maxBatchSequences * (maxDraftTokens + 1), vocabSize = 248320, dflashMaskToken = 248077;
inline constexpr uint32_t gdnCheckpointTokens = 512, maxGdnCheckpoints = 8;
inline constexpr uint32_t unbound = UINT32_MAX;
inline constexpr uint64_t convStateBytes = 8192 * 4 * 2, recurrentStateBytes = uint64_t(32) * 128 * 128 * 4;
inline constexpr uint64_t gdnCheckpointBytes = (targetLayers - targetKvLayers) * (recurrentStateBytes + convStateBytes);
inline constexpr uint64_t gdnOffset(uint32_t layer) {
  return uint64_t(layer - layer / fullAttentionInterval) * (recurrentStateBytes + convStateBytes);
}

enum class QuantType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ4_XS = 23 };
enum class Drafter : uint8_t { none, mtp, dflash };
enum LinearKernel : uint8_t { linearDecode, linearDecodeAdd, linearPrefill, linearPrefillSmall };
struct Linear {
  Tensor weight;
  Pipeline* pipeline[4]{};
  uint32_t k = 0, n = 0;
  uint8_t outputsPerGroup = 0;
};
struct MlpWeights {
  Linear gate, up, down;
  Pipeline* fusedDecode = nullptr;
  uint8_t outputsPerGroup = 0;
};
struct AttentionWeights {
  Linear q, k, v, out;
  Tensor qNorm, kNorm;
};
struct GdnWeights {
  Linear qkv, z, out;
  Tensor b, a, conv, norm, dt, A;
  bool baF32 = false;
};
struct Layer {
  bool fullAttention = false;
  uint8_t kvIndex = 0;
  Tensor inputNorm, postNorm;
  MlpWeights mlp;
  AttentionWeights attention;
  GdnWeights gdn;
};
struct MtpWeights {
  Linear fusion;
  Layer layer;
  Tensor embeddingNorm, hiddenNorm, outputNorm;
};
struct DflashWeights {
  Linear fusion;
  std::array<Layer, dflashLayers> layers;
  Tensor hiddenNorm, outputNorm;
};
struct Scratch {
  bool decodeMode = false;
  Pipeline* padRows = nullptr;
  Tensor hidden[2], inputNorm, postNorm, padInput, mlpGate, mlpUp, mlpMixed;
  Tensor attnQG, attnK, attnV, attnQRope, attnKRope, attnOut, attnGated, attnPartials;
  Tensor gdnMixed, gdnZ, gdnB, gdnG, gdnConvolved, gdnQ, gdnK, gdnV, gdnDelta, gdnNormed;
  Tensor mid, projected, mtpFused, targetHidden, dflashFeatures, dflashContext, targetLogits;
  void ensure(Device& device, uint32_t rows, Drafter drafter);
};
struct LayerState {
  Tensor conv[2], recurrent[2], candidateConv, candidateRecurrent;
};
struct PhysicalBlock {
  uint32_t refs = 0;
  uint64_t hash = 0, touch = 0;
};
struct HybridCheckpoint {
  Tensor arena;
  uint64_t touch = 0;
};
struct Sequence;
struct Query {
  Sequence* sequence = nullptr;
  const int32_t* tokens = nullptr;
  uint32_t valid = 0, count = 1, logit = unbound;
  bool draft = false, sample = false;
};
struct Batch {
  std::array<Query, maxBatchSequences> queries{};
  std::array<uint32_t, maxBatchSequences + 1> query_start_loc{}; // First packed token row for each query.
  std::array<uint32_t, maxBatchSequences + 1> state_start_loc{}; // First temporary GDN-state snapshot for each query.
  uint32_t size = 0, drafts = 0;
};
struct Model {
  Device device;
  uint32_t maxContext;
  std::unique_ptr<SparseKV> kv;
  Tensor embedding, norm, rope, dflashRope;
  Linear head;
  std::array<Layer, targetLayers> layers;
  MtpWeights mtp;
  DflashWeights dflash;
  Drafter drafter = Drafter::none;
  std::array<LayerState, targetLayers> states;
  Scratch workspace;
  Tensor inputIds, batchKvValid, queryStartLoc, draftPositions, sequenceSlots, stateBanks;
  Tensor draftTokens, outputTokens, rng, mtpSeeds, logitRows;
  std::vector<PhysicalBlock> blocks;
  std::unordered_map<uint64_t, uint32_t> prefixTable;
  std::unordered_map<uint64_t, HybridCheckpoint> checkpointCache;
  std::array<bool, maxBatchSequences> slots{};
  uint64_t clock = 0;
  uint32_t physicalBlocks = 0, maxLogicalBlocks, candidateCapacity = 0;
  uint64_t parameterCount = 0, modelBytes = 0;
  Batch pending;
  std::array<uint64_t, maxBatchSequences> rngBefore{};
  bool pendingSampling = false;
  Model(const std::filesystem::path& weights, const std::filesystem::path& kernels, uint32_t maxContext, bool profile, Drafter drafter,
        const std::filesystem::path& draftWeights);
  bool hasMtp() const {
    return drafter == Drafter::mtp;
  }
  bool hasDflash() const {
    return drafter == Drafter::dflash;
  }
  Scratch& scratch(uint32_t query, uint32_t rows) {
    workspace.decodeMode = query <= maxDecodeRows;
    workspace.ensure(device, rows, drafter);
    return workspace;
  }
  uint8_t acquireSlot();
  uint32_t acquireBlock();
  void reserve(const Batch& batch);
  void ensureCandidates(uint32_t rows);
  void bind(Sequence& sequence, uint32_t logical, uint32_t physical);
  uint32_t lookupPrefix(Sequence& sequence, const int32_t* tokens, uint32_t length);
  void publishPrefix(Sequence& sequence, const int32_t* tokens, uint32_t tokenOffset, uint32_t oldValid, uint32_t valid);
  void release(Sequence& sequence);
  void execute(Batch& batch, float temperature, float topP, int32_t topK);
  void commit(const int32_t* tokens, const uint32_t* starts, const uint32_t* offsets, const uint32_t* valid, const uint8_t* accepted);
  void abort();
};
struct Sequence {
  Model& model;
  std::vector<uint32_t> bindings;
  uint8_t slot, bank = 0;
  Sequence(Model& model);
  ~Sequence();
};

Tensor mtpSeed(Model& model, const Sequence& sequence, uint32_t bank);
void draft(Model& model, Batch& batch);
void forward(Model& model, Batch& batch, float temperature, float topP, int32_t topK);
} // namespace infeng::qwen35
