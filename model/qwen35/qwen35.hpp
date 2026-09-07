#pragma once
#include "backend/metal/device.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
namespace infeng::qwen35 {
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
};
struct Layer {
  Tensor inputNorm, postNorm;
  MlpWeights mlp;
  AttentionWeights attention;
  GdnWeights gdn;
};
struct DrafterWeights {
  Linear fusion;
  std::array<Layer, dflashLayers> layers;
  Tensor embeddingNorm, hiddenNorm, outputNorm;
};
struct Scratch {
  bool decodeMode = false;
  Tensor hidden[2], inputNorm, postNorm, mlpGate, mlpUp;
  Tensor attnQG, attnK, attnV, attnQRope, attnKRope, attnOut, attnPartials;
  Tensor gdnMixed, gdnZ, gdnB, gdnG, gdnConvolved, gdnQ, gdnK, gdnV;
  Tensor mid, targetHidden, dflashFeatures, targetLogits;
  void ensure(Device& device, uint32_t rows, Drafter drafter);
};
struct GdnState {
  Tensor conv, recurrent;
};
inline GdnState gdnState(const Tensor& arena, uint32_t rows, uint32_t layer) {
  uint64_t offset = gdnOffset(layer) * rows;
  return {arena.view(offset, convStateBytes * rows), arena.view(offset + convStateBytes * rows, recurrentStateBytes * rows)};
}
struct PhysicalBlock {
  uint32_t refs = 0;
  uint64_t touch = 0;
};
struct HybridCheckpoint {
  Tensor arena;
  std::vector<uint32_t> bindings;
  uint64_t touch = 0;
};
struct Sequence;
struct Query {
  Sequence* sequence = nullptr;
  uint32_t count = 1, logit = unbound, state = unbound;
};
struct Batch {
  std::array<Query, maxBatchSequences> queries{};
  uint32_t size = 0;
};
struct Engine {
  Device device;
  uint32_t maxContext;
  std::unique_ptr<SparseKV> kv;
  Tensor embedding, norm, rope, dflashRope;
  Linear head;
  std::array<Layer, targetLayers> layers;
  DrafterWeights draftModel;
  Drafter drafter = Drafter::none;
  Tensor gdnStates[2], candidateStates;
  Scratch workspace;
  Tensor inputIds, batchKvValid, queryStartLoc, draftPositions, sequenceSlots, stateBanks;
  Tensor draftTokens, outputTokens, sampledRng, rng, mtpSeeds, logitRows;
  std::vector<PhysicalBlock> blocks;
  std::unordered_map<uint64_t, HybridCheckpoint> checkpointCache;
  std::array<Sequence*, maxBatchSequences> sequences{};
  std::mutex mutex;
  std::condition_variable condition;
  std::thread worker;
  std::atomic_bool closing = false;
  uint64_t clock = 0;
  uint32_t physicalBlocks = 0;
  uint64_t parameterCount = 0, modelBytes = 0;
  Engine(const std::filesystem::path& weights, const std::filesystem::path& kernels, uint32_t maxContext, Drafter drafter,
         const std::filesystem::path& draftWeights);
  ~Engine();
  Scratch& scratch(uint32_t query, uint32_t rows) {
    workspace.decodeMode = query <= maxDecodeRows;
    workspace.ensure(device, rows, drafter);
    return workspace;
  }
  uint32_t acquireBlock();
  bool reserve(const Batch& batch);
  void bind(Sequence& sequence, uint32_t logical, uint32_t physical);
  void commitCandidate(Device& commands, Sequence& sequence, uint32_t stateRow, uint32_t queryRow, uint32_t accepted);
  uint32_t lookupPrefix(Sequence& sequence, const int32_t* tokens, uint32_t length);
  void publishPrefix(Sequence& sequence, uint32_t oldValid, uint32_t valid);
  void schedule();
  bool execute(Batch& batch);
};
struct Sequence {
  Engine& engine;
  std::vector<int32_t> request;
  std::vector<uint32_t> bindings;
  std::vector<int32_t> stops;
  bool error = false;
  uint64_t drafted = 0, accepted = 0, prefixHash = 0;
  uint32_t kvValid = 0;
  uint32_t slot, bank = 0, draftTokens;
  float temperature, topP;
  int32_t topK;
  bool active = false, busy = false;
  Sequence(Engine&, const int32_t* stops, uint32_t stopCount, float temperature, float topP, int32_t topK, uint32_t draftTokens);
  ~Sequence();
};
Tensor mtpSeed(Engine& engine, const Sequence& sequence, uint32_t bank);
void copyGdnState(Device& commands, const Tensor& source, uint32_t sourceRows, uint32_t sourceRow, const Tensor& destination,
                  uint32_t destinationRows, uint32_t destinationRow);
void draft(Engine& engine, Batch& batch, uint32_t drafts);
void forward(Engine& engine, Batch& batch, uint32_t drafts, uint32_t stateRows);
} // namespace infeng::qwen35
