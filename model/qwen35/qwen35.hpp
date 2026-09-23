#pragma once
#include "backend/metal/device.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <thread>
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
enum class Drafter : uint8_t { none, mtp, dflash };

struct Kernel {
  Pipeline* pipeline = nullptr;
  uint32_t threads = 0, group = 0;
};

struct Weight {
  Tensor tensor;
  uint32_t n, k, type;
  Kernel kernels[4]{};
};

using Weights = std::unordered_map<std::string, Weight>;

struct Scratch {
  Tensor hidden[2], norm, temporary, mlpGate, mlpUp;
  Tensor mixed, q, k, v, attnQRope, attnKRope, attnPartials;
  Tensor gdnB, gdnG, gdnConvolved;
  Tensor mid, draftContext, targetLogits;
  void allocate(Device& device, Drafter drafter);
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
  uint64_t hash = 0;
  Tensor arena;
  std::vector<uint32_t> bindings;
};
struct Sequence;
struct Engine;

struct Query {
  Sequence* sequence = nullptr;
  uint32_t pending = 1, room = 1, start = 0, count = 1, logit = unbound, state = unbound;
};

struct Batch {
  std::array<Query, maxBatchSequences> queries{};
  uint32_t size = 0, candidateRows = 0, rows = 0, packed = 0, logits = 0, maxQuery = 0;
  void pack(Engine& engine, bool drafting);
};

struct Engine {
  Device device;
  uint32_t maxContext, draftWidth = 0;
  std::unique_ptr<SparseKV> kv;
  Weights weights[2];
  Tensor rope, dflashRope;
  Drafter drafter = Drafter::none;
  Tensor gdnStates[2], candidateStates;
  Scratch workspace;
  Tensor inputIds, batchKvValid, queryStartLoc, draftPositions, sequenceSlots, stateBanks;
  Tensor draftTokens, outputTokens, sampledRng, rng, mtpSeeds, logitRows;
  std::vector<PhysicalBlock> blocks;
  std::vector<HybridCheckpoint> checkpointCache;
  std::array<Sequence*, maxBatchSequences> sequences{};
  std::mutex mutex;
  std::condition_variable condition;
  std::thread worker;
  std::atomic_bool closing = false;
  uint64_t clock = 0;
  uint32_t physicalBlocks = 0;
  uint64_t parameterCount = 0, modelBytes = 0;
  Engine(const std::filesystem::path& weights, const std::filesystem::path& kernels, uint32_t maxContext, const std::filesystem::path& draftWeights);
  ~Engine();
  void loadModel(const std::filesystem::path& weights, const std::filesystem::path& draftWeights);

  bool reserve(const Batch& batch);
  void bind(Sequence& sequence, uint32_t physical);
  void restorePrefix(Sequence& sequence);
  void publishPrefix(Sequence& sequence);
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
  uint32_t slot, bank = 0;
  float temperature, topP;
  int32_t topK;
  bool speculative, active = false, busy = false;
  Sequence(Engine&, const int32_t* stops, uint32_t stopCount, float temperature, float topP, int32_t topK, bool speculative);
  ~Sequence();
};

Tensor mtpSeed(Engine& engine, const Sequence& sequence, uint32_t bank);
void forward(Engine& engine, Batch& batch, bool drafting = false, bool prepare = false);
} // namespace infeng::qwen35
