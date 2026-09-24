#pragma once
#include "backend/metal/device.hpp"
#include "backend/metal/kernel/batch.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <list>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace infeng::qwen35 {
using metal::Device;
using metal::Pipeline;
using metal::SparseKV;
using metal::Tensor;
inline constexpr uint32_t blockTokens = 128, targetLayers = 32, fullAttentionInterval = 4, targetKvLayers = 8;
inline constexpr uint32_t mtpLayers = 1, dflashLayers = 6;
inline constexpr uint32_t maxDraftTokens = 7, maxBatchSequences = 8, maxDecodeRows = 5, maxBatchTokens = blockTokens;
inline constexpr uint32_t maxLogitRows = maxBatchSequences * (maxDraftTokens + 1), vocabSize = 248320;
inline constexpr uint32_t gdnCheckpointTokens = 512, maxGdnCheckpoints = 8;
inline constexpr uint32_t unbound = UINT32_MAX;

enum class Drafter : uint8_t { none, mtp, dflash };

struct Kernel {
  Pipeline* pipeline = nullptr;
  uint32_t threads = 0, group = 0;
};

struct Weight : Tensor {
  uint32_t n, k, type;
  Kernel kernels[4]{};
};

using Weights = std::unordered_map<std::string, Weight>;

struct Scratch {
  Tensor hidden, norm, temporary, mlpGate;
  Tensor mixed, q, k, v, attnQRope, attnKRope, attnPartials;
  Tensor gdnB, gdnG;
  Tensor draftContext, targetLogits;
};

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

struct Batch {
  GpuQuery* queries = nullptr;
  uint32_t size = 0, candidateRows = 0;
};

struct Engine {
  std::atomic_uint32_t owners = 1; // One API handle and one reference per live sequence.
  Device device;
  uint32_t maxContext, draftWidth = 0;
  std::unique_ptr<SparseKV> kv;
  Weights weights[2];
  Tensor rope, dflashRope;
  Drafter drafter = Drafter::none;
  Tensor candidateStates;
  Scratch workspace;
  Tensor requestData, queryData;
  Tensor draftTokens, outputTokens, sampledRng, rng;
  std::vector<PhysicalBlock> blocks;
  std::list<HybridCheckpoint> checkpointCache;
  std::array<Sequence*, maxBatchSequences> sequences{};
  std::array<std::pair<Tensor, Tensor>, maxBatchSequences> statePool;
  std::mutex mutex;
  std::condition_variable condition;
  std::thread worker;
  bool closing = false;
  uint64_t clock = 0;
  uint64_t parameterCount = 0, modelBytes = 0;
  Engine(const std::filesystem::path& weights, const std::filesystem::path& kernels, uint32_t maxContext, const std::filesystem::path& draftWeights);
  ~Engine();
  void loadModel(const std::filesystem::path& weights, const std::filesystem::path& draftWeights);

  bool reserve(const Batch& batch);
  void bind(Sequence& sequence, uint32_t physical);
  void restorePrefix(Sequence& sequence);
  void publishPrefix(Sequence& sequence);
  void schedule();
};

struct Sequence {
  Engine& engine;
  int32_t* request;
  uint32_t requested = 0;
  std::vector<uint32_t> bindings;
  std::vector<int32_t> stops;
  bool error = false;
  uint64_t drafted = 0, accepted = 0, prefixHash = 0;
  uint32_t kvValid = 0;
  uint32_t slot;
  float temperature, topP;
  int32_t topK;
  bool speculative, active = false, busy = false;
  Sequence(Engine&, const int32_t* stops, uint32_t stopCount, float temperature, float topP, int32_t topK, bool speculative);
  ~Sequence();
};

void forward(Engine& engine, const Batch& batch, bool drafting = false, uint32_t prepareRows = 0);
} // namespace infeng::qwen35
