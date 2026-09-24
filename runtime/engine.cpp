#include "model/qwen35/qwen35.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <stdexcept>

namespace infeng::qwen35 {
// Extend the preceding prefix hash with one 512-token checkpoint block.
static uint64_t hashCheckpoint(uint64_t hash, const int32_t* tokens) {
  hash ^= 0x9e3779b97f4a7c15ull;
  for (uint32_t i = 0; i < gdnCheckpointTokens; ++i) {
    hash ^= uint32_t(tokens[i]) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    hash *= 0x100000001b3ull;
  }
  return hash;
}

void Engine::bind(Sequence& sequence, uint32_t physical) {
  kv->map(sequence.slot * blocks.size() + sequence.bindings.size(), physical);
  sequence.bindings.push_back(physical);
  ++blocks[physical].refs;
}

bool Engine::reserve(const Batch& batch) {
  for (uint32_t row = 0; row < batch.size; ++row) {
    const GpuQuery& query = batch.queries[row];
    Sequence& sequence = *sequences[query.slot];
    for (uint32_t logical = sequence.kvValid / blockTokens; logical <= (sequence.kvValid + query.count - 1) / blockTokens; ++logical) {
      if (logical == sequence.bindings.size()) {
        auto victim = std::min_element(blocks.begin(), blocks.end(), [](const PhysicalBlock& a, const PhysicalBlock& b) {
          return (a.refs ? UINT64_MAX : a.touch) < (b.refs ? UINT64_MAX : b.touch);
        });
        if (victim->refs)
          return false;
        uint32_t physical = victim - blocks.begin();
        kv->ensure(physical + 1);
        checkpointCache.remove_if([physical](const HybridCheckpoint& checkpoint) {
          return std::find(checkpoint.bindings.begin(), checkpoint.bindings.end(), physical) != checkpoint.bindings.end();
        });
        bind(sequence, physical);
      }
      blocks[sequence.bindings[logical]].touch = ++clock;
    }
  }
  return true;
}

void Engine::restorePrefix(Sequence& sequence) {
  uint64_t hash = 0;
  auto checkpoint = checkpointCache.end();
  for (uint32_t index = 0; index < (sequence.requested - 1) / gdnCheckpointTokens; ++index) {
    hash = hashCheckpoint(hash, sequence.request + uint64_t(index) * gdnCheckpointTokens);
    auto found = std::find_if(checkpointCache.begin(), checkpointCache.end(), [hash](const HybridCheckpoint& entry) { return entry.hash == hash; });
    if (found != checkpointCache.end())
      checkpoint = found;
  }
  if (checkpoint == checkpointCache.end())
    return;
  checkpointCache.splice(checkpointCache.end(), checkpointCache, checkpoint);
  sequence.prefixHash = checkpoint->hash;
  uint64_t touch = ++clock;
  statePool[sequence.slot].first = checkpoint->arena;
  for (uint32_t physical : checkpoint->bindings) {
    bind(sequence, physical);
    blocks[physical].touch = touch;
  }
  sequence.kvValid = checkpoint->bindings.size() * blockTokens;
}

void Engine::publishPrefix(Sequence& sequence) {
  if (sequence.kvValid % gdnCheckpointTokens)
    return;
  sequence.prefixHash = hashCheckpoint(sequence.prefixHash, sequence.request + sequence.kvValid - gdnCheckpointTokens);
  auto found = std::find_if(checkpointCache.begin(), checkpointCache.end(),
                            [&](const HybridCheckpoint& checkpoint) { return checkpoint.hash == sequence.prefixHash; });
  if (found != checkpointCache.end())
    checkpointCache.splice(checkpointCache.end(), checkpointCache, found);
  else {
    if (checkpointCache.size() == maxGdnCheckpoints)
      checkpointCache.pop_front();
    checkpointCache.push_back({sequence.prefixHash, statePool[sequence.slot].first, sequence.bindings});
  }
}

Engine::Engine(const std::filesystem::path& path, const std::filesystem::path& kernels, uint32_t context, const std::filesystem::path& draftPath)
    : device(kernels), maxContext(context), blocks((context + blockTokens - 1) / blockTokens) {
  loadModel(path, draftPath);
  requestData = device.empty(uint64_t(maxBatchSequences) * (maxContext + 1) * sizeof(int32_t), true);
  Tensor* controls[]{&draftTokens, &outputTokens, &sampledRng, &rng};
  Tensor control = device.empty(sizeof(controls) / sizeof(*controls) * 512, true);
  for (uint32_t i = 0; i < sizeof(controls) / sizeof(*controls); ++i)
    *controls[i] = control.view(i, 1, 512);
  queryData = device.empty(2 * maxBatchSequences * sizeof(GpuQuery), true);
  for (auto& states : statePool)
    states = {device.empty(stateBytes), device.empty(stateBytes)};
  auto* seeds = rng.contents<uint64_t>();
  uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (uint32_t i = 0; i < maxBatchSequences; ++i)
    seeds[i] = ++seed;
  kv = std::make_unique<SparseKV>(device, maxBatchSequences * blocks.size(), blocks.size(), targetKvLayers,
                                  drafter == Drafter::mtp      ? mtpLayers
                                  : drafter == Drafter::dflash ? dflashLayers
                                                               : 0);
  // Resolve weights and cache each kernel variant through the model definition, without dispatching.
  for (uint32_t rows : {1u, 8u, maxBatchTokens})
    forward(*this, {}, false, rows);
  if (drafter != Drafter::none)
    forward(*this, {}, true, drafter == Drafter::dflash ? draftWidth + 1 : 1);
  worker = std::thread(&Engine::schedule, this);
}

Sequence::Sequence(Engine& owner, const int32_t* stopTokens, uint32_t stopCount, float samplingTemperature, float samplingTopP, int32_t samplingTopK,
                   bool useSpeculation)
    : engine(owner), temperature(samplingTemperature), topP(samplingTopP), topK(samplingTopK), speculative(useSpeculation) {
  std::lock_guard lock(engine.mutex);
  stops.assign(stopTokens, stopTokens + stopCount);
  slot = std::find(engine.sequences.begin(), engine.sequences.end(), nullptr) - engine.sequences.begin();
  if (slot == maxBatchSequences)
    throw std::runtime_error("maximum live sequence count reached");
  request = engine.requestData.contents<int32_t>() + slot * (engine.maxContext + 1);
  engine.sequences[slot] = this;
  ++engine.owners;
}

Sequence::~Sequence() {
  std::unique_lock lock(engine.mutex);
  active = false;
  engine.condition.wait(lock, [&] { return !busy; });
  for (uint32_t physical : bindings)
    --engine.blocks[physical].refs;
  engine.sequences[slot] = nullptr;
}

Engine::~Engine() {
  std::unique_lock lock(mutex);
  closing = true;
  condition.notify_all();
  lock.unlock();
  worker.join();
}

void Engine::schedule() {
  std::unique_lock lock(mutex);
  auto ready = [](Sequence* sequence) { return sequence && sequence->active; };
  while (true) {
    condition.wait(lock, [&] { return closing || std::any_of(sequences.begin(), sequences.end(), ready); });
    if (closing)
      return;
    Batch batch{queryData.contents<GpuQuery>()};
    std::array<GpuQuery*, maxBatchSequences> prefills;
    uint32_t prefillCount = 0, remaining = maxBatchTokens;
    for (Sequence* sequence : sequences)
      if (ready(sequence)) {
        sequence->busy = true;
        if (!sequence->kvValid)
          restorePrefix(*sequence);
        if (statePool[sequence->slot].second.buffer.use_count() > 1)
          statePool[sequence->slot].second = device.empty(stateBytes);
        GpuQuery& query = batch.queries[batch.size++];
        query = {0, 0, 0, 1, sequence->kvValid, sequence->slot, unbound, 0, 0, sequence->temperature, sequence->topP, sequence->topK};
        uint32_t pending = sequence->requested - sequence->kvValid;
        uint32_t room = std::min(maxContext, (sequence->kvValid / gdnCheckpointTokens + 1) * gdnCheckpointTokens) - sequence->kvValid;
        // draft width to each t.g sequence (without crossing 512 token boundary)
        if (pending > 1) {
          query.count = std::min(pending, room);
          prefills[prefillCount++] = &query;
          continue;
        }
        if (sequence->speculative && room > draftWidth) {
          query.count += draftWidth;
          query.state = batch.candidateRows;
          batch.candidateRows += query.count;
        }
        remaining -= query.count;
      }
    // remaining space for p.p queries
    for (uint32_t row = 0; row < prefillCount; ++row) {
      GpuQuery& query = *prefills[row];
      query.count = std::min(query.count, remaining / (prefillCount - row));
      remaining -= query.count;
    }
    bool success = reserve(batch);
    lock.unlock();
    if (success) {
      if (stateBytes * batch.candidateRows > candidateStates.bytes)
        candidateStates = device.empty(stateBytes * batch.candidateRows);
      device.command();
      if (batch.candidateRows)
        forward(*this, batch, true);
      forward(*this, batch);
      device.commit();
    }
    lock.lock();
    auto *sampled = outputTokens.contents<int32_t>(), *proposed = draftTokens.contents<int32_t>();
    Device* copies = success && batch.candidateRows ? &device.command() : nullptr;
    for (uint32_t row = 0; row < batch.size; ++row) {
      const GpuQuery& query = batch.queries[row];
      Sequence& sequence = *sequences[query.slot];
      sequence.busy = false;
      sequence.error = !success;
      sequence.active &= success;
      if (!sequence.active)
        continue;
      bool speculative = query.state != unbound;
      uint32_t accepted = 0;
      bool stopped = false;
      for (uint32_t i = 0; i < query.samples && !stopped; ++i) {
        int32_t token = sampled[query.logit + i];
        sequence.request[sequence.requested++] = token;
        if (sequence.temperature > 0)
          rng.contents<uint64_t>()[sequence.slot] = sampledRng.contents<uint64_t>()[query.logit + i];
        stopped = std::find(sequence.stops.begin(), sequence.stops.end(), token) != sequence.stops.end();
        if (!speculative || i == draftWidth || token != proposed[(i + 1) * maxBatchSequences + query.state / query.count])
          break;
        ++accepted;
      }
      if (speculative)
        copies->copy(candidateStates.view(query.state + accepted, 1, stateBytes), statePool[query.slot].second);
      sequence.kvValid += speculative ? 1 + accepted : query.count;
      sequence.drafted += speculative ? draftWidth : 0;
      sequence.accepted += accepted;
      std::swap(statePool[query.slot].first, statePool[query.slot].second);
      sequence.active = sequence.kvValid < maxContext && !stopped;
      publishPrefix(sequence);
    }
    if (copies)
      copies->commit();
    condition.notify_all();
  }
}

} // namespace infeng::qwen35

using namespace infeng::qwen35;

template <class T, class... Args> static T* create(char* error, Args&&... args) {
  try {
    return new T(std::forward<Args>(args)...);
  } catch (const std::exception& exception) {
    std::strcpy(error, exception.what());
    return nullptr;
  }
}

#define API extern "C" __attribute__((visibility("default")))

API Engine* infeng_engine_create(const char* weights, const char* draftWeights, const char* kernels, uint32_t context, char* error) {
  return create<Engine>(error, weights, kernels, context, draftWeights ? draftWeights : "");
}

API void infeng_engine_release(Engine* engine) {
  if (--engine->owners == 0)
    delete engine;
}

API Sequence* infeng_sequence_create(Engine* engine, const int32_t* stops, uint32_t stopCount, float temperature, float topP, int32_t topK,
                                     uint32_t speculative, char* error) {
  return create<Sequence>(error, *engine, stops, stopCount, temperature, topP, topK, speculative);
}

API void infeng_sequence_release(Sequence* sequence) {
  Engine* engine = &sequence->engine;
  delete sequence;
  infeng_engine_release(engine);
}

API int32_t infeng_sequence_append(Sequence* sequence, const int32_t* tokens, uint32_t count) {
  std::lock_guard lock(sequence->engine.mutex);
  if (count == UINT32_MAX) {
    sequence->active = false;
    sequence->engine.condition.notify_all();
    return 0;
  }
  if (sequence->active || sequence->busy)
    return -3;
  if ((!count && sequence->requested == sequence->kvValid) || uint64_t(sequence->requested) + count > sequence->engine.maxContext)
    return -4;
  std::copy_n(tokens, count, sequence->request + sequence->requested);
  sequence->requested += count;
  sequence->error = false;
  sequence->active = true;
  sequence->engine.condition.notify_all();
  return int32_t(sequence->requested);
}

API int32_t infeng_sequence_read(Sequence* sequence, uint32_t cursor) {
  std::unique_lock lock(sequence->engine.mutex);
  sequence->engine.condition.wait(lock, [&] { return cursor < sequence->requested || (!sequence->active && !sequence->busy); });
  if (sequence->error)
    return -5;
  return cursor < sequence->requested ? sequence->request[cursor] : -1;
}

API uint64_t infeng_engine_info(Engine* engine, uint32_t field) {
  uint64_t values[]{engine->parameterCount, engine->modelBytes, engine->kv->mappedBytes(), uint64_t(engine->drafter), engine->draftWidth};
  return values[field];
}

API uint64_t infeng_sequence_info(Sequence* sequence, uint32_t field) {
  std::lock_guard lock(sequence->engine.mutex);
  uint64_t values[]{sequence->drafted, sequence->accepted, sequence->kvValid};
  return values[field];
}
