#include "model/qwen35/qwen35.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace infeng::qwen35 {
namespace {
// hash is the previous prefix hash, chained with the new 512 token block
uint64_t hashCheckpoint(uint64_t hash, const int32_t* tokens) {
  hash ^= 0x9e3779b97f4a7c15ull;
  for (uint32_t i = 0; i < gdnCheckpointTokens; ++i) {
    hash ^= uint32_t(tokens[i]) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
    hash *= 0x100000001b3ull;
  }
  return hash;
}

struct State {
  const Tensor& gdn;
  uint32_t row;
  Tensor mtp;
};

// copy gdnState for publish/restoring prefix checkpoints and candidate commits
void copyState(Device& commands, const State& source, const State& destination) {
  uint32_t sourceRows = source.gdn.bytes / gdnCheckpointBytes, destinationRows = destination.gdn.bytes / gdnCheckpointBytes;
  for (uint32_t layer = 0; layer < targetLayers; ++layer)
    if ((layer + 1) % fullAttentionInterval) {
      GdnState from = gdnState(source.gdn, sourceRows, layer), to = gdnState(destination.gdn, destinationRows, layer);
      commands.copy(from.recurrent.view(uint64_t(source.row) * recurrentStateBytes, recurrentStateBytes),
                    to.recurrent.view(uint64_t(destination.row) * recurrentStateBytes, recurrentStateBytes));
      commands.copy(from.conv.view(uint64_t(source.row) * convStateBytes, convStateBytes),
                    to.conv.view(uint64_t(destination.row) * convStateBytes, convStateBytes));
    }
  if (source.mtp.buffer)
    commands.copy(source.mtp, destination.mtp);
}
} // namespace

Tensor mtpSeed(Engine& model, const Sequence& sequence, uint32_t bank) {
  return model.mtpSeeds.view(uint64_t(bank * maxBatchSequences + sequence.slot) * 8192, 8192);
}

void Engine::bind(Sequence& sequence, uint32_t physical) {
  uint32_t logical = sequence.bindings.size();
  kv->map(uint32_t(sequence.slot) * blocks.size() + logical, physical);
  sequence.bindings.push_back(physical);
  ++blocks[physical].refs;
}

bool Engine::reserve(const Batch& batch) {
  for (uint32_t row = 0; row < batch.size; ++row) {
    const Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    for (uint32_t logical = sequence.kvValid / blockTokens; logical <= (sequence.kvValid + query.count - 1) / blockTokens; ++logical) {
      if (logical == sequence.bindings.size()) {
        uint32_t physical = physicalBlocks;
        if (physical < blocks.size()) {
          kv->ensure(physical + 1);
          ++physicalBlocks;
        } else {
          auto victim = std::min_element(blocks.begin(), blocks.end(), [](const PhysicalBlock& a, const PhysicalBlock& b) {
            return (a.refs ? UINT64_MAX : a.touch) < (b.refs ? UINT64_MAX : b.touch);
          });
          if (victim->refs)
            return false;
          physical = victim - blocks.begin();
          checkpointCache.erase(std::remove_if(checkpointCache.begin(), checkpointCache.end(),
                                               [physical](const HybridCheckpoint& checkpoint) {
                                                 return std::find(checkpoint.bindings.begin(), checkpoint.bindings.end(), physical) !=
                                                        checkpoint.bindings.end();
                                               }),
                                checkpointCache.end());
        }
        bind(sequence, physical);
      }
      blocks[sequence.bindings[logical]].touch = ++clock;
    }
  }
  return true;
}

void Engine::restorePrefix(Sequence& sequence) {
  uint32_t limit = (sequence.request.size() - 1) / gdnCheckpointTokens;
  uint64_t hash = 0;
  auto checkpoint = checkpointCache.end();
  // this can be updated to a hash map if the prefix cache size increase makes it worthwile
  for (uint32_t index = 0; index < limit; ++index) {
    hash = hashCheckpoint(hash, sequence.request.data() + uint64_t(index) * gdnCheckpointTokens);
    auto found = std::find_if(checkpointCache.begin(), checkpointCache.end(), [hash](const HybridCheckpoint& entry) { return entry.hash == hash; });
    if (found != checkpointCache.end())
      checkpoint = found;
  }
  if (checkpoint == checkpointCache.end())
    return;
  HybridCheckpoint& restored = *std::rotate(checkpoint, checkpoint + 1, checkpointCache.end());
  sequence.prefixHash = restored.hash;
  uint64_t touch = ++clock;
  Device& copies = device.command();
  copyState(copies, {restored.arena, 0, drafter == Drafter::mtp ? restored.arena.view(gdnCheckpointBytes, 8192) : Tensor{}},
            {gdnStates[0], sequence.slot, drafter == Drafter::mtp ? mtpSeed(*this, sequence, 0) : Tensor{}});
  copies.commit();
  for (uint32_t physical : restored.bindings) {
    bind(sequence, physical);
    blocks[physical].touch = touch;
  }
  sequence.kvValid = restored.bindings.size() * blockTokens;
}

void Engine::publishPrefix(Sequence& sequence) {
  if (sequence.kvValid % gdnCheckpointTokens)
    return;
  uint64_t hash = sequence.prefixHash = hashCheckpoint(sequence.prefixHash, sequence.request.data() + sequence.kvValid - gdnCheckpointTokens);
  auto found =
      std::find_if(checkpointCache.begin(), checkpointCache.end(), [hash](const HybridCheckpoint& checkpoint) { return checkpoint.hash == hash; });
  if (found != checkpointCache.end()) {
    std::rotate(found, found + 1, checkpointCache.end());
    return;
  }
  if (checkpointCache.size() < maxGdnCheckpoints)
    checkpointCache.push_back({0, device.empty(gdnCheckpointBytes + (drafter == Drafter::mtp ? 8192 : 0))});
  else
    std::rotate(checkpointCache.begin(), checkpointCache.begin() + 1, checkpointCache.end());
  HybridCheckpoint& checkpoint = checkpointCache.back();
  checkpoint.hash = hash;
  checkpoint.bindings = sequence.bindings;
  Device& copies = device.command();
  copyState(copies, {gdnStates[sequence.bank], sequence.slot, drafter == Drafter::mtp ? mtpSeed(*this, sequence, sequence.bank) : Tensor{}},
            {checkpoint.arena, 0, drafter == Drafter::mtp ? checkpoint.arena.view(gdnCheckpointBytes, 8192) : Tensor{}});
  copies.commit();
}

Sequence::Sequence(Engine& owner, const int32_t* stopTokens, uint32_t stopCount, float samplingTemperature, float samplingTopP, int32_t samplingTopK,
                   bool useSpeculation)
    : engine(owner), temperature(samplingTemperature), topP(samplingTopP), topK(samplingTopK), speculative(useSpeculation) {
  std::lock_guard lock(engine.mutex);
  request.reserve(owner.maxContext + 1);
  stops.assign(stopTokens, stopTokens + stopCount);
  slot = std::find(engine.sequences.begin(), engine.sequences.end(), nullptr) - engine.sequences.begin();
  if (slot == maxBatchSequences)
    throw std::runtime_error("maximum live sequence count reached");
  engine.sequences[slot] = this;
}

Sequence::~Sequence() {
  std::unique_lock lock(engine.mutex);
  active = false;
  engine.condition.wait(lock, [&] { return !busy; });
  for (uint32_t logical = 0; logical < bindings.size(); ++logical) {
    engine.kv->map(uint32_t(slot) * engine.blocks.size() + logical, unbound);
    --engine.blocks[bindings[logical]].refs;
  }
  engine.sequences[slot] = nullptr;
}

Engine::~Engine() {
  closing = true;
  condition.notify_all();
  worker.join();
}

void Engine::schedule() {
  std::unique_lock lock(mutex);
  auto ready = [](Sequence* sequence) { return sequence && sequence->active; };
  while (true) {
    // delay for coalescing arriving sequences can be added if execute() time becomes too high
    condition.wait(lock, [&] { return closing || std::any_of(sequences.begin(), sequences.end(), ready); });
    if (closing)
      return;
    Batch batch;
    std::array<Query*, maxBatchSequences> prefills;
    uint32_t prefillCount = 0, remaining = maxBatchTokens;
    for (Sequence* sequence : sequences)
      if (ready(sequence)) {
        sequence->busy = true;
        if (!sequence->kvValid)
          restorePrefix(*sequence);
        Query& query = batch.queries[batch.size++];
        query.sequence = sequence;
        query.pending = sequence->request.size() - sequence->kvValid;
        query.room = std::min(maxContext, (sequence->kvValid / gdnCheckpointTokens + 1) * gdnCheckpointTokens) - sequence->kvValid;
        // draft width to each t.g sequence (without crossing 512 token boundary)
        if (query.pending > 1) {
          prefills[prefillCount++] = &query;
          continue;
        }
        query.logit = 0;
        if (sequence->speculative && query.room > draftWidth) {
          query.count += draftWidth;
          query.state = batch.candidateRows;
          batch.candidateRows += query.count;
        }
        remaining -= query.count;
      }
    // remaining space for p.p queries
    for (uint32_t row = 0; row < prefillCount; ++row) {
      Query& query = *prefills[row];
      query.count = std::min({query.pending, query.room, remaining / (prefillCount - row)});
      query.logit = query.count == query.pending ? 0 : unbound;
      remaining -= query.count;
    }
    lock.unlock();
    bool success = execute(batch);
    lock.lock();
    for (uint32_t row = 0; row < batch.size; ++row) {
      Sequence* sequence = batch.queries[row].sequence;
      sequence->busy = false;
      sequence->error = !success;
      sequence->active &= success;
    }
    condition.notify_all();
  }
}

bool Engine::execute(Batch& batch) {
  // allocate KV capacity
  if (!reserve(batch))
    return false;
  if (batch.candidateRows)
    draft(*this, batch);
  forward(*this, batch, batch.candidateRows);
  auto *sampled = outputTokens.contents<int32_t>(), *proposed = draftTokens.contents<int32_t>();
  Device* copies = batch.candidateRows ? &device.command() : nullptr;
  std::lock_guard lock(mutex);
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    if (!sequence.active) {
      query.count = 0;
      continue;
    }
    bool speculative = query.state != unbound;
    uint32_t accepted = 0;
    bool stopped = false;
    auto isStop = [&](int32_t token) { return std::find(sequence.stops.begin(), sequence.stops.end(), token) != sequence.stops.end(); };
    if (speculative) {
      uint32_t draftRow = query.state / query.count;
      while (accepted < draftWidth && sampled[query.logit + accepted] == proposed[(accepted + 1) * maxBatchSequences + draftRow]) {
        int32_t token = proposed[(accepted + 1) * maxBatchSequences + draftRow];
        sequence.request.push_back(token);
        ++accepted;
        if ((stopped = isStop(token)))
          break;
      }
      copyState(*copies,
                {candidateStates, query.state + accepted,
                 drafter == Drafter::mtp ? workspace.targetHidden.view(uint64_t(queryStartLoc.contents<uint32_t>()[row] + accepted) * 8192, 8192)
                                         : Tensor{}},
                {gdnStates[1 - sequence.bank], sequence.slot, drafter == Drafter::mtp ? mtpSeed(*this, sequence, 1 - sequence.bank) : Tensor{}});
    }
    if (sequence.temperature > 0 && query.logit != unbound)
      rng.contents<uint64_t>()[sequence.slot] = sampledRng.contents<uint64_t>()[query.logit + accepted - stopped];
    if (query.logit != unbound && !stopped) {
      int32_t token = sampled[query.logit + accepted];
      sequence.request.push_back(token);
      stopped = isStop(token);
    }
    sequence.kvValid += speculative ? 1 + accepted : query.count;
    sequence.drafted += speculative ? draftWidth : 0;
    sequence.accepted += accepted;
    sequence.bank ^= 1;
    sequence.active = sequence.kvValid < maxContext && !stopped;
  }
  if (copies)
    copies->commit();
  for (uint32_t row = 0; row < batch.size; ++row)
    if (batch.queries[row].count)
      publishPrefix(*batch.queries[row].sequence);
  return true;
}
} // namespace infeng::qwen35

using namespace infeng::qwen35;

template <class F> static void* create(F&& function, char* error) {
  try {
    return function();
  } catch (const std::exception& exception) {
    std::strcpy(error, exception.what());
    return nullptr;
  }
}

#define API extern "C" __attribute__((visibility("default")))

API void* infeng_engine_create(const char* weights, const char* draftWeights, const char* kernels, uint32_t context, char* error) {
  return create([&] { return new Engine(weights, kernels, context, draftWeights ? draftWeights : ""); }, error);
}

API void infeng_engine_release(void* value) {
  delete static_cast<Engine*>(value);
}

API void* infeng_sequence_create(void* value, const int32_t* stops, uint32_t stopCount, float temperature, float topP, int32_t topK,
                                 uint32_t speculative, char* error) {
  return create([&] { return new Sequence(*static_cast<Engine*>(value), stops, stopCount, temperature, topP, topK, speculative); }, error);
}

API void infeng_sequence_release(void* value) {
  delete static_cast<Sequence*>(value);
}

API int32_t infeng_sequence_append(void* value, const int32_t* tokens, uint32_t count) {
  Sequence& sequence = *static_cast<Sequence*>(value);
  std::lock_guard lock(sequence.engine.mutex);
  if (count == UINT32_MAX) {
    sequence.active = false;
    sequence.engine.condition.notify_all();
    return 0;
  }
  if (sequence.active || sequence.busy)
    return -3;
  if ((!count && sequence.request.size() == sequence.kvValid) || sequence.request.size() + count > sequence.engine.maxContext)
    return -4;
  sequence.request.insert(sequence.request.end(), tokens, tokens + count);
  sequence.error = false;
  sequence.active = true;
  sequence.engine.condition.notify_all();
  return int32_t(sequence.request.size());
}

API int32_t infeng_sequence_read(void* value, uint32_t cursor) {
  Sequence& sequence = *static_cast<Sequence*>(value);
  std::unique_lock lock(sequence.engine.mutex);
  sequence.engine.condition.wait(lock, [&] { return sequence.error || cursor < sequence.request.size() || (!sequence.active && !sequence.busy); });
  if (sequence.error)
    return -5;
  return cursor < sequence.request.size() ? sequence.request[cursor] : -1;
}

API uint64_t infeng_engine_info(void* value, uint32_t field) {
  Engine& engine = *static_cast<Engine*>(value);
  uint64_t values[]{engine.parameterCount, engine.modelBytes, engine.kv->mappedBytes(), uint64_t(engine.drafter), engine.draftWidth};
  return values[field];
}

API uint64_t infeng_sequence_info(void* value, uint32_t field) {
  Sequence& sequence = *static_cast<Sequence*>(value);
  std::lock_guard lock(sequence.engine.mutex);
  uint64_t values[]{sequence.drafted, sequence.accepted, sequence.kvValid};
  return values[field];
}
