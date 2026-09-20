#include "model/qwen35/qwen35.hpp"
#include <algorithm>
#include <chrono>
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

uint32_t Engine::acquireBlock() {
  if (physicalBlocks < blocks.size()) {
    kv->ensure(physicalBlocks + 1);
    return physicalBlocks++;
  }
  auto victim = std::min_element(blocks.begin(), blocks.end(), [](const PhysicalBlock& a, const PhysicalBlock& b) {
    return (a.refs ? UINT64_MAX : a.touch) < (b.refs ? UINT64_MAX : b.touch);
  });
  if (victim->refs)
    return unbound;
  uint32_t physical = victim - blocks.begin();
  for (auto checkpoint = checkpointCache.begin(); checkpoint != checkpointCache.end();)
    if (std::find(checkpoint->second.bindings.begin(), checkpoint->second.bindings.end(), physical) != checkpoint->second.bindings.end())
      checkpoint = checkpointCache.erase(checkpoint);
    else
      ++checkpoint;
  return physical;
}

bool Engine::reserve(const Batch& batch) {
  for (uint32_t row = 0; row < batch.size; ++row) {
    Sequence& sequence = *batch.queries[row].sequence;
    for (uint32_t logical = sequence.kvValid / blockTokens; logical <= (sequence.kvValid + batch.queries[row].count - 1) / blockTokens; ++logical) {
      if (logical == sequence.bindings.size()) {
        uint32_t physical = acquireBlock();
        if (physical == unbound)
          return false;
        bind(sequence, physical);
      }
      blocks[sequence.bindings[logical]].touch = ++clock;
    }
  }
  return true;
}

uint32_t Engine::lookupPrefix(Sequence& sequence, const int32_t* tokens, uint32_t length) {
  uint32_t limit = (length - 1) / gdnCheckpointTokens;
  uint64_t hash = 0;
  HybridCheckpoint* checkpoint = nullptr;
  for (uint32_t index = 0; index < limit; ++index) {
    hash = hashCheckpoint(hash, tokens + uint64_t(index) * gdnCheckpointTokens);
    auto found = checkpointCache.find(hash);
    if (found != checkpointCache.end()) {
      checkpoint = &found->second;
      sequence.prefixHash = hash;
    }
  }
  if (!checkpoint)
    return 0;
  checkpoint->touch = ++clock;
  Device& copies = device.command();
  copyState(copies, {checkpoint->arena, 0, drafter == Drafter::mtp ? checkpoint->arena.view(gdnCheckpointBytes, 8192) : Tensor{}},
            {gdnStates[0], sequence.slot, drafter == Drafter::mtp ? mtpSeed(*this, sequence, 0) : Tensor{}});
  copies.commit();
  for (uint32_t physical : checkpoint->bindings)
    bind(sequence, physical);
  sequence.bank = 0;
  return checkpoint->bindings.size() * blockTokens;
}

void Engine::publishPrefix(Sequence& sequence) {
  try {
    if (sequence.kvValid % gdnCheckpointTokens)
      return;
    sequence.prefixHash = hashCheckpoint(sequence.prefixHash, sequence.request.data() + sequence.kvValid - gdnCheckpointTokens);
    if (checkpointCache.count(sequence.prefixHash)) {
      checkpointCache.at(sequence.prefixHash).touch = ++clock;
      return;
    }
    if (checkpointCache.size() == maxGdnCheckpoints) {
      auto victim = std::min_element(checkpointCache.begin(), checkpointCache.end(),
                                     [](const auto& a, const auto& b) { return a.second.touch < b.second.touch; });
      checkpointCache.erase(victim);
    }
    HybridCheckpoint checkpoint{device.empty(gdnCheckpointBytes + (drafter == Drafter::mtp ? 8192 : 0)), sequence.bindings, ++clock};
    Device& copies = device.command();
    copyState(copies, {gdnStates[sequence.bank], sequence.slot, drafter == Drafter::mtp ? mtpSeed(*this, sequence, sequence.bank) : Tensor{}},
              {checkpoint.arena, 0, drafter == Drafter::mtp ? checkpoint.arena.view(gdnCheckpointBytes, 8192) : Tensor{}});
    copies.commit();
    checkpointCache.emplace(sequence.prefixHash, std::move(checkpoint));
  } catch (...) {
  }
}

Sequence::Sequence(Engine& owner, const int32_t* stopTokens, uint32_t stopCount, float samplingTemperature, float samplingTopP, int32_t samplingTopK,
                   uint32_t drafts)
    : engine(owner), draftTokens(drafts), temperature(samplingTemperature), topP(samplingTopP), topK(samplingTopK) {
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
  auto ready = [](Sequence* sequence) { return sequence && sequence->active && sequence->request.size() > sequence->kvValid; };
  while (true) {
    condition.wait(lock, [&] { return closing || std::any_of(sequences.begin(), sequences.end(), ready); });
    uint32_t delay = std::count_if(sequences.begin(), sequences.end(), [](Sequence* sequence) { return sequence; }) > 1 ? 250 : 50;
    condition.wait_for(lock, std::chrono::microseconds(delay),
                       [&] { return closing || std::count_if(sequences.begin(), sequences.end(), ready) == maxBatchSequences; });
    if (closing)
      return;
    Batch batch;
    for (Sequence* sequence : sequences)
      if (ready(sequence)) {
        sequence->busy = true;
        batch.queries[batch.size++].sequence = sequence;
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
  uint32_t drafts = 0, draftRows = 0, stateRows = 0;
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    if (!sequence.kvValid)
      sequence.kvValid = lookupPrefix(sequence, sequence.request.data(), sequence.request.size());
    query.count = sequence.request.size() - sequence.kvValid;
    query.state = std::min(maxContext, (sequence.kvValid / gdnCheckpointTokens + 1) * gdnCheckpointTokens) - sequence.kvValid;
    uint32_t available = query.count == 1 && query.state > 1 ? std::min(sequence.draftTokens, query.state - 1) : 0;
    drafts = available ? (drafts ? std::min(drafts, available) : available) : drafts;
    draftRows += available != 0;
  }
  uint32_t budget = maxBatchTokens - batch.size - draftRows * drafts;
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    bool prompt = query.count > 1;
    uint32_t count = prompt ? std::min({query.count, query.state, budget + 1}) : 1 + (sequence.draftTokens && query.state > 1 ? drafts : 0);
    budget -= prompt * (count - 1);
    query = {&sequence, count, !prompt || count == query.count ? 0 : unbound, !prompt && count > 1 ? stateRows : unbound};
    stateRows += (!prompt && count > 1) * count;
  }
  if (!reserve(batch))
    return false;
  if (drafts)
    draft(*this, batch, drafts);
  forward(*this, batch, drafts, stateRows);
  auto *sampled = outputTokens.contents<int32_t>(), *proposed = draftTokens.contents<int32_t>();
  Device* copies = drafts ? &device.command() : nullptr;
  std::lock_guard lock(mutex);
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    bool draft = query.state != unbound;
    uint32_t accepted = 0;
    bool acceptedStop = false;
    uint32_t draftRow = draft ? query.state / query.count : 0;
    if (draft) {
      while (accepted < drafts && sampled[query.logit + accepted] == proposed[(accepted + 1) * maxBatchSequences + draftRow]) {
        int32_t token = proposed[(accepted + 1) * maxBatchSequences + draftRow];
        sequence.request.push_back(token);
        ++accepted;
        if ((acceptedStop = std::count(sequence.stops.begin(), sequence.stops.end(), token)))
          break;
      }
      copyState(*copies,
                {candidateStates, query.state + accepted,
                 drafter == Drafter::mtp ? workspace.targetHidden.view(uint64_t(queryStartLoc.contents<uint32_t>()[row] + accepted) * 8192, 8192)
                                         : Tensor{}},
                {gdnStates[1 - sequence.bank], sequence.slot, drafter == Drafter::mtp ? mtpSeed(*this, sequence, 1 - sequence.bank) : Tensor{}});
    }
    if (query.logit != unbound && (!draft || !acceptedStop))
      sequence.request.push_back(sampled[query.logit + accepted]);
    sequence.kvValid += draft ? 1 + accepted : query.count;
    sequence.drafted += draft ? drafts : 0;
    sequence.accepted += accepted;
    if (sequence.temperature > 0 && query.logit != unbound)
      rng.contents<uint64_t>()[sequence.slot] = sampledRng.contents<uint64_t>()[query.logit + accepted - acceptedStop];
    sequence.bank ^= 1;
    sequence.active = sequence.active && sequence.kvValid < maxContext &&
                      (query.logit == unbound || !std::count(sequence.stops.begin(), sequence.stops.end(), sequence.request.back()));
  }
  if (copies)
    copies->commit();
  for (uint32_t row = 0; row < batch.size; ++row)
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
                                 uint32_t draftTokens, char* error) {
  return create([&] { return new Sequence(*static_cast<Engine*>(value), stops, stopCount, temperature, topP, topK, draftTokens); }, error);
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
  uint64_t values[]{engine.parameterCount, engine.modelBytes, engine.kv->mappedBytes(), uint64_t(engine.drafter)};
  return values[field];
}

API uint64_t infeng_sequence_info(void* value, uint32_t field) {
  Sequence& sequence = *static_cast<Sequence*>(value);
  std::lock_guard lock(sequence.engine.mutex);
  uint64_t values[]{sequence.drafted, sequence.accepted, sequence.kvValid};
  return values[field];
}
