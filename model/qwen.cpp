#include "qwen35.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
namespace infeng::qwen35 {
Engine::~Engine() {
  {
    std::lock_guard lock(mutex);
    closing = true;
  }
  condition.notify_all();
  worker.join();
}
void Engine::schedule() {
  std::unique_lock lock(mutex);
  auto ready = [&] {
    return std::count_if(sequences.begin(), sequences.end(),
                         [](Sequence* sequence) { return sequence && sequence->active && !sequence->busy && sequence->request.size() > sequence->kvValid; });
  };
  while (true) {
    condition.wait(lock, [&] { return closing || ready(); });
    if (closing)
      return;
    auto first = std::find_if(sequences.begin(), sequences.end(),
                              [](Sequence* sequence) { return sequence && sequence->active && sequence->request.size() > sequence->kvValid; });
    uint32_t live = std::count_if(sequences.begin(), sequences.end(), [](Sequence* sequence) { return sequence; });
    if (ready() < maxBatchSequences) {
      uint32_t delay = live > 1 ? ((*first)->request.size() - (*first)->kvValid > 1 ? 5000 : 250) : 50;
      condition.wait_for(lock, std::chrono::microseconds(delay), [&] { return closing || ready() == maxBatchSequences; });
    }
    if (closing)
      return;
    Batch batch;
    for (Sequence* sequence : sequences)
      if (sequence && sequence->active && !sequence->busy && sequence->request.size() > sequence->kvValid) {
        sequence->busy = true;
        batch.queries[batch.size++].sequence = sequence;
      }
    if (!batch.size)
      continue;
    running = true;
    lock.unlock();
    std::exception_ptr error;
    try {
      execute(batch);
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    running = false;
    for (uint32_t row = 0; row < batch.size; ++row) {
      Sequence* sequence = batch.queries[row].sequence;
      sequence->busy = false;
      if (error) {
        sequence->error = error;
        sequence->active = false;
      }
    }
    condition.notify_all();
  }
}
void Engine::execute(Batch& batch) {
  uint32_t drafts = maxDraftTokens, draftRows = 0, prompts = 0, stateRows = 0;
  for (uint32_t row = 0; row < batch.size; ++row) {
    Sequence& sequence = *batch.queries[row].sequence;
    if (!sequence.kvValid)
      sequence.kvValid = lookupPrefix(sequence, sequence.request.data(), sequence.request.size());
    uint32_t pending = sequence.request.size() - sequence.kvValid;
    prompts += pending > 1;
    uint32_t room = std::min(maxContext, (sequence.kvValid / gdnCheckpointTokens + 1) * gdnCheckpointTokens) - sequence.kvValid;
    uint32_t available = room > 1 ? std::min(sequence.draftTokens, room - 1) : 0;
    if (pending == 1 && available) {
      drafts = std::min(drafts, available);
      ++draftRows;
    }
  }
  if (!draftRows)
    drafts = 0;
  uint32_t budget = maxBatchTokens - batch.size - draftRows * drafts;
  for (uint32_t row = 0, left = prompts; row < batch.size; ++row) {
    Sequence& sequence = *batch.queries[row].sequence;
    uint32_t pending = sequence.request.size() - sequence.kvValid, count = 1;
    uint32_t room = std::min(maxContext, (sequence.kvValid / gdnCheckpointTokens + 1) * gdnCheckpointTokens) - sequence.kvValid;
    bool prompt = pending > 1;
    bool speculative = !prompt && drafts && sequence.draftTokens && room > 1;
    if (prompt) {
      count += std::min(std::min(pending, room) - 1, (budget + left - 1) / left);
      budget -= count - 1;
      --left;
    } else if (speculative)
      count += drafts;
    batch.queries[row] = {&sequence, count, !prompt || count == pending ? 0 : unbound, speculative ? stateRows : unbound};
    stateRows += speculative ? count : 0;
  }
  std::array<uint64_t, maxBatchSequences> rngBefore{};
  auto* rngData = rng.contents<uint64_t>();
  for (uint32_t row = 0; row < batch.size; ++row)
    if (batch.queries[row].logit != unbound && batch.queries[row].sequence->temperature > 0)
      rngBefore[row] = rngData[batch.queries[row].sequence->slot];
  reserve(batch);
  if (drafts)
    draft(*this, batch, drafts);
  forward(*this, batch, drafts, stateRows);
  auto *sampled = outputTokens.contents<int32_t>(), *proposed = draftTokens.contents<int32_t>();
  auto* starts = queryStartLoc.contents<uint32_t>();
  std::array<uint8_t, maxBatchSequences> acceptedCounts{};
  std::array<bool, maxBatchSequences> acceptedStops{};
  Device* copies = drafts ? &device.command() : nullptr;
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    bool draft = query.state != unbound;
    uint8_t& accepted = acceptedCounts[row];
    bool& acceptedStop = acceptedStops[row];
    if (draft) {
      uint32_t draftRow = query.state / query.count;
      for (; accepted < drafts; ++accepted) {
        int32_t token = proposed[(accepted + 1) * maxBatchSequences + draftRow];
        if (sampled[query.logit + accepted] != token)
          break;
        if (std::count(sequence.stops.begin(), sequence.stops.end(), token)) {
          ++accepted;
          acceptedStop = true;
          break;
        }
      }
      uint32_t source = query.state + accepted;
      for (uint32_t i = 0; i < layers.size(); ++i)
        if ((i + 1) % fullAttentionInterval) {
          GdnState candidates = gdnState(candidateStates, candidateStates.bytes / gdnCheckpointBytes, i);
          GdnState state = gdnState(gdnStates[1 - sequence.bank], maxBatchSequences, i);
          copies->copy(candidates.recurrent.view(uint64_t(source) * recurrentStateBytes, recurrentStateBytes),
                       state.recurrent.view(uint64_t(sequence.slot) * recurrentStateBytes, recurrentStateBytes));
          copies->copy(candidates.conv.view(uint64_t(source) * convStateBytes, convStateBytes),
                       state.conv.view(uint64_t(sequence.slot) * convStateBytes, convStateBytes));
        }
      if (drafter == Drafter::mtp)
        copies->copy(workspace.targetHidden.view(uint64_t(starts[row] + accepted) * 8192, 8192),
                     mtpSeed(*this, sequence, 1 - sequence.bank));
    }
  }
  if (copies)
    copies->commit();
  std::lock_guard lock(mutex);
  for (uint32_t row = 0; row < batch.size; ++row) {
    Query& query = batch.queries[row];
    Sequence& sequence = *query.sequence;
    bool draft = query.state != unbound, acceptedStop = acceptedStops[row];
    uint32_t accepted = acceptedCounts[row];
    uint32_t draftRow = draft ? query.state / query.count : 0;
    query.state = sequence.kvValid;
    if (draft) {
      for (uint32_t step = 0; step < accepted; ++step)
        sequence.request.push_back(proposed[(step + 1) * maxBatchSequences + draftRow]);
      if (!acceptedStop)
        sequence.request.push_back(sampled[query.logit + accepted]);
      sequence.kvValid += 1 + accepted;
    } else {
      sequence.kvValid += query.count;
      if (query.logit != unbound)
        sequence.request.push_back(sampled[query.logit]);
    }
    if (sequence.temperature > 0 && query.logit != unbound) {
      uint64_t state = rngBefore[row];
      for (uint32_t count = draft ? accepted + !acceptedStop : 1; count; --count) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
      }
      rngData[sequence.slot] = state;
    }
    if (draft) {
      sequence.drafted += drafts;
      sequence.accepted += accepted;
    }
    sequence.bank ^= 1;
    int32_t last = sequence.request.back();
    bool stopped = query.logit != unbound && std::count(sequence.stops.begin(), sequence.stops.end(), last);
    sequence.active = sequence.active && !stopped && sequence.kvValid < maxContext;
  }
  for (uint32_t row = 0; row < batch.size; ++row)
    publishPrefix(*batch.queries[row].sequence, batch.queries[row].state, batch.queries[row].sequence->kvValid);
}
} // namespace infeng::qwen35
using namespace infeng::qwen35;
namespace {
thread_local std::string lastError;
template <class F> auto guard(F&& function) {
  using R = decltype(function());
  try {
    return function();
  } catch (const std::exception& error) {
    lastError = error.what();
    if constexpr (std::is_pointer_v<R>)
      return R(nullptr);
    else
      return R(-2);
  }
}
Engine& engine(void* value) {
  return *static_cast<Engine*>(value);
}
} // namespace
#define API extern "C" __attribute__((visibility("default")))
struct InfengInfo {
  uint64_t parameters, weightBytes, mappedBytes, drafted, accepted;
  uint32_t valid;
};
API const char* infeng_last_error() {
  return lastError.c_str();
}
API void* infeng_engine_create(const char* weights, const char* draftWeights, const char* kernels, uint32_t context, uint32_t drafter) {
  return guard([&] { return new Engine(weights, kernels, context, Drafter(drafter), draftWeights ? draftWeights : ""); });
}
API void infeng_engine_release(void* value) {
  delete static_cast<Engine*>(value);
}
API void* infeng_sequence_create(void* value, const int32_t* stops, uint32_t stopCount, float temperature, float topP, int32_t topK,
                                 uint32_t draftTokens) {
  return guard([&] { return new Sequence(engine(value), stops, stopCount, temperature, topP, topK, draftTokens); });
}
API void infeng_sequence_release(void* value) {
  delete static_cast<Sequence*>(value);
}
API int32_t infeng_sequence_append(void* value, const int32_t* tokens, uint32_t count) {
  return guard([&] {
    Sequence& sequence = *static_cast<Sequence*>(value);
    std::lock_guard lock(sequence.engine.mutex);
    if (sequence.active || sequence.busy) throw std::runtime_error("sequence is still generating");
    if ((!count && sequence.request.size() == sequence.kvValid) || sequence.request.size() + count > sequence.engine.maxContext)
      throw std::runtime_error("completion has no available model context");
    sequence.request.insert(sequence.request.end(), tokens, tokens + count);
    sequence.error = nullptr;
    sequence.active = true;
    sequence.engine.condition.notify_all();
    return int32_t(sequence.request.size());
  });
}
API int32_t infeng_sequence_read(void* value, uint32_t cursor) {
  return guard([&] {
    Sequence& sequence = *static_cast<Sequence*>(value);
    std::unique_lock lock(sequence.engine.mutex);
    sequence.engine.condition.wait(lock, [&] { return sequence.error || cursor < sequence.request.size() || (!sequence.active && !sequence.busy); });
    if (sequence.error)
      std::rethrow_exception(sequence.error);
    return cursor < sequence.request.size() ? sequence.request[cursor] : -1;
  });
}
API void infeng_sequence_cancel(void* value) {
  Sequence& sequence = *static_cast<Sequence*>(value);
  std::lock_guard lock(sequence.engine.mutex);
  sequence.active = false;
  sequence.engine.condition.notify_all();
}
API void infeng_info(void* value, void* sequenceValue, InfengInfo* output) {
  Engine& owner = engine(value);
  Sequence* sequence = static_cast<Sequence*>(sequenceValue);
  std::unique_lock lock(owner.mutex);
  owner.condition.wait(lock, [&] { return !owner.running; });
  *output = {owner.parameterCount, owner.modelBytes, owner.kv->mappedBytes(), sequence ? sequence->drafted : 0, sequence ? sequence->accepted : 0,
             sequence ? sequence->kvValid : 0};
}
