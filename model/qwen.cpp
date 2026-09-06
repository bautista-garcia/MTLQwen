#include "qwen35.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
namespace infeng::qwen35 {
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
    uint32_t accepted = 0, oldValid = sequence.kvValid;
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
      commitCandidate(*copies, sequence, query.state + accepted, row, accepted);
    }
    if (query.logit != unbound && (!draft || !acceptedStop))
      sequence.request.push_back(sampled[query.logit + accepted]);
    sequence.kvValid += draft ? 1 + accepted : query.count;
    sequence.drafted += draft ? drafts : 0;
    sequence.accepted += accepted;
    if (sequence.temperature > 0 && query.logit != unbound)
      rng.contents<uint64_t>()[sequence.slot] = sampledRng.contents<uint64_t>()[query.logit + accepted - acceptedStop];
    query.state = oldValid;
    sequence.bank ^= 1;
    sequence.active = sequence.active && sequence.kvValid < maxContext &&
                      (query.logit == unbound || !std::count(sequence.stops.begin(), sequence.stops.end(), sequence.request.back()));
  }
  if (copies)
    copies->commit();
  for (uint32_t row = 0; row < batch.size; ++row)
    publishPrefix(*batch.queries[row].sequence, batch.queries[row].state, batch.queries[row].sequence->kvValid);
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
API void* infeng_engine_create(const char* weights, const char* draftWeights, const char* kernels, uint32_t context, uint32_t drafter, char* error) {
  return create([&] { return new Engine(weights, kernels, context, Drafter(drafter), draftWeights ? draftWeights : ""); }, error);
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
  uint64_t values[]{engine.parameterCount, engine.modelBytes, engine.kv->mappedBytes()};
  return values[field];
}
API uint64_t infeng_sequence_info(void* value, uint32_t field) {
  Sequence& sequence = *static_cast<Sequence*>(value);
  std::lock_guard lock(sequence.engine.mutex);
  uint64_t values[]{sequence.drafted, sequence.accepted, sequence.kvValid};
  return values[field];
}
