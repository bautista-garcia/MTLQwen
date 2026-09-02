#include "qwen35.hpp"
#include <stdexcept>

namespace infeng::qwen35 {
void Model::execute(Batch& batch, float temperature, float topP, int32_t topK) {
  pending = batch;
  pendingSampling = temperature > 0;
  auto* rngData = rng.contents<uint64_t>();
  for (uint32_t row = 0; row < pending.size; ++row)
    if (pending.queries[row].sample)
      rngBefore[row] = rngData[pending.queries[row].sequence->slot];
  reserve(pending);
  if (pending.drafts)
    draft(*this, pending);
  forward(*this, pending, temperature, topP, topK);
}

void Model::commit(const int32_t* tokens, const uint32_t* starts, const uint32_t* offsets, const uint32_t* valid, const uint8_t* accepted) {
  if (pending.drafts) {
    CommandBuffer copies(device, 0);
    for (uint32_t row = 0; row < pending.size; ++row) {
      Query& query = pending.queries[row];
      if (!query.draft)
        continue;
      Sequence& sequence = *query.sequence;
      uint32_t source = pending.state_start_loc[row] + accepted[row];
      for (uint32_t i = 0; i < layers.size(); ++i)
        if (!layers[i].fullAttention) {
          copies.copy(states[i].candidateRecurrent.view(uint64_t(source) * recurrentStateBytes, recurrentStateBytes),
                      states[i].recurrent[1 - sequence.bank].view(uint64_t(sequence.slot) * recurrentStateBytes, recurrentStateBytes),
                      recurrentStateBytes);
          copies.copy(states[i].candidateConv.view(uint64_t(source) * convStateBytes, convStateBytes),
                      states[i].conv[1 - sequence.bank].view(uint64_t(sequence.slot) * convStateBytes, convStateBytes), convStateBytes);
        }
      if (hasMtp())
        copies.copy(workspace.targetHidden.view(uint64_t(pending.query_start_loc[row] + accepted[row]) * 8192, 8192),
                    mtpSeed(*this, sequence, 1 - sequence.bank), 8192);
    }
    copies.commit();
  }
  auto* rngData = rng.contents<uint64_t>();
  for (uint32_t row = 0; row < pending.size; ++row) {
    Query& query = pending.queries[row];
    Sequence& sequence = *query.sequence;
    if (pendingSampling && query.sample) {
      uint32_t consumed = query.draft ? accepted[row] + (starts[row + 1] - starts[row] > valid[row]) : 1;
      uint64_t state = rngBefore[row];
      while (consumed--) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
      }
      rngData[sequence.slot] = state;
    }
    sequence.bank ^= 1;
    publishPrefix(sequence, tokens + starts[row], offsets[row], query.valid, valid[row]);
  }
  pending = {};
  pendingSampling = false;
}

void Model::abort() {
  if (pendingSampling) {
    auto* rngData = rng.contents<uint64_t>();
    for (uint32_t row = 0; row < pending.size; ++row)
      if (pending.queries[row].sample)
        rngData[pending.queries[row].sequence->slot] = rngBefore[row];
  }
  pending = {};
  pendingSampling = false;
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
      return R(-1);
  }
}
Model& model(void* value) {
  return *static_cast<Model*>(value);
}
} // namespace

#define API extern "C" __attribute__((visibility("default")))
struct InfengInfo {
  uint64_t parameters, weightBytes, mappedBytes, gpuTime, passes;
};

API const char* infeng_last_error() {
  return lastError.c_str();
}
API void* infeng_model_create(const char* weights, const char* draftWeights, const char* kernels, uint32_t context, int32_t profile,
                              uint32_t drafter) {
  return guard([&] { return new Model(weights, kernels, context, profile != 0, Drafter(drafter), draftWeights ? draftWeights : ""); });
}
API void infeng_model_release(void* value) {
  delete static_cast<Model*>(value);
}
API void* infeng_sequence_create(void* value) {
  return guard([&] { return new Sequence(model(value)); });
}
API void infeng_sequence_release(void* value) {
  delete static_cast<Sequence*>(value);
}
API int32_t infeng_prefix(void* value, const int32_t* tokens, uint32_t length, uint32_t* valid) {
  return guard([&] {
    Sequence& owner = *static_cast<Sequence*>(value);
    *valid = owner.model.lookupPrefix(owner, tokens, length);
    return 0;
  });
}
API int32_t infeng_forward(void* value, void* const* sequences, const int32_t* tokens, const uint32_t* starts, const uint32_t* valid,
                           const uint32_t* counts, const uint8_t* flags, uint32_t count, uint32_t drafts, float temperature, float topP, int32_t topK,
                           const int32_t** output) {
  return guard([&] {
    Model& owner = model(value);
    infeng::qwen35::Batch batch;
    batch.size = count;
    batch.drafts = drafts;
    for (uint32_t row = 0; row < count; ++row) {
      batch.queries[row] = {static_cast<Sequence*>(sequences[row]),
                            tokens + starts[row],
                            valid[row],
                            counts[row],
                            infeng::qwen35::unbound,
                            bool(flags[row] & 1),
                            bool(flags[row] & 2)};
      batch.query_start_loc[row + 1] = batch.query_start_loc[row] + counts[row];
      batch.state_start_loc[row + 1] = batch.state_start_loc[row] + ((flags[row] & 1) ? counts[row] : 0);
    }
    try {
      owner.execute(batch, temperature, topP, topK);
      output[0] = owner.outputTokens.contents<int32_t>();
      output[1] = owner.draftTokens.contents<int32_t>();
    } catch (...) {
      owner.abort();
      throw;
    }
    return 0;
  });
}
API int32_t infeng_commit(void* value, const int32_t* tokens, const uint32_t* starts, const uint32_t* offsets, const uint32_t* valid,
                          const uint8_t* accepted) {
  return guard([&] {
    model(value).commit(tokens, starts, offsets, valid, accepted);
    return 0;
  });
}
API void infeng_abort(void* value) {
  model(value).abort();
}
API void infeng_info(void* value, InfengInfo* output) {
  Model& owner = model(value);
  const auto& counters = owner.device.stats;
  *output = {owner.parameterCount, owner.modelBytes, owner.kv->mappedBytes(), counters.gpuTimeNs, counters.passes};
}
API uint32_t infeng_kernel_counter_count(void* value) {
  return model(value).device.kernelCounters().size();
}
API void infeng_kernel_counter(void* value, uint32_t index, const char** phase, const char** name, uint64_t* gpuTime, uint64_t* launches) {
  const auto& counter = model(value).device.kernelCounters()[index];
  *phase = counter.phase.c_str();
  *name = counter.name.c_str();
  *gpuTime = counter.gpuTimeNs;
  *launches = counter.launches;
}
