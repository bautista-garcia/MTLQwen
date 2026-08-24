#include "qwen35.hpp"
#include <stdexcept>

using infeng::qwen35::Model;
using infeng::qwen35::Session;
namespace {
thread_local std::string lastError;
template <class F> int32_t status(F&& function) {
    try { function(); return 0; } catch (const std::exception& error) { lastError = error.what(); return -1; }
}
Model& model(void* value) { return *static_cast<Model*>(value); }
Session& sequence(void* value) { return *static_cast<Session*>(value); }
}
#define API extern "C" __attribute__((visibility("default")))
struct InfengCounters { uint64_t gpu_time_ns, passes; };
struct InfengSpecCounters { uint64_t drafted_tokens, accepted_tokens; };
API const char* infeng_last_error() { return lastError.c_str(); }
API void* infeng_model_create(const char* weights, const char* kernels, uint32_t context, int32_t profile) {
    try { return new Model(weights, kernels, context, profile != 0); }
    catch (const std::exception& error) { lastError = error.what(); return nullptr; }
}
API void infeng_model_release(void* value) { delete static_cast<Model*>(value); }
API void* infeng_session_create(void* value) {
    try { return new Session(model(value)); }
    catch (const std::exception& error) { lastError = error.what(); return nullptr; }
}
API void infeng_session_release(void* value) { delete static_cast<Session*>(value); }
API int32_t infeng_session_configure(void* pointer, int32_t speculative, uint32_t drafts, const int32_t* stops,
                                     uint32_t stopCount) {
    return status([&] {
        auto& session = sequence(pointer); std::lock_guard lock(session.model.control);
        if (speculative && !session.model.hasMtp) throw std::runtime_error("speculative decoding requires a Qwen3.5 MTP GGUF");
        if (!drafts || drafts > infeng::qwen35::maxDraftTokens)
            throw std::runtime_error("draft token count must be between 1 and 4");
        if (stopCount && !stops) throw std::runtime_error("stop token pointer is null");
        session.speculative = speculative; session.draftCount = drafts; session.stopTokens.clear();
        if (stopCount) session.stopTokens.assign(stops, stops + stopCount);
    });
}
API int32_t infeng_forward_batch(void* const* pointers, const int32_t* ids, const uint32_t* starts, uint32_t count,
                                 float temperature, float topP, int32_t topK, int32_t* output, uint8_t* ready) {
    return status([&] {
        if (!count || count > infeng::qwen35::maxBatchSequences)
            throw std::runtime_error("batch must contain 1 to 8 sequences");
        std::array<Session*, infeng::qwen35::maxBatchSequences> sessions{};
        for (uint32_t i = 0; i < count; ++i) sessions[i] = static_cast<Session*>(pointers[i]);
        infeng::qwen35::step(sessions[0]->model, sessions.data(), ids, starts, count, temperature, topP, topK,
                            output, ready);
    });
}
API uint64_t infeng_session_length(void* pointer) {
    auto& session = sequence(pointer);
    std::lock_guard lock(session.model.control); return session.kvValid;
}
API uint64_t infeng_session_id(void* pointer) { return sequence(pointer).sequenceId; }
API uint32_t infeng_session_pending_outputs(void* pointer) {
    auto& session = sequence(pointer);
    std::lock_guard lock(session.model.control); return session.ready.size();
}
API uint64_t infeng_session_mapped_bytes(void* pointer) {
    auto& session = sequence(pointer); std::lock_guard lock(session.model.control);
    return session.model.kv->mappedBytes();
}
API uint64_t infeng_model_parameter_count(void* value) { return model(value).parameterCount; }
API uint64_t infeng_model_weight_bytes(void* value) { return model(value).modelBytes; }
API uint64_t infeng_model_vocab_size(void*) { return 248320; }
API int32_t infeng_model_has_mtp(void* value) { return model(value).hasMtp; }
API int32_t infeng_session_spec_counters(void* pointer, InfengSpecCounters* output) {
    return status([&] { auto& session = sequence(pointer);
        std::lock_guard lock(session.model.control); *output = {session.draftedTokens, session.acceptedTokens}; });
}
API int32_t infeng_model_counters(void* pointer, InfengCounters* output) {
    return status([&] { const auto& value = model(pointer).device.counters();
        *output = {value.gpuTimeNs, value.passes}; });
}
API uint32_t infeng_model_kernel_counter_count(void* pointer) {
    return model(pointer).device.kernelCounters().size();
}
API int32_t infeng_model_kernel_counter(void* pointer, uint32_t index, const char** phase, const char** name,
                                        uint64_t* gpuTime, uint64_t* launches) {
    return status([&] { const auto& values = model(pointer).device.kernelCounters();
        if (index >= values.size()) throw std::runtime_error("kernel counter index out of range");
        const auto& value = values[index]; *phase = value.phase.c_str(); *name = value.name.c_str();
        *gpuTime = value.gpuTimeNs; *launches = value.launches; });
}
