#include "qwen35.hpp"
#include <algorithm>
#include <stdexcept>

namespace infeng::qwen35 {
namespace {
uint64_t advance(uint64_t state, uint32_t count) {
    while (count--) { state ^= state << 13; state ^= state >> 7; state ^= state << 17; } return state;
}
int32_t emit(Session& session) {
    if (session.ready.empty()) throw std::runtime_error("no generated token is ready");
    int32_t token = session.ready.front(); session.ready.pop_front(); session.lastReturned = token; return token;
}
void attach(Session& session, const int32_t* ids, uint32_t count) {
    bool fresh = session.request.empty(), generation = session.phase == SequencePhase::tg;
    if (fresh) session.request.assign(ids, ids + count);
    else if (!generation) {
        if (count <= session.kvValid || !std::equal(ids, ids + session.kvValid, session.request.begin()))
            throw std::runtime_error("prompt retry must preserve the committed token prefix");
        session.request.assign(ids, ids + count);
    } else if (session.request.size() == session.kvValid && session.kvValid && ids[0] == session.request.back())
        session.request.insert(session.request.end(), ids + 1, ids + count);
    else {
        session.request.resize(session.kvValid + 1);
        if (ids[0] == session.request[session.kvValid]) session.request.insert(session.request.end(), ids + 1, ids + count);
        else { session.request.resize(session.kvValid); session.request.insert(session.request.end(), ids, ids + count); }
    }
    if (session.request.size() <= session.kvValid) throw std::runtime_error("request has no uncomputed token");
    if (!generation || session.request.size() != session.kvValid + 1) {
        session.phase = SequencePhase::pp; if (fresh) session.model.lookupPrefix(session);
    }
}
void prepare(Model& model, Batch& batch) {
    uint32_t draftRows = 0, drafts = maxDraftTokens, prompts = 0;
    for (uint32_t row = 0; row < batch.size; ++row) {
        Query& query = batch.queries[row]; Session& session = *query.session;
        prompts += session.phase == SequencePhase::pp;
        if (session.phase == SequencePhase::tg && session.speculative) {
            uint32_t checkpoint = (session.kvValid / gdnCheckpointTokens + 1) * gdnCheckpointTokens;
            uint32_t room = std::min(model.maxContext, checkpoint) - session.kvValid;
            if (room > 1) query.draft = true, ++draftRows, drafts = std::min<uint32_t>(
                drafts, std::min<uint32_t>(session.draftCount, room - 1));
        }
    }
    batch.drafts = draftRows ? drafts : 0;
    uint32_t budget = maxBatchTokens - batch.size - draftRows * batch.drafts, left = prompts;
    for (uint32_t row = 0; row < batch.size; ++row) if (batch.queries[row].session->phase == SequencePhase::pp) {
        Query& query = batch.queries[row]; Session& session = *query.session;
        uint32_t checkpoint = (session.kvValid / gdnCheckpointTokens + 1) * gdnCheckpointTokens;
        uint32_t available = std::min<uint32_t>(session.request.size() - session.kvValid, checkpoint - session.kvValid);
        uint32_t extra = std::min(available - 1, (budget + left - 1) / left);
        query.count += extra; budget -= extra; --left;
    }
    for (uint32_t row = 0; row < batch.size; ++row) {
        Query& query = batch.queries[row]; Session& session = *query.session;
        if (query.draft) query.count = batch.drafts + 1;
        batch.starts[row + 1] = batch.starts[row] + query.count;
        batch.candidates[row + 1] = batch.candidates[row] + (query.draft ? query.count : 0);
        query.sample = query.draft || session.phase == SequencePhase::tg || session.kvValid + query.count == session.request.size();
    }
    model.reserve(batch);
}
void commit(Model& model, Batch& batch, float temperature,
            const std::array<uint64_t, maxBatchSequences>& rngBefore, int32_t* output, uint8_t* ready) {
    auto* sampled = static_cast<int32_t*>(model.outputTokens.buffer->metalBuffer->contents());
    auto* proposed = batch.drafts
        ? static_cast<int32_t*>(model.draftTokens.buffer->metalBuffer->contents()) : nullptr;
    auto* rng = static_cast<uint64_t*>(model.rng.buffer->metalBuffer->contents());
    for (uint32_t row = 0; row < batch.size; ++row) {
        Query& query = batch.queries[row];
        if (!query.draft) continue;
        Session& session = *query.session;
        for (; query.accepted < batch.drafts; ++query.accepted) {
            uint32_t draftRow = batch.candidates[row] / query.count;
            int32_t token = proposed[(query.accepted + 1) * maxBatchSequences + draftRow];
            if (sampled[query.logit + query.accepted] != token) break;
            session.ready.push_back(token);
            if (std::find(session.stopTokens.begin(), session.stopTokens.end(), token) != session.stopTokens.end()) {
                ++query.accepted; query.terminal = true; break;
            }
        }
        if (!query.terminal) session.ready.push_back(sampled[query.logit + query.accepted]);
        if (temperature > 0) rng[session.slot] = advance(rngBefore[row], query.accepted + !query.terminal);
    }
    if (batch.drafts) {
        CommandBuffer copies(model.device, 0);
        for (uint32_t row = 0; row < batch.size; ++row) if (batch.queries[row].draft) {
            const Query& query = batch.queries[row]; const Session& session = *query.session;
            uint32_t source = batch.candidates[row] + query.accepted;
            for (uint32_t i = 0; i < model.layers.size(); ++i) if (!model.layers[i].fullAttention) {
                copies.copy(model.states[i].candidateRecurrent.view(uint64_t(source) * recurrentStateBytes,
                                                                     recurrentStateBytes),
                            model.states[i].recurrent[1 - session.bank].view(uint64_t(session.slot) * recurrentStateBytes,
                                                                             recurrentStateBytes), recurrentStateBytes);
                copies.copy(model.states[i].candidateConv.view(uint64_t(source) * convStateBytes, convStateBytes),
                            model.states[i].conv[1 - session.bank].view(uint64_t(session.slot) * convStateBytes,
                                                                       convStateBytes), convStateBytes);
            }
            copies.copy(model.workspace.targetHidden.view(uint64_t(batch.starts[row] + query.accepted) * 8192, 8192),
                        mtpSeed(model, session, 1 - session.bank), 8192);
        }
        copies.commit();
    }
    for (uint32_t row = 0; row < batch.size; ++row) {
        Query& query = batch.queries[row]; Session& session = *query.session; uint32_t oldValid = session.kvValid;
        if (query.draft) {
            session.request.resize(oldValid + 1 + query.accepted);
            if (!query.terminal) session.request.push_back(session.ready.back());
            session.kvValid += 1 + query.accepted; session.phase = SequencePhase::tg;
            session.draftedTokens += batch.drafts; session.acceptedTokens += query.accepted;
        } else session.kvValid += query.count;
        session.bank ^= 1; model.publishPrefix(session, oldValid);
        if (query.draft) output[query.output] = emit(session), ready[query.output] = 1;
        else if (query.sample) {
            int32_t token = sampled[query.logit]; session.request.push_back(token); session.lastReturned = token;
            session.phase = SequencePhase::tg; output[query.output] = token; ready[query.output] = 1;
        }
    }
}
}  // namespace

// One scheduler step may mix prompt chunks, ordinary decode anchors, and speculative verification queries.
void step(Model& model, Session* const* sessions, const int32_t* ids, const uint32_t* starts, uint32_t count,
          float temperature, float topP, int32_t topK, int32_t* output, uint8_t* ready) {
    if (starts[0]) throw std::runtime_error("query_start_loc must begin at zero");
    if (temperature > 0 && (topK < 0 || topK > 64)) throw std::runtime_error("GPU top_k must be between 1 and 64");
    if (temperature > 0 && !topK && topP < 1)
        throw std::runtime_error("top_p below 1 requires top_k on this specialized runtime");
    std::lock_guard lock(model.control); std::array<bool, maxBatchSequences> seen{}; std::fill_n(ready, count, 0);
    Batch batch;
    for (uint32_t row = 0; row < count; ++row) {
        Session& session = *sessions[row]; uint32_t begin = starts[row], end = starts[row + 1];
        if (&session.model != &model || session.phase == SequencePhase::eos)
            throw std::runtime_error("batch requires live sequences from one model");
        if (seen[session.slot]) throw std::runtime_error("batch contains a sequence more than once");
        if (begin >= end) throw std::runtime_error("every query must contain at least one token");
        seen[session.slot] = true;
        if (!session.ready.empty()) {
            if (end - begin != 1 || ids[begin] != session.lastReturned)
                throw std::runtime_error("queued speculative output is out of order");
            output[row] = emit(session); ready[row] = 1; continue;
        }
        attach(session, ids + begin, end - begin); session.request.reserve(session.request.size() + maxDraftTokens + 1);
        if (session.request.size() > model.maxContext) throw std::runtime_error("forward exceeds maximum context");
        Query& query = batch.queries[batch.size++]; query.session = &session; query.output = row;
    }
    if (!batch.size) return; prepare(model, batch);
    std::array<uint64_t, maxBatchSequences> rngBefore{};
    auto* rng = static_cast<uint64_t*>(model.rng.buffer->metalBuffer->contents());
    for (uint32_t row = 0; row < batch.size; ++row) if (batch.queries[row].sample)
        rngBefore[row] = rng[batch.queries[row].session->slot];
    try {
        if (batch.drafts) draft(model, batch);
        forward(model, batch, temperature, topP, topK);
        commit(model, batch, temperature, rngBefore, output, ready);
    } catch (...) {
        for (uint32_t row = 0; row < batch.size; ++row) {
            Query& query = batch.queries[row]; Session& session = *query.session;
            if (temperature > 0 && query.sample) rng[session.slot] = rngBefore[row];
            if (query.draft) { session.request.resize(session.kvValid + 1); session.ready.clear(); }
        }
        throw;
    }
}
}  // namespace infeng::qwen35
