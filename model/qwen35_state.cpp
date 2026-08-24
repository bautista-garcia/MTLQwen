#include "qwen35.hpp"
#include <algorithm>
#include <stdexcept>

namespace infeng::qwen35 {
namespace {
uint64_t hashBlock(uint64_t hash, const int32_t* tokens) {
    hash ^= 0x9e3779b97f4a7c15ull;
    for (uint32_t i = 0; i < blockTokens; ++i) {
        hash ^= uint32_t(tokens[i]) + 0x9e3779b9u + (hash << 6) + (hash >> 2); hash *= 0x100000001b3ull;
    }
    return hash;
}
}

Tensor mtpSeed(Model& model, const Session& session, uint32_t bank) {
    return model.mtpSeeds.view(uint64_t(bank * maxBatchSequences + session.slot) * 4096 * 2, 4096 * 2);
}

uint8_t Model::acquireSlot() {
    for (uint8_t slot = 0; slot < slots.size(); ++slot) if (!slots[slot]) { slots[slot] = true; return slot; }
    throw std::runtime_error("maximum live sequence count reached");
}

void Model::bind(Session& session, uint32_t logical, uint32_t physical) {
    kv->map(uint32_t(session.slot) * maxLogicalBlocks + logical, physical);
    session.bindings[logical] = physical; ++blocks[physical].refs; blocks[physical].touch = ++clock;
}
void Model::unbind(Session& session, uint32_t logical) {
    uint32_t physical = session.bindings[logical];
    kv->unmap(uint32_t(session.slot) * maxLogicalBlocks + logical);
    if (!blocks[physical].refs) std::terminate();
    --blocks[physical].refs; blocks[physical].touch = ++clock; session.bindings[logical] = unbound;
}
uint32_t Model::acquireBlock() {
    uint32_t physical = physicalBlocks < blocks.size() ? physicalBlocks++ : unbound;
    if (physical == unbound) {
        uint64_t oldest = UINT64_MAX;
        for (uint32_t i = 0; i < blocks.size(); ++i)
            if (!blocks[i].refs && blocks[i].touch < oldest) physical = i, oldest = blocks[i].touch;
        if (physical == unbound) std::terminate();
        auto found = prefixTable.find(blocks[physical].hash);
        if (found != prefixTable.end() && found->second == physical) prefixTable.erase(found);
    }
    blocks[physical] = {0, 0, ++clock}; return physical;
}

void Model::reserve(const Batch& batch) {
    uint32_t required = 0;
    for (uint32_t row = 0; row < batch.size; ++row) {
        const Query& query = batch.queries[row]; Session& session = *query.session;
        uint32_t first = session.kvValid / blockTokens, last = (session.kvValid + query.count - 1) / blockTokens;
        for (uint32_t logical = first; logical <= last; ++logical)
            required += session.bindings[logical] == unbound;
    }
    uint32_t free = blocks.size() - physicalBlocks;
    for (uint32_t i = 0; i < physicalBlocks; ++i) free += !blocks[i].refs;
    if (required > free) throw std::runtime_error("KV block pool has no evictable capacity");
    kv->ensure(physicalBlocks + std::min<uint32_t>(required, blocks.size() - physicalBlocks));
    for (uint32_t row = 0; row < batch.size; ++row) {
        const Query& query = batch.queries[row]; Session& session = *query.session;
        uint32_t first = session.kvValid / blockTokens, last = (session.kvValid + query.count - 1) / blockTokens;
        for (uint32_t logical = first; logical <= last; ++logical) {
            if (session.bindings[logical] == unbound) bind(session, logical, acquireBlock());
            blocks[session.bindings[logical]].touch = ++clock;
        }
    }
}

void Model::ensureCandidates(uint32_t rows) {
    if (rows <= candidateCapacity) return;
    for (uint32_t i = 0; i < layers.size(); ++i) if (!layers[i].fullAttention) {
        states[i].candidateConv = device.empty(convStateBytes * rows);
        states[i].candidateRecurrent = device.empty(recurrentStateBytes * rows);
    }
    candidateCapacity = rows;
}

void Model::lookupPrefix(Session& session) {
    uint32_t limit = (session.request.size() - 1) / blockTokens, checkpointBlock = unbound;
    uint64_t hash = cacheNamespace, checkpointHash = 0;
    for (uint32_t logical = 0; logical < limit; ++logical) {
        hash = hashBlock(hash, session.request.data() + uint64_t(logical) * blockTokens);
        auto prefix = prefixTable.find(hash); if (prefix == prefixTable.end()) break;
        if (!((logical + 1) * blockTokens % gdnCheckpointTokens) && checkpointCache.count(hash))
            checkpointBlock = logical, checkpointHash = hash;
    }
    if (checkpointBlock == unbound) return;
    HybridCheckpoint& checkpoint = checkpointCache.at(checkpointHash); checkpoint.touch = ++clock;
    CommandBuffer copies(device, 0);
    for (uint32_t i = 0; i < layers.size(); ++i) if (!layers[i].fullAttention) {
        Tensor recurrent = states[i].recurrent[0].view(uint64_t(session.slot) * recurrentStateBytes,
                                                       recurrentStateBytes);
        Tensor conv = states[i].conv[0].view(uint64_t(session.slot) * convStateBytes, convStateBytes);
        copies.copy(checkpoint.arena.view(gdnOffsets[i], recurrentStateBytes), recurrent, recurrentStateBytes);
        copies.copy(checkpoint.arena.view(gdnOffsets[i] + recurrentStateBytes, convStateBytes), conv, convStateBytes);
    }
    if (hasMtp) copies.copy(checkpoint.arena.view(gdnBytes, 4096 * 2), mtpSeed(*this, session, 0), 4096 * 2);
    copies.commit();
    hash = cacheNamespace;
    for (uint32_t logical = 0; logical <= checkpointBlock; ++logical) {
        hash = hashBlock(hash, session.request.data() + uint64_t(logical) * blockTokens);
        bind(session, logical, prefixTable.at(hash));
    }
    session.bank = 0; session.kvValid = (checkpointBlock + 1) * blockTokens;
}

void Model::publishPrefix(Session& session, uint32_t oldValid) {
    // Prefix publication is a soft cache side effect; committed sequence state never depends on it succeeding.
    try {
        uint32_t first = oldValid / blockTokens, count = session.kvValid / blockTokens; if (first >= count) return;
        uint64_t hash = first ? blocks[session.bindings[first - 1]].hash : cacheNamespace;
        for (uint32_t logical = first; logical < count; ++logical) {
            hash = hashBlock(hash, session.request.data() + uint64_t(logical) * blockTokens);
            uint32_t physical = session.bindings[logical]; blocks[physical].hash = hash;
            prefixTable.emplace(hash, physical); blocks[physical].touch = ++clock;
        }
        if (!session.kvValid || session.kvValid % gdnCheckpointTokens) return;
        hash = blocks[session.bindings[session.kvValid / blockTokens - 1]].hash;
        if (checkpointCache.count(hash)) { checkpointCache.at(hash).touch = ++clock; return; }
        if (checkpointCache.size() == maxBatchSequences) {
            auto victim = std::min_element(checkpointCache.begin(), checkpointCache.end(),
                                           [](const auto& a, const auto& b) { return a.second.touch < b.second.touch; });
            checkpointCache.erase(victim);
        }
        HybridCheckpoint checkpoint{device.empty(gdnBytes + (hasMtp ? 4096 * 2 : 0)), ++clock};
        CommandBuffer copies(device, 0);
        for (uint32_t i = 0; i < layers.size(); ++i) if (!layers[i].fullAttention) {
            Tensor recurrent = states[i].recurrent[session.bank].view(uint64_t(session.slot) * recurrentStateBytes,
                                                                      recurrentStateBytes);
            Tensor conv = states[i].conv[session.bank].view(uint64_t(session.slot) * convStateBytes, convStateBytes);
            copies.copy(recurrent, checkpoint.arena.view(gdnOffsets[i], recurrentStateBytes), recurrentStateBytes);
            copies.copy(conv, checkpoint.arena.view(gdnOffsets[i] + recurrentStateBytes, convStateBytes), convStateBytes);
        }
        if (hasMtp) copies.copy(mtpSeed(*this, session, session.bank), checkpoint.arena.view(gdnBytes, 4096 * 2),
                                4096 * 2);
        copies.commit(); checkpointCache.emplace(hash, std::move(checkpoint));
    } catch (...) {}
}

void Model::release(Session& session) {
    std::lock_guard lock(control); if (session.phase == SequencePhase::eos) return;
    for (uint32_t logical = 0; logical < session.bindings.size(); ++logical)
        if (session.bindings[logical] != unbound) unbind(session, logical);
    slots[session.slot] = false; session.phase = SequencePhase::eos;
}

Session::Session(Model& owner) : model(owner), sequenceId(0), bindings(owner.maxLogicalBlocks, unbound) {
    std::lock_guard lock(model.control); sequenceId = model.nextSequenceId++; slot = model.acquireSlot();
}
Session::~Session() { model.release(*this); }
}  // namespace infeng::qwen35
