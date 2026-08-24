#include <metal_stdlib>
using namespace metal;

constant uint Q_HEADS = 16;
constant uint KV_HEADS = 4;
constant uint Q_HEADS_PER_KV_HEAD = Q_HEADS / KV_HEADS;
constant uint HEAD_DIM = 256;
constant uint SIMDGROUPS_PER_THREADGROUP = 4;

static inline ulong kv_offset(uint slot, uint slot_stride, uint token, uint head, uint dim = 0) {
    return ((ulong(slot) * slot_stride + token) * KV_HEADS + head) * HEAD_DIM + dim;
}

// Verified target rows materialize persistent MTP K/V directly from the packed layout.
[[max_total_threads_per_threadgroup(256)]]
kernel void mtp_store_kv(
        device half* cache_k [[buffer(0)]], device half* cache_v [[buffer(1)]],
        device const half* raw_k [[buffer(2)]], device const half* v [[buffer(3)]],
        device const uint* slots [[buffer(4)]], device const uint* query_positions [[buffer(5)]],
        device const uint* query_start_loc [[buffer(6)]], device const float* norm_weight [[buffer(7)]],
        device const half2* rope [[buffer(8)]], constant uint& batch_size [[buffer(9)]],
        constant uint& rows [[buffer(10)]], constant uint& slot_stride [[buffer(11)]],
        uint3 lane3 [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_index [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    uint lane = lane3.x, row = group.x / KV_HEADS, head = group.x % KV_HEADS;
    if (row >= rows) return;
    uint batch = 0; while (batch + 1 < batch_size && row >= query_start_loc[batch + 1]) ++batch;
    uint source = (row * KV_HEADS + head) * HEAD_DIM;
    uint position = query_positions[batch] + row - query_start_loc[batch];
    float value = float(raw_k[source + lane]), sum = simd_sum(value * value);
    threadgroup float partial[8];
    if (simd_lane == 0) partial[simd_index] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f; for (uint i = 0; i < 8; ++i) total += partial[i];
    float inv_rms = rsqrt(total / float(HEAD_DIM) + 1.0e-6f);
    float key = float(half(value * inv_rms * norm_weight[lane]));
    if (lane < 64) {
        uint other = lane < 32 ? lane + 32 : lane - 32, pair = lane & 31;
        float paired = float(half(float(raw_k[source + other]) * inv_rms * norm_weight[other]));
        half2 cs = rope[position * 32 + pair];
        key = key * float(cs.x) + (lane < 32 ? -paired : paired) * float(cs.y);
    }
    ulong offset = kv_offset(slots[batch], slot_stride, position, head, lane);
    cache_k[offset] = half(key); cache_v[offset] = v[source + lane];
}

// query_start_loc maps each packed token row back to its sequence and causal position.
[[max_total_threads_per_threadgroup(128)]]
kernel void attention(
        device half* output [[buffer(0)]], device const half* q [[buffer(1)]],
        device const half* k [[buffer(2)]], device const half* v [[buffer(3)]],
        device half* cache_k [[buffer(4)]], device half* cache_v [[buffer(5)]],
        device const uint* slots [[buffer(6)]], device const uint* query_positions [[buffer(7)]],
        device const uint* query_start_loc [[buffer(8)]], constant uint& batch_size [[buffer(9)]],
        constant uint& rows [[buffer(10)]], constant uint& slot_stride [[buffer(11)]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_index [[simdgroup_index_in_threadgroup]],
        uint3 threadgroup_position [[threadgroup_position_in_grid]]) {
    uint q_head = threadgroup_position.x % Q_HEADS, row = threadgroup_position.x / Q_HEADS;
    if (row >= rows) return;
    uint batch = 0; while (batch + 1 < batch_size && row >= query_start_loc[batch + 1]) ++batch;
    uint start = query_start_loc[batch], q_position = row - start, cached_tokens = query_positions[batch];
    uint kv_head = q_head / Q_HEADS_PER_KV_HEAD, attended_tokens = cached_tokens + q_position + 1;
    uint q_offset = (row * Q_HEADS + q_head) * HEAD_DIM;
    threadgroup half q_shared[HEAD_DIM];
    threadgroup float simd_numerator[SIMDGROUPS_PER_THREADGROUP][HEAD_DIM];
    threadgroup float simd_max[SIMDGROUPS_PER_THREADGROUP], simd_norm[SIMDGROUPS_PER_THREADGROUP];
    threadgroup float attention_max, attention_norm;

    if (simd_index == 0) {
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) q_shared[dim] = q[q_offset + dim];
        if (q_head % Q_HEADS_PER_KV_HEAD == 0) {
            ulong cache_offset = kv_offset(slots[batch], slot_stride, cached_tokens + q_position, kv_head);
            uint token_offset = (row * KV_HEADS + kv_head) * HEAD_DIM;
            for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) {
                cache_k[cache_offset + dim] = k[token_offset + dim];
                cache_v[cache_offset + dim] = v[token_offset + dim];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -1.0e30f, running_norm = 0.0f, numerator[HEAD_DIM / 32] = {0.0f};
    for (uint token = simd_index; token < attended_tokens; token += SIMDGROUPS_PER_THREADGROUP) {
        bool current_chunk = token >= cached_tokens;
        ulong item_offset = current_chunk
            ? ((start + token - cached_tokens) * KV_HEADS + kv_head) * HEAD_DIM
            : kv_offset(slots[batch], slot_stride, token, kv_head);
        float score = 0.0f;
        for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32)
            score = fma(float(q_shared[dim]), float(current_chunk ? k[item_offset + dim] : cache_k[item_offset + dim]),
                        score);
        score = simd_sum(score) * (1.0f / 16.0f);
        float next_max = max(running_max, score), scale = exp(running_max - next_max);
        float weight = exp(score - next_max); running_norm = running_norm * scale + weight;
        for (uint dim = simd_lane, i = 0; dim < HEAD_DIM; dim += 32, ++i) {
            float value = float(current_chunk ? v[item_offset + dim] : cache_v[item_offset + dim]);
            numerator[i] = fma(numerator[i], scale, weight * value);
        }
        running_max = next_max;
    }
    if (simd_lane == 0) { simd_max[simd_index] = running_max; simd_norm[simd_index] = running_norm; }
    for (uint dim = simd_lane, i = 0; dim < HEAD_DIM; dim += 32, ++i)
        simd_numerator[simd_index][dim] = numerator[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0 && simd_lane == 0) {
        attention_max = simd_max[0];
        for (uint s = 1; s < SIMDGROUPS_PER_THREADGROUP; ++s) attention_max = max(attention_max, simd_max[s]);
        attention_norm = 0.0f;
        for (uint s = 0; s < SIMDGROUPS_PER_THREADGROUP; ++s)
            attention_norm += simd_norm[s] * exp(simd_max[s] - attention_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0) for (uint dim = simd_lane; dim < HEAD_DIM; dim += 32) {
        float value = 0.0f;
        for (uint s = 0; s < SIMDGROUPS_PER_THREADGROUP; ++s)
            value += simd_numerator[s][dim] * exp(simd_max[s] - attention_max);
        output[q_offset + dim] = half(value / attention_norm);
    }
}
