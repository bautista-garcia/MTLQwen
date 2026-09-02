#include <metal_stdlib>
using namespace metal;

constant uint DFLASH_Q_HEADS = 32;
constant uint DFLASH_KV_HEADS = 8;
constant uint DFLASH_HEAD_DIM = 128;
constant uint DFLASH_WINDOW = 4096;
constant uint DFLASH_SIMDGROUPS = 4;

static inline ulong dflash_kv_offset(uint slot, uint slot_stride, uint token, uint head, uint dim = 0) {
  return ((ulong(slot) * slot_stride + token) * DFLASH_KV_HEADS + head) * DFLASH_HEAD_DIM + dim;
}

kernel void capture_hidden(device half* features [[buffer(0)]], device const half* hidden [[buffer(1)]], constant uint& feature [[buffer(2)]],
                           constant uint& rows [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= rows * 4096)
    return;
  uint row = i / 4096, dim = i % 4096;
  features[(row * 8 + feature) * 4096 + dim] = hidden[i];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void dflash_attention_prepare(device half* q [[buffer(0)]], device half* k [[buffer(1)]], device const half* raw_q [[buffer(2)]],
                                     device const half* raw_k [[buffer(3)]], device const float* q_norm [[buffer(4)]],
                                     device const float* k_norm [[buffer(5)]], device const half2* rope [[buffer(6)]],
                                     device const uint* positions [[buffer(7)]], device const uint* query_start_loc [[buffer(8)]],
                                     constant uint& batch_size [[buffer(9)]], uint3 lane3 [[thread_position_in_threadgroup]],
                                     uint simd_lane [[thread_index_in_simdgroup]], uint simd_index [[simdgroup_index_in_threadgroup]],
                                     uint3 group [[threadgroup_position_in_grid]]) {
  uint lane = lane3.x;
  uint head_row = group.y, row = head_row / 40, head = head_row % 40;
  bool query = head < DFLASH_Q_HEADS;
  uint local_head = query ? head : head - DFLASH_Q_HEADS;
  uint source = (row * (query ? DFLASH_Q_HEADS : DFLASH_KV_HEADS) + local_head) * DFLASH_HEAD_DIM;
  device const half* raw = query ? raw_q : raw_k;
  device const float* weight = query ? q_norm : k_norm;
  float value = float(raw[source + lane]), sum = simd_sum(value * value);
  threadgroup float partial[DFLASH_SIMDGROUPS], scale;
  if (!simd_lane)
    partial[simd_index] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!lane) {
    float total = 0.0f;
    for (uint i = 0; i < DFLASH_SIMDGROUPS; ++i)
      total += partial[i];
    scale = rsqrt(total / float(DFLASH_HEAD_DIM) + 1.0e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  half normalized = half(value * scale * weight[lane]);
  uint batch = 0;
  while (batch + 1 < batch_size && row >= query_start_loc[batch + 1])
    ++batch;
  uint position = positions[batch] + row - query_start_loc[batch], pair = lane & 63;
  uint other = lane < 64 ? lane + 64 : lane - 64;
  half2 cs = rope[position * 64 + pair];
  half paired = half(float(raw[source + other]) * scale * weight[other]);
  (query ? q : k)[source + lane] = half(float(normalized) * float(cs.x) + (lane < 64 ? -float(paired) : float(paired)) * float(cs.y));
}

[[max_total_threads_per_threadgroup(128)]]
kernel void dflash_store_kv(device half* cache_k [[buffer(0)]], device half* cache_v [[buffer(1)]], device const half* raw_k [[buffer(2)]],
                            device const half* v [[buffer(3)]], device const uint* slots [[buffer(4)]], device const uint* positions [[buffer(5)]],
                            device const uint* query_start_loc [[buffer(6)]], device const float* norm [[buffer(7)]],
                            device const half2* rope [[buffer(8)]], constant uint& batch_size [[buffer(9)]], constant uint& rows [[buffer(10)]],
                            constant uint& slot_stride [[buffer(11)]], uint3 lane3 [[thread_position_in_threadgroup]],
                            uint simd_lane [[thread_index_in_simdgroup]], uint simd_index [[simdgroup_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {
  uint lane = lane3.x;
  uint row = group.x / DFLASH_KV_HEADS, head = group.x % DFLASH_KV_HEADS;
  if (row >= rows)
    return;
  uint source = (row * DFLASH_KV_HEADS + head) * DFLASH_HEAD_DIM;
  float value = float(raw_k[source + lane]), sum = simd_sum(value * value);
  threadgroup float partial[DFLASH_SIMDGROUPS], scale;
  if (!simd_lane)
    partial[simd_index] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!lane) {
    float total = 0.0f;
    for (uint i = 0; i < DFLASH_SIMDGROUPS; ++i)
      total += partial[i];
    scale = rsqrt(total / float(DFLASH_HEAD_DIM) + 1.0e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint batch = 0;
  while (batch + 1 < batch_size && row >= query_start_loc[batch + 1])
    ++batch;
  uint position = positions[batch] + row - query_start_loc[batch], pair = lane & 63;
  uint other = lane < 64 ? lane + 64 : lane - 64;
  half2 cs = rope[position * 64 + pair];
  float key = float(half(value * scale * norm[lane]));
  float paired = float(half(float(raw_k[source + other]) * scale * norm[other]));
  key = key * float(cs.x) + (lane < 64 ? -paired : paired) * float(cs.y);
  ulong destination = dflash_kv_offset(slots[batch], slot_stride, position, head, lane);
  cache_k[destination] = half(key);
  cache_v[destination] = v[source + lane];
}

[[max_total_threads_per_threadgroup(128)]]
kernel void dflash_attention_scan(device float* partials [[buffer(0)]], device const half* q [[buffer(1)]], device const half* k [[buffer(2)]],
                                  device const half* v [[buffer(3)]], device const half* cache_k [[buffer(4)]],
                                  device const half* cache_v [[buffer(5)]], device const uint* slots [[buffer(6)]],
                                  device const uint* positions [[buffer(7)]], device const uint* query_start_loc [[buffer(8)]],
                                  constant uint& batch_size [[buffer(9)]], constant uint& rows [[buffer(10)]],
                                  constant uint& slot_stride [[buffer(11)]], constant uint& splits [[buffer(12)]],
                                  constant uint& sliding [[buffer(13)]], uint simd_lane [[thread_index_in_simdgroup]],
                                  uint simd_index [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  uint item = group.x, split = item % splits, head_row = item / splits;
  uint q_head = head_row % DFLASH_Q_HEADS, row = head_row / DFLASH_Q_HEADS;
  if (row >= rows)
    return;
  uint batch = 0;
  while (batch + 1 < batch_size && row >= query_start_loc[batch + 1])
    ++batch;
  uint start = query_start_loc[batch], block = query_start_loc[batch + 1] - start, q_position = row - start;
  uint cached = positions[batch], first = sliding && cached + q_position + 1 > DFLASH_WINDOW ? cached + q_position + 1 - DFLASH_WINDOW : 0;
  uint context = cached - first, temporary = sliding ? q_position + 1 : block, attended = context + temporary;
  uint split_begin = attended * split / splits, split_end = attended * (split + 1) / splits;
  uint kv_head = q_head / 4, q_offset = (row * DFLASH_Q_HEADS + q_head) * DFLASH_HEAD_DIM;
  threadgroup half q_shared[DFLASH_HEAD_DIM];
  threadgroup float numerators[DFLASH_SIMDGROUPS][DFLASH_HEAD_DIM];
  threadgroup float maxima[DFLASH_SIMDGROUPS], norms[DFLASH_SIMDGROUPS], attention_max, attention_norm;
  if (!simd_index)
    for (uint dim = simd_lane; dim < DFLASH_HEAD_DIM; dim += 32)
      q_shared[dim] = q[q_offset + dim];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float running_max = -1.0e30f, running_norm = 0.0f, numerator[DFLASH_HEAD_DIM / 32] = {0.0f};
  for (uint token = split_begin + simd_index; token < split_end; token += DFLASH_SIMDGROUPS) {
    bool current = token >= context;
    uint position = current ? token - context : first + token;
    ulong offset = current ? ulong(start + position) * DFLASH_KV_HEADS * DFLASH_HEAD_DIM + kv_head * DFLASH_HEAD_DIM
                           : dflash_kv_offset(slots[batch], slot_stride, position, kv_head);
    float score = 0.0f;
    for (uint dim = simd_lane; dim < DFLASH_HEAD_DIM; dim += 32)
      score = fma(float(q_shared[dim]), float((current ? k : cache_k)[offset + dim]), score);
    score = simd_sum(score) * 0.08838834764831845f;
    float next_max = max(running_max, score), scale = exp(running_max - next_max), weight = exp(score - next_max);
    running_norm = running_norm * scale + weight;
    for (uint dim = simd_lane, i = 0; dim < DFLASH_HEAD_DIM; dim += 32, ++i)
      numerator[i] = fma(numerator[i], scale, weight * float((current ? v : cache_v)[offset + dim]));
    running_max = next_max;
  }
  if (!simd_lane) {
    maxima[simd_index] = running_max;
    norms[simd_index] = running_norm;
  }
  for (uint dim = simd_lane, i = 0; dim < DFLASH_HEAD_DIM; dim += 32, ++i)
    numerators[simd_index][dim] = numerator[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!simd_index && !simd_lane) {
    attention_max = maxima[0];
    for (uint i = 1; i < DFLASH_SIMDGROUPS; ++i)
      attention_max = max(attention_max, maxima[i]);
    attention_norm = 0.0f;
    for (uint i = 0; i < DFLASH_SIMDGROUPS; ++i)
      attention_norm += norms[i] * exp(maxima[i] - attention_max);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong output = (ulong(head_row) * splits + split) * (DFLASH_HEAD_DIM + 2);
  if (!simd_index)
    for (uint dim = simd_lane; dim < DFLASH_HEAD_DIM; dim += 32) {
      float value = 0.0f;
      for (uint i = 0; i < DFLASH_SIMDGROUPS; ++i)
        value += numerators[i][dim] * exp(maxima[i] - attention_max);
      partials[output + dim] = value;
    }
  if (!simd_index && !simd_lane) {
    partials[output + DFLASH_HEAD_DIM] = attention_max;
    partials[output + DFLASH_HEAD_DIM + 1] = attention_norm;
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void dflash_attention_reduce(device half* output [[buffer(0)]], device const float* partials [[buffer(1)]],
                                    device const half* qg [[buffer(2)]], constant uint& rows [[buffer(3)]], constant uint& splits [[buffer(4)]],
                                    uint simd_lane [[thread_index_in_simdgroup]], uint simd_index [[simdgroup_index_in_threadgroup]],
                                    uint3 group [[threadgroup_position_in_grid]]) {
  uint head_row = group.x, row = head_row / DFLASH_Q_HEADS, head = head_row % DFLASH_Q_HEADS;
  if (row >= rows)
    return;
  threadgroup float split_max[8], maximum, norm;
  if (!simd_index && simd_lane < splits)
    split_max[simd_lane] = partials[(ulong(head_row) * splits + simd_lane) * (DFLASH_HEAD_DIM + 2) + DFLASH_HEAD_DIM];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!simd_index && !simd_lane) {
    maximum = split_max[0];
    for (uint i = 1; i < splits; ++i)
      maximum = max(maximum, split_max[i]);
    norm = 0.0f;
    for (uint i = 0; i < splits; ++i) {
      ulong offset = (ulong(head_row) * splits + i) * (DFLASH_HEAD_DIM + 2);
      norm += partials[offset + DFLASH_HEAD_DIM + 1] * exp(split_max[i] - maximum);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint dim = simd_index * 32 + simd_lane; dim < DFLASH_HEAD_DIM; dim += 128) {
    float numerator = 0.0f;
    for (uint i = 0; i < splits; ++i) {
      ulong offset = (ulong(head_row) * splits + i) * (DFLASH_HEAD_DIM + 2);
      numerator += partials[offset + dim] * exp(split_max[i] - maximum);
    }
    output[(row * DFLASH_Q_HEADS + head) * DFLASH_HEAD_DIM + dim] = half(numerator / norm);
  }
}
