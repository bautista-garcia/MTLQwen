#include <metal_stdlib>
using namespace metal;
constant uint MAX_SEQUENCES = 8;
constant uint HIDDEN = 4096;
constant uint VOCAB = 248320;

kernel void rmsnorm(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], device const float* w [[buffer(2)]],
                    uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                    uint simd_group [[simdgroup_index_in_threadgroup]], uint2 pos [[threadgroup_position_in_grid]]) {
  uint row = pos.y;
  threadgroup float sums[9];
  float sum = 0.0f;
  for (uint d = lane; d < HIDDEN; d += 256) {
    float v = float(x[row * HIDDEN + d]);
    sum += v * v;
  }
  sum = simd_sum(sum);
  if (simd_lane == 0)
    sums[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    sum = simd_lane < 8 ? sums[simd_lane] : 0.0f;
    sum = simd_sum(sum);
    if (simd_lane == 0)
      sums[8] = rsqrt(sum / float(HIDDEN) + 1.0e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = lane; d < HIDDEN; d += 256)
    y[row * HIDDEN + d] = half(float(x[row * HIDDEN + d]) * sums[8] * w[d]);
}

// Packed rows use query_start_loc to recover the owning sequence without padding short decode queries.
kernel void mtp_fuse(device half* y [[buffer(0)]], device const half* embedding [[buffer(1)]], device const half* hidden [[buffer(2)]],
                     device const half* seeds [[buffer(3)]], device const uint* slots [[buffer(4)]], device const uint* banks [[buffer(5)]],
                     device const uint* kv_valid [[buffer(6)]], device const uint* query_start_loc [[buffer(7)]],
                     device const float* embedding_weight [[buffer(8)]], device const float* hidden_weight [[buffer(9)]],
                     constant uint& batch [[buffer(10)]], constant uint& rows [[buffer(11)]], constant uint& mode [[buffer(12)]],
                     uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                     uint simd_group [[simdgroup_index_in_threadgroup]], uint2 pos [[threadgroup_position_in_grid]]) {
  uint row = pos.y;
  if (row >= rows)
    return;
  uint b = 0;
  while (b + 1 < batch && row >= query_start_loc[b + 1])
    ++b;
  uint token = row - query_start_loc[b];
  device const half* seed = seeds + (banks[b] * MAX_SEQUENCES + slots[b]) * 4096;
  device const half* h = mode == 2 ? seed : mode == 1 && !token ? seed : hidden + (mode == 1 ? row - 1 : row) * 4096;
  bool initial = !kv_valid[b] && token == 0;
  threadgroup float sums[18];
  float es = 0.0f, hs = 0.0f;
  for (uint d = lane; d < 4096; d += 256) {
    float ev = float(embedding[row * 4096 + d]), hv = initial ? 0.0f : float(h[d]);
    es += ev * ev;
    hs += hv * hv;
  }
  es = simd_sum(es);
  hs = simd_sum(hs);
  if (simd_lane == 0) {
    sums[simd_group] = es;
    sums[8 + simd_group] = hs;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    es = simd_sum(simd_lane < 8 ? sums[simd_lane] : 0.0f);
    hs = simd_sum(simd_lane < 8 ? sums[8 + simd_lane] : 0.0f);
    if (simd_lane == 0) {
      sums[16] = rsqrt(es / 4096.0f + 1.0e-6f);
      sums[17] = rsqrt(hs / 4096.0f + 1.0e-6f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = lane; d < 4096; d += 256) {
    float hv = initial ? 0.0f : float(h[d]);
    y[row * 8192 + d] = half(float(embedding[row * 4096 + d]) * sums[16] * embedding_weight[d]);
    y[row * 8192 + 4096 + d] = half(hv * sums[17] * hidden_weight[d]);
  }
}

kernel void gather_rows(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], device const uint* rows [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
  y[i] = x[rows[i / HIDDEN] * HIDDEN + i % HIDDEN];
}

kernel void add_half(device half* y [[buffer(0)]], device const half* a [[buffer(1)]], device const half* b [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  y[i] = a[i] + b[i];
}

kernel void silu_mul(device half* y [[buffer(0)]], device const half* gate [[buffer(1)]], device const half* up [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float g = float(gate[i]);
  y[i] = half((g / (1.0f + exp(-g))) * float(up[i]));
}

[[max_total_threads_per_threadgroup(64)]]
kernel void gdn_ba_prepare_4096x32(device half* beta [[buffer(0)]], device float* g [[buffer(1)]], device const half* x [[buffer(2)]],
                                   device const uchar* wb [[buffer(3)]], device const uchar* wa [[buffer(4)]], device const float* A [[buffer(5)]],
                                   device const float* dt [[buffer(6)]], constant bool& f32 [[buffer(7)]],
                                   ushort lane [[thread_index_in_simdgroup]], ushort simd_group [[simdgroup_index_in_threadgroup]],
                                   uint2 pos [[threadgroup_position_in_grid]]) {
  uint row = pos.y, out = pos.x;
  device const uchar* w = simd_group ? wa : wb;
  float sum = 0.0f;
  for (uint k = lane; k < 4096; k += 32) {
    uint i = out * 4096 + k;
    float weight = f32 ? reinterpret_cast<device const float*>(w)[i] : float(reinterpret_cast<device const half*>(w)[i]);
    sum += float(x[row * 4096 + k]) * weight;
  }
  threadgroup float ba[2];
  sum = simd_sum(sum);
  if (lane == 0)
    ba[simd_group] = float(half(sum));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    uint i = row * 32 + out;
    float av = ba[1] + dt[out];
    beta[i] = half(1.0f / (1.0f + exp(-ba[0])));
    g[i] = A[out] * log(1.0f + exp(av));
  }
}

kernel void split_repeat_qk(device half* q [[buffer(0)]], device half* k [[buffer(1)]], device half* v [[buffer(2)]],
                            device const half* mixed [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  uint row = i / 4096, d = i % 4096, source = (d % 2048);
  q[i] = mixed[row * 8192 + source];
  k[i] = mixed[row * 8192 + 2048 + source];
  v[i] = mixed[row * 8192 + 4096 + d];
}

kernel void init_rope(device half2* rope [[buffer(0)]], constant float& theta [[buffer(1)]], constant uint& pairs [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
  uint p = i / pairs, d = i % pairs;
  float angle = float(p) / pow(theta, float(d) / float(pairs));
  rope[i] = half2(half(cos(angle)), half(sin(angle)));
}

kernel void pad_rows(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], constant uint& rows [[buffer(2)]],
                     constant uint& padded [[buffer(3)]], constant uint& dim [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i < padded * dim)
    y[i] = i < rows * dim ? x[i] : half(0.0);
}

kernel void argmax_logits(device uint* token [[buffer(0)]], device const half* logits [[buffer(1)]], constant uint& group [[buffer(2)]],
                          constant uint& stride [[buffer(3)]], uint lane [[thread_index_in_threadgroup]],
                          uint2 position [[threadgroup_position_in_grid]]) {
  uint row = position.y;
  threadgroup float values[256];
  threadgroup uint indices[256];
  float best = -INFINITY;
  uint index = 0;
  for (uint i = lane; i < VOCAB; i += 256) {
    float v = float(logits[row * VOCAB + i]);
    if (v > best) {
      best = v;
      index = i;
    }
  }
  values[lane] = best;
  indices[lane] = index;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 128; s; s >>= 1) {
    if (lane < s && values[lane + s] > values[lane]) {
      values[lane] = values[lane + s];
      indices[lane] = indices[lane + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    token[(row % group) * stride + row / group] = indices[0];
}

kernel void sample_logits(device int* token [[buffer(0)]], device ulong* rng [[buffer(1)]], device const half* logits [[buffer(2)]],
                          constant float& temperature [[buffer(3)]], constant float& top_p [[buffer(4)]], constant uint& top_k [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
  if (i)
    return;
  ulong state = rng[0];
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  rng[0] = state;
  float random = float(state >> 40) * (1.0f / 16777216.0f);
  if (!top_k) {
    float maximum = -INFINITY;
    for (uint j = 0; j < VOCAB; ++j)
      maximum = max(maximum, float(logits[j]) / temperature);
    float total = 0.0f;
    for (uint j = 0; j < VOCAB; ++j)
      total += exp(float(logits[j]) / temperature - maximum);
    float target = random * total, cumulative = 0.0f;
    for (uint j = 0; j < VOCAB; ++j) {
      cumulative += exp(float(logits[j]) / temperature - maximum);
      if (cumulative >= target) {
        token[0] = int(j);
        return;
      }
    }
    token[0] = int(VOCAB - 1);
    return;
  }
  float values[64];
  uint indices[64], count = min(top_k, 64u);
  for (uint j = 0; j < count; ++j) {
    values[j] = -INFINITY;
    indices[j] = 0;
  }
  for (uint j = 0; j < VOCAB; ++j) {
    float value = float(logits[j]) / temperature;
    if (value <= values[count - 1])
      continue;
    uint p = count - 1;
    while (p && value > values[p - 1]) {
      values[p] = values[p - 1];
      indices[p] = indices[p - 1];
      --p;
    }
    values[p] = value;
    indices[p] = j;
  }
  float total = 0.0f;
  for (uint j = 0; j < count; ++j)
    total += exp(values[j] - values[0]);
  float limit = top_p * total, nucleus = 0.0f;
  uint keep = 0;
  do {
    nucleus += exp(values[keep] - values[0]);
    ++keep;
  } while (keep < count && nucleus < limit);
  float target = random * nucleus, cumulative = 0.0f;
  for (uint j = 0; j < keep; ++j) {
    cumulative += exp(values[j] - values[0]);
    if (cumulative >= target) {
      token[0] = int(indices[j]);
      return;
    }
  }
  token[0] = int(indices[keep - 1]);
}
