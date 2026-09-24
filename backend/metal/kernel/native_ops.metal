#include <metal_stdlib>
using namespace metal;
constant uint HIDDEN = 4096;
constant uint VOCAB = 248320;

template <bool Gather>
kernel void rms_norm(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], device const float* w [[buffer(2)]],
                     device const GpuQuery* queries [[buffer(3)]], uint lane [[thread_index_in_threadgroup]],
                     uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
                     uint2 pos [[threadgroup_position_in_grid]]) {
  uint row = pos.y;
  if (Gather) {
    device const GpuQuery& info = queryLogit(queries, row);
    x += (info.start + info.count - info.samples - info.logit) * HIDDEN;
  }
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

template [[host_name("rms_norm")]] kernel void rms_norm<false>(device half*, device const half*, device const float*, device const GpuQuery*, uint,
                                                               uint, uint, uint2);
template [[host_name("rms_norm_gather")]] kernel void rms_norm<true>(device half*, device const half*, device const float*, device const GpuQuery*,
                                                                     uint, uint, uint, uint2);

// Packed rows identify their owning query without padding short decode queries.
kernel void mtp_fuse(device half* y [[buffer(0)]], device const half* embedding [[buffer(1)]], device const half* hidden [[buffer(2)]],
                     device const GpuQuery* queries [[buffer(3)]], device const float* embedding_weight [[buffer(4)]],
                     device const float* hidden_weight [[buffer(5)]], constant uint& rows [[buffer(6)]], constant uint& mode [[buffer(7)]],
                     uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                     uint simd_group [[simdgroup_index_in_threadgroup]], uint2 pos [[threadgroup_position_in_grid]]) {
  uint row = pos.y;
  if (row >= rows)
    return;
  device const GpuQuery& info = queryRow(queries, row);
  uint token = row - info.start;
  device const half* seed = reinterpret_cast<device const half*>(info.previous + gdnCheckpointBytes);
  device const half* h = mode == 2 ? seed : mode == 1 && !token ? seed : hidden + (mode == 1 ? row - 1 : row) * 4096;
  bool initial = !info.previous && token == 0;
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
    if (mode == 1 && (info.state != 0xffffffffu || token + 1 == info.count))
      reinterpret_cast<device half*>(info.next + (info.state != 0xffffffffu ? token * stateBytes : 0) + gdnCheckpointBytes)[d] =
          hidden[row * 4096 + d];
  }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void gdn_ba_prepare_4096x32(device half* beta [[buffer(0)]], device float* g [[buffer(1)]], device const half* x [[buffer(2)]],
                                   device const uchar* wb [[buffer(3)]], device const uchar* wa [[buffer(4)]], device const float* A [[buffer(5)]],
                                   device const float* dt [[buffer(6)]], constant bool& f32 [[buffer(7)]], ushort lane [[thread_index_in_simdgroup]],
                                   ushort simd_group [[simdgroup_index_in_threadgroup]], uint2 pos [[threadgroup_position_in_grid]]) {
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

kernel void init_rope(device half2* rope [[buffer(0)]], constant float& theta [[buffer(1)]], constant uint& pairs [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
  uint p = i / pairs, d = i % pairs;
  float angle = float(p) / pow(theta, float(d) / float(pairs));
  rope[i] = half2(half(cos(angle)), half(sin(angle)));
}

kernel void sample_logits(device int* token [[buffer(0)]], device const ulong* rng [[buffer(1)]], device ulong* sampled_rng [[buffer(2)]],
                          device const half* logits [[buffer(3)]], device GpuQuery* queries [[buffer(4)]], constant uint& mode [[buffer(5)]],
                          uint lane [[thread_index_in_threadgroup]], uint2 position [[threadgroup_position_in_grid]]) {
  uint i = position.y;
  device const GpuQuery& info = queryLogit(queries, i);
  // Mode 0 samples the target; mode 1 advances one MTP token per query; mode 2 lays out parallel DFlash proposals.
  if (mode == 1 && lane == 0)
    ++queries[i].position;
  if (mode || info.temperature <= 0) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best = -INFINITY;
    uint index = 0;
    for (uint j = lane; j < VOCAB; j += 256) {
      float v = float(logits[i * VOCAB + j]);
      if (v > best) {
        best = v;
        index = j;
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
      token[mode == 2 ? (i % info.samples) * 8 + i / info.samples : i] = indices[0];
    return;
  }
  if (lane)
    return;
  float temperature = info.temperature, top_p = info.topP;
  uint top_k = info.topK;
  ulong state = rng[info.slot];
  for (uint step = 0; step <= i - info.logit; ++step) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
  }
  sampled_rng[i] = state;
  logits += ulong(i) * VOCAB;
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
        token[i] = int(j);
        return;
      }
    }
    token[i] = int(VOCAB - 1);
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
      token[i] = int(indices[j]);
      return;
    }
  }
  token[i] = int(indices[keep - 1]);
}
