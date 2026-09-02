#include <metal_stdlib>
#include <metal_matrix>
using namespace metal;

constant int D = 128;
constant int DECODE_PARTS = 4;
constant int C_PREFILL = 32;
constant int C_PREFILL_CHUNK = 16;
constant int D_PREFILL_TILE = 32;

static inline long qkv_offset(long b, long t, long h, long d, long seq_len, long num_heads) {
  return ((b * seq_len + t) * num_heads + h) * D + d;
}

static inline long state_offset(long b, long h, long k, long v, long num_heads) {
  return ((b * num_heads + h) * D + k) * D + v;
}

static inline long value_offset(long b, long t, long h, long d, long s0, long s1, long s2, long s3) {
  return b * s0 + t * s1 + h * s2 + d * s3;
}

static inline void mma32x32(threadgroup half* dst, threadgroup half* a_src, threadgroup half* b_src, threadgroup float* scratch, uint simd_lane,
                            uint simd_group, bool add, bool lower_only, bool fp32_accum) {
  for (uint block = simd_group; block < (lower_only ? 10 : 16); block += 4) {
    uint br = lower_only ? (block < 1 ? 0 : (block < 3 ? 1 : (block < 6 ? 2 : 3))) : block >> 2;
    uint rb = br << 3, cb = (lower_only ? block - (br * (br + 1) >> 1) : block & 3) << 3, base = simd_group << 6;
    simdgroup_matrix<half, 8, 8> a, b;
    if (fp32_accum) {
      simdgroup_matrix<float, 8, 8> c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (uint ko = 0; ko < 32; ko += 8) {
        simdgroup_load(a, a_src, 32, ulong2(ko, rb));
        simdgroup_load(b, b_src, 32, ulong2(cb, ko));
        simdgroup_multiply_accumulate(c, a, b, c);
      }
      simdgroup_store(c, scratch + base, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = simd_lane; e < 64; e += 32) {
        uint r = e >> 3, col = e & 7, di = (rb + r) * 32 + cb + col;
        dst[di] = half((add ? float(dst[di]) : 0.0f) + scratch[base + e]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      simdgroup_matrix<half, 8, 8> c;
      if (add)
        simdgroup_load(c, dst, 32, ulong2(cb, rb));
      else
        c = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
      for (uint ko = 0; ko < 32; ko += 8) {
        simdgroup_load(a, a_src, 32, ulong2(ko, rb));
        simdgroup_load(b, b_src, 32, ulong2(cb, ko));
        simdgroup_multiply_accumulate(c, a, b, c);
      }
      simdgroup_store(c, dst, 32, ulong2(cb, rb));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void invert32_unipotent(threadgroup half* p, threadgroup half* p_next, threadgroup half* m, threadgroup half* m_next,
                                      threadgroup float* scratch, uint lane, uint simd_lane, uint simd_group) {
  for (uint stage = 0; stage < 5; ++stage) {
    for (uint idx = lane; idx < 32 * 32; idx += 128)
      p_next[idx] = p[idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma32x32(p_next, m, p, scratch, simd_lane, simd_group, true, true, false);
    mma32x32(m_next, m, m, scratch, simd_lane, simd_group, false, true, false);
    for (uint idx = lane; idx < 32 * 32; idx += 128) {
      uint r = idx >> 5, c = idx & 31;
      p[idx] = p_next[idx];
      m[idx] = c <= r ? m_next[idx] : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

static inline void run_delta_rule_token(device half* output, device float* state, device const float* previous_state, device const half* query,
                                        device const half* key, device const half* value, device const float* g, device const half* beta, long b,
                                        long state_row, long previous_row, long t, long h, long seq_len, long num_heads, long vs0, long vs1, long vs2,
                                        long vs3, uint lane, uint simd_lane, uint simd_group, threadgroup half* q, threadgroup half* k,
                                        threadgroup float* scratch) {
  float qv = 0.0f, kv = 0.0f;
  if (simd_group < D / 32) {
    qv = float(query[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
    kv = float(key[qkv_offset(b, t, h, lane, seq_len, num_heads)]);
    float q_partial = simd_sum(qv * qv), k_partial = simd_sum(kv * kv), qk_partial = simd_sum(qv * kv);
    if (simd_lane == 0) {
      scratch[simd_group] = q_partial;
      scratch[D / 32 + simd_group] = k_partial;
      scratch[D / 16 + simd_group] = qk_partial;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float q_total = simd_sum(simd_lane < D / 32 ? scratch[simd_lane] : 0.0f);
    float k_total = simd_sum(simd_lane < D / 32 ? scratch[D / 32 + simd_lane] : 0.0f);
    float qk_total = simd_sum(simd_lane < D / 32 ? scratch[D / 16 + simd_lane] : 0.0f);
    if (simd_lane == 0) {
      scratch[0] = q_total;
      scratch[1] = k_total;
      scratch[1024] = qk_total;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group < D / 32) {
    float q_norm = rsqrt(scratch[0] + 1.0e-6f) * 0.08838834764831845f;
    float k_norm = rsqrt(scratch[1] + 1.0e-6f);
    if (lane == 0)
      scratch[1024] *= q_norm * k_norm;
    q[lane] = half(qv * q_norm);
    k[lane] = half(kv * k_norm);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Decay + Prediction (v = kt @ St-1 * decay)
  uint vv = lane & 127, part = lane >> 7;
  long base_offset = state_offset(state_row, h, 0, vv, num_heads);
  long previous_offset = state_offset(previous_row, h, 0, vv, num_heads);
  long row_stride = state_offset(state_row, h, 1, vv, num_heads) - base_offset;
  long previous_stride = state_offset(previous_row, h, 1, vv, num_heads) - previous_offset;
  long loop_stride = DECODE_PARTS * row_stride;
  long previous_loop_stride = DECODE_PARTS * previous_stride;

  long off = base_offset + (part * row_stride);
  long previous_off = previous_offset + (part * previous_stride);
  float prediction = 0.0f, q_state = 0.0f;
  float decay = exp(g[(b * seq_len + t) * num_heads + h]);

  for (uint kk = part; kk < D; kk += DECODE_PARTS * 4) {
    float s0 = previous_state[previous_off] * decay, s1 = previous_state[previous_off + previous_loop_stride] * decay;
    float s2 = previous_state[previous_off + previous_loop_stride * 2] * decay;
    float s3 = previous_state[previous_off + previous_loop_stride * 3] * decay;
    state[off] = s0;
    state[off + loop_stride] = s1;
    state[off + loop_stride * 2] = s2;
    state[off + loop_stride * 3] = s3;
    prediction += s0 * float(k[kk]) + s1 * float(k[kk + DECODE_PARTS]) + s2 * float(k[kk + DECODE_PARTS * 2]) + s3 * float(k[kk + DECODE_PARTS * 3]);
    q_state += s0 * float(q[kk]) + s1 * float(q[kk + DECODE_PARTS]) + s2 * float(q[kk + DECODE_PARTS * 2]) + s3 * float(q[kk + DECODE_PARTS * 3]);
    off += loop_stride * 4;
    previous_off += previous_loop_stride * 4;
  }
  scratch[lane] = prediction;
  scratch[512 + lane] = q_state;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The first row of 128 threads reduce the partials that summed give v[lane]
  if (part == 0) {
    float pred = scratch[0 * 128 + vv] + scratch[1 * 128 + vv] + scratch[2 * 128 + vv] + scratch[3 * 128 + vv];
    float cached_delta = (float(value[value_offset(b, t, h, vv, vs0, vs1, vs2, vs3)]) - pred) * float(beta[(b * seq_len + t) * num_heads + h]);
    float q_total = scratch[512 + 0 * 128 + vv] + scratch[512 + 1 * 128 + vv] + scratch[512 + 2 * 128 + vv] + scratch[512 + 3 * 128 + vv];
    scratch[vv] = cached_delta;
    output[qkv_offset(b, t, h, vv, seq_len, num_heads)] = half(q_total + cached_delta * scratch[1024]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float cached_delta = scratch[vv];        // Cache into local register execution space
  off = base_offset + (part * row_stride); // Reset memory pointer to the top row

  for (uint kk = part; kk < D; kk += DECODE_PARTS * 4) {
    float s0 = state[off] + float(k[kk]) * cached_delta, s1 = state[off + loop_stride] + float(k[kk + DECODE_PARTS]) * cached_delta;
    float s2 = state[off + loop_stride * 2] + float(k[kk + DECODE_PARTS * 2]) * cached_delta,
          s3 = state[off + loop_stride * 3] + float(k[kk + DECODE_PARTS * 3]) * cached_delta;
    state[off] = s0;
    state[off + loop_stride] = s1;
    state[off + loop_stride * 2] = s2;
    state[off + loop_stride * 3] = s3;
    off += loop_stride * 4;
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void delta_rule_prefill(device half* output [[buffer(0)]], device float* state0 [[buffer(1)]], device float* state1 [[buffer(2)]],
                               device const uint* slots [[buffer(3)]], device const uint* banks [[buffer(4)]], device const uint* valid [[buffer(5)]],
                               device const half* query [[buffer(6)]], device const half* key [[buffer(7)]], device const half* value [[buffer(8)]],
                               device const float* g [[buffer(9)]], device const half* beta [[buffer(10)]], constant long& batch_size [[buffer(11)]],
                               constant long& seq_len [[buffer(12)]], constant long& num_heads [[buffer(13)]], constant long& vs0 [[buffer(14)]],
                               constant long& vs1 [[buffer(15)]], constant long& vs2 [[buffer(16)]], constant long& vs3 [[buffer(17)]],
                               uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
                               uint3 lane3 [[thread_position_in_threadgroup]], uint3 group3 [[threadgroup_position_in_grid]]) {
  uint lane = lane3.x;
  long h = group3.y % num_heads, b = group3.y / num_heads;
  if (b >= batch_size || h >= num_heads || group3.x || lane >= 128)
    return;
  long slot = slots[b];
  bool has_initial_state = valid[b];
  device float* state = banks[b] ? state0 : state1;
  device const float* previous_state = banks[b] ? state1 : state0;
  threadgroup half k_tile[C_PREFILL * D_PREFILL_TILE], w_tile[C_PREFILL * D_PREFILL_TILE], u_tile[C_PREFILL * D_PREFILL_TILE];
  threadgroup half k_full[C_PREFILL * D];
  threadgroup half l_tile[C_PREFILL * C_PREFILL], qk_tile[C_PREFILL * C_PREFILL], m_tile[32 * 32];
  threadgroup half p_w_tile[32 * 32];
  threadgroup float gamma[C_PREFILL], log_gamma[C_PREFILL], beta_tile[C_PREFILL], q_norm[C_PREFILL], k_norm[C_PREFILL], scratch[256];
  for (long chunk = 0; chunk < seq_len; chunk += C_PREFILL_CHUNK) {
    long C = min(long(C_PREFILL_CHUNK), seq_len - chunk);
    bool initial = has_initial_state || chunk > 0;
    device const float* current_state = chunk ? state : previous_state;

    // Phase 1: gamma_i = prod_{m<=i} alpha_m and L2 norm factors (k & q)
    if (lane < C_PREFILL) { // Only first SIMD works (4 SIMD available)
      // log(gamma_i) = sum_{m<=i} g_m; keep ratios in log-space to avoid fp32 under/overflow.
      float x = lane < C ? g[(b * seq_len + chunk + lane) * num_heads + h] : 0.0f;
      for (uint o = 1; o < 32; o <<= 1) {
        float y = simd_shuffle_up(x, o);
        if (simd_lane >= o)
          x += y;
      }
      gamma[lane] = exp(x);
      log_gamma[lane] = x;
      beta_tile[lane] = lane < C ? float(beta[(b * seq_len + chunk + lane) * num_heads + h]) : 0.0f;
    }

    { // This defines a local scope for variables like t and p
      uint t = lane >> 2, p = lane & 3;
      float qs = 0.0f, ks = 0.0f;
      // C reductions (t) with 4 partials each (p)
      for (uint d = p; d < D; d += 4) {
        float qv = t < C ? float(query[qkv_offset(b, chunk + t, h, d, seq_len, num_heads)]) : 0.0f;
        float kv = t < C ? float(key[qkv_offset(b, chunk + t, h, d, seq_len, num_heads)]) : 0.0f;
        qs += qv * qv;
        ks += kv * kv;
      }
      // Reduce + broadcast partials (shuffle_xor reads from thread m lanes apart)
      qs += simd_shuffle_xor(qs, 1);
      qs += simd_shuffle_xor(qs, 2);
      ks += simd_shuffle_xor(ks, 1);
      ks += simd_shuffle_xor(ks, 2);
      if ((simd_lane & 3) == 0) {
        q_norm[t] = rsqrt(qs + 1.0e-6f) * 0.08838834764831845f;
        k_norm[t] = rsqrt(ks + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: GEMM (K @ K^T, Q @ K^T)
    for (uint d0 = 0; d0 < D; d0 += D_PREFILL_TILE) { // Tiles of (C_PREFILL, D_PREFILL)
      // Load (C_PREFILL, D_PREFILL) tile cooperatively and L2 norm
      for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
        uint i = idx / D_PREFILL_TILE, d = d0 + idx % D_PREFILL_TILE;
        k_tile[idx] = half(i < C ? float(key[qkv_offset(b, chunk + i, h, d, seq_len, num_heads)]) * k_norm[i] : 0.0f);
        u_tile[idx] = half(i < C ? float(query[qkv_offset(b, chunk + i, h, d, seq_len, num_heads)]) * q_norm[i] : 0.0f);
        k_full[d * C_PREFILL + i] = k_tile[idx];
        // K^T
        w_tile[(idx % D_PREFILL_TILE) * C_PREFILL + i] = k_tile[idx];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma32x32(l_tile, k_tile, w_tile, scratch, simd_lane, simd_group, d0 != 0, true, false);
      mma32x32(qk_tile, u_tile, w_tile, scratch, simd_lane, simd_group, d0 != 0, true, false);
    }
    // Lw = strictLower(-diag(beta) K K^T), A_local = causalLower(Gamma o (Q K^T)).
    for (uint idx = lane; idx < C_PREFILL * C_PREFILL; idx += 128) {
      uint i = idx >> 5, j = idx & 31;
      l_tile[idx] = half(j < i && i < C && j < C ? -beta_tile[i] * float(l_tile[idx]) : 0.0f);
      qk_tile[idx] = half(j <= i && i < C && j < C ? exp(log_gamma[i] - log_gamma[j]) * float(qk_tile[idx]) : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Phase 3: Solve (I + Lw)^-1 via unipotent neumann series (reuse for (I + Lu)^-1)
    for (uint idx = lane; idx < 32 * 32; idx += 128) {
      uint r = idx >> 5, c = idx & 31;
      m_tile[idx] = l_tile[idx];
      p_w_tile[idx] = half(r == c ? 1.0f : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    invert32_unipotent(p_w_tile, l_tile, m_tile, k_tile, scratch, lane, simd_lane, simd_group);

    for (uint v0 = 0; v0 < D; v0 += D_PREFILL_TILE) {
      for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128)
        u_tile[idx] = half(0.0f);
      if (initial) {
        for (uint k0 = 0; k0 < D; k0 += D_PREFILL_TILE) {
          for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
            uint i = idx / D_PREFILL_TILE, kd = idx % D_PREFILL_TILE;
            k_tile[idx] = half(i < C ? beta_tile[i] * float(k_full[(k0 + kd) * C_PREFILL + i]) : 0.0f);
          }
          for (uint idx = lane; idx < D_PREFILL_TILE * D_PREFILL_TILE; idx += 128) {
            uint kd = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE;
            m_tile[idx] = half(current_state[state_offset(slot, h, k0 + kd, v0 + vd, num_heads)]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          // Phase 3: W = P_W @ (diag(beta) * K)
          mma32x32(w_tile, p_w_tile, k_tile, scratch, simd_lane, simd_group, false, false, false);
          // Phase 4: W S_[t]^T.
          mma32x32(u_tile, w_tile, m_tile, scratch, simd_lane, simd_group, true, false, true);
        }
      }
      // Phase 3: Apply Utilde = diag(gamma) P_U diag(inv_gamma) diag(beta) V, M_U = Gamma o M_W.
      for (uint idx = lane; idx < C_PREFILL * C_PREFILL; idx += 128) {
        uint i = idx >> 5, j = idx & 31;
        l_tile[idx] = half(i < C && j <= i ? exp(log_gamma[i] - log_gamma[j]) * float(p_w_tile[idx]) : 0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
        uint i = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, v = v0 + vd;
        k_tile[idx] = half(i < C ? beta_tile[i] * float(value[value_offset(b, chunk + i, h, v, vs0, vs1, vs2, vs3)]) : 0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma32x32(w_tile, l_tile, k_tile, scratch, simd_lane, simd_group, false, false, false);

      // Phase 4: Data payload resolution, DeltaV = Utilde - W S_[t]^T.
      for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
        uint i = idx / D_PREFILL_TILE;
        u_tile[idx] = half(i < C ? float(w_tile[idx]) - gamma[i] * float(u_tile[idx]) : 0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Phase 5: Sequence output, O = A_local DeltaV + diag(gamma) Q S_[t]^T.
      mma32x32(k_tile, qk_tile, u_tile, scratch, simd_lane, simd_group, false, false, false);
      if (initial) {
        for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128)
          w_tile[idx] = half(0.0f);
        for (uint k0 = 0; k0 < D; k0 += D_PREFILL_TILE) {
          for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
            uint i = idx / D_PREFILL_TILE, kd = idx % D_PREFILL_TILE, k = k0 + kd;
            l_tile[idx] = half(i < C ? float(query[qkv_offset(b, chunk + i, h, k, seq_len, num_heads)]) * q_norm[i] : 0.0f);
          }
          for (uint idx = lane; idx < D_PREFILL_TILE * D_PREFILL_TILE; idx += 128) {
            uint kd = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE;
            m_tile[idx] = half(current_state[state_offset(slot, h, k0 + kd, v0 + vd, num_heads)]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          mma32x32(w_tile, l_tile, m_tile, scratch, simd_lane, simd_group, true, false, true);
        }
      }
      for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
        uint i = idx / D_PREFILL_TILE, vd = idx % D_PREFILL_TILE, v = v0 + vd;
        if (i < C)
          output[qkv_offset(b, chunk + i, h, v, seq_len, num_heads)] = half(float(k_tile[idx]) + (initial ? gamma[i] * float(w_tile[idx]) : 0.0f));
      }
      // Phase 6: State transition, S_[t+1] = gamma_C S_[t] + DeltaV_state^T K.
      float gamma_c = gamma[C - 1], log_gamma_c = log_gamma[C - 1];
      for (uint idx = lane; idx < C_PREFILL * D_PREFILL_TILE; idx += 128) {
        uint i = idx / D_PREFILL_TILE;
        u_tile[idx] = half(i < C ? exp(log_gamma_c - log_gamma[i]) * float(u_tile[idx]) : 0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k0 = 0; k0 < D; k0 += D_PREFILL_TILE)
        for (uint block = simd_group; block < (D_PREFILL_TILE / 8) * (D_PREFILL_TILE / 8); block += 4) {
          uint kb = (block / (D_PREFILL_TILE / 8)) << 3, vb = (block % (D_PREFILL_TILE / 8)) << 3;
          simdgroup_matrix<half, 8, 8> kt, dv;
          simdgroup_matrix<float, 8, 8> c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          if (initial) {
            simdgroup_load(c, current_state + state_offset(slot, h, 0, 0, num_heads), D, ulong2(v0 + vb, k0 + kb));
            c.thread_elements() *= gamma_c;
          }
          for (uint ko = 0; ko < C_PREFILL; ko += 8) {
            simdgroup_load(kt, k_full, C_PREFILL, ulong2(ko, k0 + kb));
            simdgroup_load(dv, u_tile, D_PREFILL_TILE, ulong2(vb, ko));
            simdgroup_multiply_accumulate(c, kt, dv, c);
          }
          simdgroup_store(c, state + state_offset(slot, h, 0, 0, num_heads), D, ulong2(v0 + vb, k0 + kb));
        }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// One threadgroup per (B, n_heads)
[[max_total_threads_per_threadgroup(512)]]
kernel void delta_rule_decode(device half* output [[buffer(0)]], device float* state0 [[buffer(1)]], device float* state1 [[buffer(2)]],
                              device const uint* slots [[buffer(3)]], device const uint* banks [[buffer(4)]], device const uint* valid [[buffer(5)]],
                              device const half* query [[buffer(6)]], device const half* key [[buffer(7)]], device const half* value [[buffer(8)]],
                              device const float* g [[buffer(9)]], device const half* beta [[buffer(10)]], constant long& batch_size [[buffer(11)]],
                              constant long& seq_len [[buffer(12)]], constant long& num_heads [[buffer(13)]], constant long& vs0 [[buffer(14)]],
                              constant long& vs1 [[buffer(15)]], constant long& vs2 [[buffer(16)]], constant long& vs3 [[buffer(17)]],
                              uint3 gid [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]], uint3 lane3 [[thread_position_in_threadgroup]],
                              uint3 group3 [[threadgroup_position_in_grid]]) {
  uint lane = lane3.x;
  long group = group3.x;
  if (group >= batch_size * num_heads)
    return;
  long b = group / num_heads, h = group - b * num_heads;
  long slot = slots[b];
  bool has_initial_state = valid[b];
  device float* state = banks[b] ? state0 : state1;
  device const float* previous_state = banks[b] ? state1 : state0;
  threadgroup half q[D], k[D];
  threadgroup float scratch[1025];
  // No divergence: all threads evaluate to same (no risk in barrier inside if)
  if (!has_initial_state) {
    for (uint i = lane; i < D * D; i += 512)
      state[state_offset(slot, h, i / D, i % D, num_heads)] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);
  }
  device const float* input_state = has_initial_state ? previous_state : state;
  run_delta_rule_token(output, state, input_state, query, key, value, g, beta, b, slot, slot, 0, h, seq_len, num_heads, vs0, vs1, vs2, vs3, lane,
                       simd_lane, simd_group, q, k, scratch);
}

// Each target query token produces a selectable recurrent-state column; only the accepted column is committed.
[[max_total_threads_per_threadgroup(512)]]
kernel void
delta_rule_candidates(device half* output [[buffer(0)]], device const float* state0 [[buffer(1)]], device const float* state1 [[buffer(2)]],
                      device float* candidates [[buffer(3)]], device const uint* slots [[buffer(4)]], device const uint* banks [[buffer(5)]],
                      device const uint* valid [[buffer(6)]], device const half* query [[buffer(7)]], device const half* key [[buffer(8)]],
                      device const half* value [[buffer(9)]], device const float* g [[buffer(10)]], device const half* beta [[buffer(11)]],
                      constant long& batch_size [[buffer(12)]], constant long& seq_len [[buffer(13)]], constant long& num_heads [[buffer(14)]],
                      constant long& vs0 [[buffer(15)]], constant long& vs1 [[buffer(16)]], constant long& vs2 [[buffer(17)]],
                      constant long& vs3 [[buffer(18)]], uint3 gid [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                      uint simd_group [[simdgroup_index_in_threadgroup]], uint3 lane3 [[thread_position_in_threadgroup]],
                      uint3 group3 [[threadgroup_position_in_grid]]) {
  uint lane = lane3.x;
  long group = group3.x;
  if (group >= batch_size * num_heads)
    return;
  long b = group / num_heads, h = group - b * num_heads, slot = slots[b];
  device const float* committed = banks[b] ? state1 : state0;
  threadgroup half q[D], k[D];
  threadgroup float scratch[1025];
  for (long t = 0; t < seq_len; ++t) {
    long row = b * seq_len + t, previous_row = t ? row - 1 : valid[b] ? slot : row;
    device const float* previous = t || !valid[b] ? candidates : committed;
    if (!valid[b] && t == 0)
      for (uint i = lane; i < D * D; i += 512)
        candidates[state_offset(row, h, i / D, i % D, num_heads)] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);
    run_delta_rule_token(output, candidates, previous, query, key, value, g, beta, b, row, previous_row, t, h, seq_len, num_heads, vs0, vs1, vs2, vs3,
                         lane, simd_lane, simd_group, q, k, scratch);
    threadgroup_barrier(mem_flags::mem_device);
  }
}
