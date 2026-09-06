#include <metal_stdlib>
#include <metal_matrix>
using namespace metal;

constant float iq4nl[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f, 1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f};

static inline half h16(device const uchar* p) {
  ushort v = ushort(p[0]) | (ushort(p[1]) << 8);
  return as_type<half>(v);
}

static inline void scale_min_k4(uint j, device const uchar* q, thread uchar& d, thread uchar& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

struct q4k_tag {};
struct q5k_tag {};
struct q6k_tag {};
struct q8_0_tag {};
struct iq4xs_tag {};

template <uint K, uint COLUMNS = 2, uint STRIDE = 33>
static inline __attribute__((always_inline)) void dequant_tile(q4k_tag, threadgroup half* b_tile, device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
  for (uint p = 0; p < COLUMNS; ++p) {
    uint n = simd_group * COLUMNS + p;
    long o = long(n0 + n) * (K / 256) * 144 + kb * 144;
    float d = float(h16(w + o)), dm = float(h16(w + o + 2));
    for (uint t = 0; t < 4; ++t) {
      uchar sc, mn, q = w[o + 16 + t * 32 + simd_lane];
      uint r = t * 64 + simd_lane, j = t * 2;
      scale_min_k4(j, w + o + 4, sc, mn);
      b_tile[r * STRIDE + n] = half(d * float(sc) * float(q & 15) - dm * float(mn));
      scale_min_k4(j + 1, w + o + 4, sc, mn);
      b_tile[(r + 32) * STRIDE + n] = half(d * float(sc) * float(q >> 4) - dm * float(mn));
    }
  }
}

template <uint K, uint COLUMNS = 2, uint STRIDE = 33>
static inline __attribute__((always_inline)) void dequant_tile(q5k_tag, threadgroup half* b_tile, device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
  for (uint p = 0; p < COLUMNS; ++p) {
    uint n = simd_group * COLUMNS + p;
    long o = long(n0 + n) * (K / 256) * 176 + kb * 176;
    float d = float(h16(w + o)), dm = float(h16(w + o + 2));
    uchar hm = w[o + 16 + simd_lane];
    for (uint t = 0; t < 4; ++t) {
      uchar sc, mn, q = w[o + 48 + t * 32 + simd_lane];
      uint r = t * 64 + simd_lane, j = t * 2;
      scale_min_k4(j, w + o + 4, sc, mn);
      b_tile[r * STRIDE + n] = half(d * float(sc) * float((q & 15) + ((hm & (1 << j)) ? 16 : 0)) - dm * float(mn));
      scale_min_k4(j + 1, w + o + 4, sc, mn);
      b_tile[(r + 32) * STRIDE + n] = half(d * float(sc) * float((q >> 4) + ((hm & (1 << (j + 1))) ? 16 : 0)) - dm * float(mn));
    }
  }
}

template <uint K, uint COLUMNS = 2, uint STRIDE = 33>
static inline __attribute__((always_inline)) void dequant_tile(q6k_tag, threadgroup half* b_tile, device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
  for (uint p = 0; p < COLUMNS; ++p) {
    uint n = simd_group * COLUMNS + p, l = simd_lane;
    long o = long(n0 + n) * (K / 256) * 210 + kb * 210;
    float d = float(h16(w + o + 208));
    for (uint h = 0; h < 2; ++h)
      for (uint s = 0; s < 4; ++s) {
        uint rr = s * 32 + l, qlo = h * 64 + ((rr & 32) ? 32 : 0) + l, r = h * 128 + rr;
        uchar q = w[o + qlo], lo = (rr & 64) ? (q >> 4) : (q & 15);
        uchar hi = (w[o + 128 + h * 32 + l] >> (2 * s)) & 3;
        char sc = char(w[o + 192 + h * 8 + (rr >> 4)]);
        b_tile[r * STRIDE + n] = half(d * float(sc) * (float((hi << 4) | lo) - 32.0f));
      }
  }
}

template <uint K, uint COLUMNS = 2, uint STRIDE = 33>
static inline __attribute__((always_inline)) void dequant_tile(q8_0_tag, threadgroup half* b_tile, device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
  for (uint p = 0; p < COLUMNS; ++p) {
    uint n = simd_group * COLUMNS + p;
    long base = long(n0 + n) * (K / 32) * 34 + kb * 8 * 34;
    for (uint t = 0; t < 8; ++t) {
      long o = base + t * 34;
      b_tile[(t * 32 + simd_lane) * STRIDE + n] = half(float(h16(w + o)) * float(char(w[o + 2 + simd_lane])));
    }
  }
}

template <uint K, uint COLUMNS = 2, uint STRIDE = 33>
static inline __attribute__((always_inline)) void dequant_tile(iq4xs_tag, threadgroup half* b_tile, device const uchar* w, long n0, long kb,
                                                               uint simd_lane, uint simd_group) {
  for (uint p = 0; p < COLUMNS; ++p) {
    uint n = simd_group * COLUMNS + p;
    long o = long(n0 + n) * (K / 256) * 136 + kb * 136;
    float d = float(h16(w + o));
    ushort sh = ushort(w[o + 2]) | (ushort(w[o + 3]) << 8);
    for (uint j = 0; j < 8; ++j) {
      uint r = j * 32 + simd_lane, qj = j * 16 + (simd_lane & 15);
      int ls = int((w[o + 4 + (j >> 1)] >> (4 * (j & 1))) & 15) | int(((sh >> (2 * j)) & 3) << 4);
      uchar q = w[o + 8 + qj], v = (simd_lane & 16) ? (q >> 4) : (q & 15);
      b_tile[r * STRIDE + n] = half(d * float(ls - 32) * iq4nl[v]);
    }
  }
}

template <typename Q, uint K, uint N>
[[max_total_threads_per_threadgroup(64)]]
kernel void prefill_qk_small(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], device const uchar* w [[buffer(2)]],
                             constant long& M [[buffer(3)]], uint3 lane3 [[thread_position_in_threadgroup]],
                             uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]]) {
  uint lane = lane3.x;
  long n0 = long(group.x) * 8;
  threadgroup half b_tile[256 * 9];
  simdgroup_matrix<float, 8, 8> c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (long k0 = 0, kb = 0; k0 < K; k0 += 256, ++kb) {
    dequant_tile<K, 4, 9>(Q(), b_tile, w, n0, kb, simd_lane, simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<half, 8, 8> a, b;
    uint k_begin = simd_group * 128;
    for (uint ko = k_begin; ko < k_begin + 128; ko += 8) {
      simdgroup_load(a, x + k0, K, ulong2(ko, 0));
      simdgroup_load(b, b_tile, 9, ulong2(0, ko));
      simdgroup_multiply_accumulate(c, a, b, c);
    }
    threadgroup_barrier(mem_flags::mem_none);
  }
  threadgroup float* scratch = reinterpret_cast<threadgroup float*>(b_tile);
  simdgroup_store(c, scratch + simd_group * 64, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint idx = lane; idx < uint(M) * 8; idx += 64) {
    uint r = idx >> 3, n = idx & 7, offset = r * 8 + n;
    y[r * N + n0 + n] = half(scratch[offset] + scratch[offset + 64]);
  }
}

template <typename Q, uint K, uint N>
[[max_total_threads_per_threadgroup(512)]]
kernel void prefill_qk(device half* y [[buffer(0)]], device const half* x [[buffer(1)]], device const uchar* w [[buffer(2)]],
                       constant long& M [[buffer(3)]], uint3 lane3 [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                       uint simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  uint lane = lane3.x, rb = simd_group * 8;
  long n0 = long(group.x) * 32, m0 = long(group.y) * 128;
  bool active = m0 + rb < M;
  threadgroup half b_tile[256 * 33];
  if (M <= 8) {
    simdgroup_matrix<float, 8, 8> c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (long k0 = 0, kb = 0; k0 < K; k0 += 256, ++kb) {
      dequant_tile<K>(Q(), b_tile, w, n0, kb, simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_group < 4) {
        simdgroup_matrix<half, 8, 8> a, b;
        for (uint ko = 0; ko < 256; ko += 8) {
          simdgroup_load(a, x + k0, K, ulong2(ko, 0));
          simdgroup_load(b, b_tile, 33, ulong2(simd_group * 8, ko));
          simdgroup_multiply_accumulate(c, a, b, c);
        }
      }
      threadgroup_barrier(mem_flags::mem_none);
    }
    threadgroup float* scratch = reinterpret_cast<threadgroup float*>(b_tile);
    if (simd_group < 4)
      simdgroup_store(c, scratch + simd_group * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < uint(M) * 32; idx += 512) {
      uint r = idx >> 5, n = idx & 31, tile = n >> 3;
      y[r * N + n0 + n] = half(scratch[tile * 64 + r * 8 + (n & 7)]);
    }
    return;
  }
  simdgroup_matrix<float, 8, 8> c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_matrix<float, 8, 8> c1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_matrix<float, 8, 8> c2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_matrix<float, 8, 8> c3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (long k0 = 0, kb = 0; k0 < K; k0 += 256, ++kb) {
    device const half* x_ptr = x + m0 * K + k0;
    dequant_tile<K>(Q(), b_tile, w, n0, kb, simd_lane, simd_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      simdgroup_matrix<half, 8, 8> a, b;
      for (uint ko = 0; ko < 256; ko += 8) {
        simdgroup_load(a, x_ptr, K, ulong2(ko, rb));
        simdgroup_load(b, b_tile, 33, ulong2(0, ko));
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_load(b, b_tile, 33, ulong2(8, ko));
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_load(b, b_tile, 33, ulong2(16, ko));
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_load(b, b_tile, 33, ulong2(24, ko));
        simdgroup_multiply_accumulate(c3, a, b, c3);
      }
    }
    threadgroup_barrier(mem_flags::mem_none);
  }
  threadgroup float* scratch = reinterpret_cast<threadgroup float*>(b_tile);
  if (active) {
    simdgroup_store(c0, scratch + simd_group * 256, 32);
    simdgroup_store(c1, scratch + simd_group * 256 + 8, 32);
    simdgroup_store(c2, scratch + simd_group * 256 + 16, 32);
    simdgroup_store(c3, scratch + simd_group * 256 + 24, 32);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device half2* y2 = reinterpret_cast<device half2*>(y);
  uint rows = uint(min(128l, M - m0));
  for (uint idx = lane; idx < rows * 16; idx += 512) {
    uint r = idx >> 4, cp = idx & 15, e = (r & 7) * 32 + (cp << 1);
    y2[((m0 + r) * N + n0 + (cp << 1)) >> 1] = half2(half(scratch[(r >> 3) * 256 + e]), half(scratch[(r >> 3) * 256 + e + 1]));
  }
}

// DECODE KERNELS

template <bool ADD>
static inline __attribute__((always_inline)) void decode_store(device half* dst, device const half* residual, float value, uint row) {
  half result = half(value);
  if (ADD)
    result += residual[row];
  dst[row] = result;
}

template <bool ADD, uint K, uint N, ushort ROWS>
static inline __attribute__((always_inline)) void decode_q4k_impl(device half* dst, device const half* src, device const uchar* weights,
                                                                  device const half* residual, ushort lane, ushort simd_group, uint3 group) {
  dst += group.y * N;
  src += group.y * K;
  residual += group.y * N;
  constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  ushort ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
  uint nb = K / 256, first_row = (group.x * 2 + simd_group) * ROWS;
  float yl[16], yh[16], sumf[ROWS] = {0.0f};
  device const half* src4 = src + ix * 256 + 64 * iq + 8 * ir;

  for (uint ib = ix; ib < nb; ib += 4) {
    float4 sumy = 0.0f;
    for (ushort i = 0; i < 8; ++i) {
      // load(x_i) -> sum(x_i)
      yl[i] = float(src4[i]);
      sumy[0] += yl[i];
      yl[i + 8] = float(src4[i + 32]);
      sumy[1] += yl[i + 8];
      yh[i] = float(src4[i + 128]);
      sumy[2] += yh[i];
      yh[i + 8] = float(src4[i + 160]);
      sumy[3] += yh[i + 8];
    }
    for (ushort row = 0; row < ROWS; ++row) { // one iter per column of w^t owned by SIMD
      long o = long(first_row + row) * nb * 144 + ib * 144;
      device const ushort* sc = reinterpret_cast<device const ushort*>(weights + o + 4) + iq;
      device const ushort* q1 = reinterpret_cast<device const ushort*>(weights + o + 16) + 16 * iq + 4 * ir;
      device const ushort* q2 = q1 + 32;
      device const half* dh = reinterpret_cast<device const half*>(weights + o);
      ushort sc16[4];
      thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);
      // load 6 bytes that the team owns (mi, si) in i = {1, 2, 5, 6} or {3, 4, 7, 8}
      sc16[0] = sc[0] & kmask1;
      sc16[1] = sc[2] & kmask1;
      sc16[2] = (sc[4] & kmask2) | ((sc[0] & kmask3) >> 2);
      sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

      float4 acc1 = 0.0f, acc2 = 0.0f;
      // 4(loop) * 2(q1,q2) * 2(2byte LOADS) = 16 bytes loaded per lane * 8 (threads_per_cohort) = 128 bytes (256 weight block)
      for (ushort i = 0; i < 4; ++i) {
        // sum(xi . qi) unshifted (only mask to isolate the rest 12b)
        acc1[0] += yl[2 * i] * float(q1[i] & 0x000f);
        acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0f00);
        acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00f0);
        acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xf000);
        acc2[0] += yh[2 * i] * float(q2[i] & 0x000f);
        acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0f00);
        acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00f0);
        acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xf000);
      }
      // remove the shifts + apply group scales and mins + global scales and mins
      sumf[row] += float(dh[0]) * ((acc1[0] + acc1[1] / 256.0f) * sc8[0] + (acc1[2] + acc1[3] / 256.0f) * sc8[1] / 16.0f +
                                   (acc2[0] + acc2[1] / 256.0f) * sc8[4] + (acc2[2] + acc2[3] / 256.0f) * sc8[5] / 16.0f) -
                   float(dh[1]) * dot(sumy, float4(sc8[2], sc8[3], sc8[6], sc8[7]));
    }
    src4 += 4 * 256;
  }
  for (ushort row = 0; row < ROWS; ++row) {
    float sum = simd_sum(sumf[row]);
    if (lane == 0 && first_row + row < N)
      decode_store<ADD>(dst, residual, sum, first_row + row);
  }
}

template <uint K, uint N, ushort ROWS = 2>
[[max_total_threads_per_threadgroup(64)]]
kernel void decode_q4k(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                       ushort lane [[thread_index_in_simdgroup]], ushort simd_group [[simdgroup_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
  decode_q4k_impl<false, K, N, ROWS>(dst, src, weights, dst, lane, simd_group, group);
}

template <uint K, uint N, ushort ROWS = 2>
[[max_total_threads_per_threadgroup(64)]]
kernel void decode_q4k_add(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                           device const half* residual [[buffer(3)]], ushort lane [[thread_index_in_simdgroup]],
                           ushort simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  decode_q4k_impl<true, K, N, ROWS>(dst, src, weights, residual, lane, simd_group, group);
}

template <bool ADD, uint K, uint N>
static inline __attribute__((always_inline)) void decode_q5k_impl(device half* dst, device const half* src, device const uchar* weights,
                                                                  device const half* residual, ushort lane, ushort simd_group, uint3 group) {
  dst += group.y * N;
  src += group.y * K;
  residual += group.y * N;
  constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  uint nb = K / 256, row = group.x * 4 + simd_group;
  ushort tid = lane / 4, ix = lane % 4, iq = tid / 4, ir = tid % 4, l0 = 8 * ir;
  ushort q_offset = 32 * iq + l0, y_offset = 64 * iq + l0;
  uchar hm1 = 1u << (2 * iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
  float sumf = 0.0f;
  device const half* src1 = src + ix * 256 + y_offset;

  for (uint ib = ix; ib < nb; ib += 4) {
    device const half* src2 = src1 + 128;
    long o = long(row) * nb * 176 + ib * 176;
    device const uchar* q1 = weights + o + 48 + q_offset;
    device const uchar* q2 = q1 + 64;
    device const uchar* qh = weights + o + 16 + l0;
    device const ushort* sc = reinterpret_cast<device const ushort*>(weights + o + 4) + iq;
    device const half* dh = reinterpret_cast<device const half*>(weights + o);
    ushort sc16[4];
    thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);
    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = (sc[4] & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
    float sumq = 0.0f, summin = 0.0f;
    // This is doing 4 sequential steps of accumulations (one per group it owns),
    // seeing the .air generated qh, q1, q2 are loaded only once and reused for the two groups they load
    {
      float acc1 = 0.0f, acc2 = 0.0f, sumy = 0.0f;
#pragma clang loop unroll(full)
      for (ushort l = 0; l < 8; ++l) {
        float y = float(src1[l]);
        uchar h = qh[l];
        sumy += y;
        acc1 += y * float(q1[l] & 0x0f);
        acc2 += h & hm1 ? y : 0.0f;
      }
      sumq += sc8[0] * (acc1 + 16.0f * acc2);
      summin += sc8[2] * sumy;
    }
    {
      float acc1 = 0.0f, acc2 = 0.0f, sumy = 0.0f;
#pragma clang loop unroll(full)
      for (ushort l = 0; l < 8; ++l) {
        float y = float(src1[l + 32]);
        uchar h = qh[l];
        sumy += y;
        acc1 += y * float(q1[l] & 0xf0);
        acc2 += h & hm2 ? y : 0.0f;
      }
      sumq += sc8[1] * (acc1 / 16.0f + 16.0f * acc2);
      summin += sc8[3] * sumy;
    }
    {
      float acc1 = 0.0f, acc2 = 0.0f, sumy = 0.0f;
#pragma clang loop unroll(full)
      for (ushort l = 0; l < 8; ++l) {
        float y = float(src2[l]);
        uchar h = qh[l];
        sumy += y;
        acc1 += y * float(q2[l] & 0x0f);
        acc2 += h & hm3 ? y : 0.0f;
      }
      sumq += sc8[4] * (acc1 + 16.0f * acc2);
      summin += sc8[6] * sumy;
    }
    {
      float acc1 = 0.0f, acc2 = 0.0f, sumy = 0.0f;
#pragma clang loop unroll(full)
      for (ushort l = 0; l < 8; ++l) {
        float y = float(src2[l + 32]);
        uchar h = qh[l];
        sumy += y;
        acc1 += y * float(q2[l] & 0xf0);
        acc2 += h & hm4 ? y : 0.0f;
      }
      sumq += sc8[5] * (acc1 / 16.0f + 16.0f * acc2);
      summin += sc8[7] * sumy;
    }
    sumf += float(dh[0]) * sumq - float(dh[1]) * summin;
    src1 += 4 * 256;
  }
  float sum = simd_sum(sumf);
  if (lane == 0 && row < N)
    decode_store<ADD>(dst, residual, sum, row);
}

template <uint K, uint N>
[[max_total_threads_per_threadgroup(128)]]
kernel void decode_q5k(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                       ushort lane [[thread_index_in_simdgroup]], ushort simd_group [[simdgroup_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
  decode_q5k_impl<false, K, N>(dst, src, weights, dst, lane, simd_group, group);
}

template <uint K, uint N>
[[max_total_threads_per_threadgroup(128)]]
kernel void decode_q5k_add(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                           device const half* residual [[buffer(3)]], ushort lane [[thread_index_in_simdgroup]],
                           ushort simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  decode_q5k_impl<true, K, N>(dst, src, weights, residual, lane, simd_group, group);
}

template <bool ADD, uint K, uint N>
static inline __attribute__((always_inline)) void decode_q6k_impl(device half* dst, device const half* src, device const uchar* weights,
                                                                  device const half* residual, ushort lane, ushort simd_group, uint3 group) {
  dst += group.y * N;
  src += group.y * K;
  residual += group.y * N;
  constexpr uint nb = K / 256;
  constexpr ulong row_stride = ulong(nb) * 210;
  const uint first_row = (group.x * 4 + simd_group) * 2;
  const ushort tid = lane / 2;
  const ushort ix = lane % 2;
  const ushort ip = tid / 8;
  const ushort il = tid % 8;
  const ushort l0 = 4 * il;
  const ushort is = 8 * ip + l0 / 16;
  const ushort y_offset = 128 * ip + l0;
  const ushort q_offset_l = 64 * ip + l0;
  const ushort q_offset_h = 32 * ip + l0;
  float yl[16];
  float sumf[2] = {0.0f, 0.0f};

  for (uint ib = ix; ib < nb; ib += 2) {
    device const half* y = src + ib * 256 + y_offset;
#pragma clang loop unroll(full)
    for (ushort l = 0; l < 4; ++l) {
      yl[4 * l + 0] = y[l + 0];
      yl[4 * l + 1] = y[l + 32];
      yl[4 * l + 2] = y[l + 64];
      yl[4 * l + 3] = y[l + 96];
    }

    device const uchar* w = weights + ulong(first_row) * row_stride + ulong(ib) * 210;
#pragma clang loop unroll(full)
    for (ushort row = 0; row < 2; ++row) {
      device const uchar* q1 = w + q_offset_l;
      device const uchar* q2 = q1 + 32;
      device const uchar* qh = w + 128 + q_offset_h;
      device const char* sc = reinterpret_cast<device const char*>(w + 192 + is);
      float4 sums = 0.0f;

#pragma clang loop unroll(full)
      for (ushort l = 0; l < 4; ++l) {
        sums[0] += yl[4 * l + 0] * (int((q1[l] & 0x0f) | ((qh[l] & 0x03) << 4)) - 32);
        sums[1] += yl[4 * l + 1] * (int((q2[l] & 0x0f) | ((qh[l] & 0x0c) << 2)) - 32);
        sums[2] += yl[4 * l + 2] * (int((q1[l] >> 4) | (qh[l] & 0x30)) - 32);
        sums[3] += yl[4 * l + 3] * (int((q2[l] >> 4) | ((qh[l] & 0xc0) >> 2)) - 32);
      }

      const float d = float(*reinterpret_cast<device const half*>(w + 208));
      sumf[row] += d * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
      w += row_stride;
    }
  }

#pragma clang loop unroll(full)
  for (ushort row = 0; row < 2; ++row) {
    const float sum = simd_sum(sumf[row]);
    if (lane == 0 && first_row + row < N)
      decode_store<ADD>(dst, residual, sum, first_row + row);
  }
}

template <uint K, uint N>
[[max_total_threads_per_threadgroup(128)]]
kernel void decode_q6k(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                       ushort lane [[thread_index_in_simdgroup]], ushort simd_group [[simdgroup_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
  decode_q6k_impl<false, K, N>(dst, src, weights, dst, lane, simd_group, group);
}

template <uint K, uint N>
[[max_total_threads_per_threadgroup(128)]]
kernel void decode_q6k_add(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                           device const half* residual [[buffer(3)]], ushort lane [[thread_index_in_simdgroup]],
                           ushort simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  decode_q6k_impl<true, K, N>(dst, src, weights, residual, lane, simd_group, group);
}

template <bool ADD, uint K, uint N>
static inline __attribute__((always_inline)) void decode_q8_0_impl(device half* dst, device const half* src, device const uchar* weights,
                                                                   device const half* residual, threadgroup float* partial, ushort lane,
                                                                   ushort simd_group, uint3 group) {
  dst += group.y * N;
  src += group.y * K;
  residual += group.y * N;
  uint nb = K / 32, first_row = group.x * 2;
  ushort ix = lane / 4, il = lane % 4;
  uint ib0 = simd_group * 8 + ix;
  float yl[8], sumf[2] = {0.0f, 0.0f};
  device const half* y = src + ib0 * 32 + il * 8;
  for (uint ib = ib0; ib < nb; ib += 32) {
    for (ushort i = 0; i < 8; ++i)
      yl[i] = float(y[i]);
    for (ushort row = 0; row < 2; ++row) {
      long o = long(first_row + row) * nb * 34 + ib * 34;
      device const char* qs = reinterpret_cast<device const char*>(weights + o + 2 + il * 8);
      float sumq = 0.0f;
      for (ushort i = 0; i < 8; ++i)
        sumq += float(qs[i]) * yl[i];
      sumf[row] += sumq * float(*reinterpret_cast<device const half*>(weights + o));
    }
    y += 32 * 32;
  }
  for (ushort row = 0; row < 2; ++row) {
    float sum = simd_sum(sumf[row]);
    if (lane == 0)
      partial[simd_group * 2 + row] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0 && lane == 0) {
    for (ushort row = 0; row < 2; ++row) {
      float sum = partial[row] + partial[2 + row] + partial[4 + row] + partial[6 + row];
      if (first_row + row < N)
        decode_store<ADD>(dst, residual, sum, first_row + row);
    }
  }
}

template <uint K, uint N>
[[max_total_threads_per_threadgroup(128)]]
kernel void decode_q8_0(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                        ushort lane [[thread_index_in_simdgroup]], ushort simd_group [[simdgroup_index_in_threadgroup]],
                        uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup float partial[8];
  decode_q8_0_impl<false, K, N>(dst, src, weights, dst, partial, lane, simd_group, group);
}

template <uint K, uint N>
[[max_total_threads_per_threadgroup(128)]]
kernel void decode_q8_0_add(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                            device const half* residual [[buffer(3)]], ushort lane [[thread_index_in_simdgroup]],
                            ushort simd_group [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup float partial[8];
  decode_q8_0_impl<true, K, N>(dst, src, weights, residual, partial, lane, simd_group, group);
}

template <uint K, uint N>
[[max_total_threads_per_threadgroup(64)]]
kernel void decode_iq4xs(device half* dst [[buffer(0)]], device const half* src [[buffer(1)]], device const uchar* weights [[buffer(2)]],
                         ushort lane [[thread_index_in_simdgroup]], ushort simd_group [[simdgroup_index_in_threadgroup]],
                         uint3 group [[threadgroup_position_in_grid]]) {
  // Qwen 3.5 IQ4_XS tensors are MLP gate/up weights; decode always uses the fused two-weight kernel.
}

[[max_total_threads_per_threadgroup(256)]]
kernel void q4_k_embed(device half* y [[buffer(0)]], device const int* ids [[buffer(1)]], device const uchar* w [[buffer(2)]],
                       uint3 lane3 [[thread_position_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
  uint col = group.x * 256 + lane3.x, t = group.y;
  long row = ids[t], nb = 4096 / 256, o = row * nb * 144 + (col >> 8) * 144;
  uint r = col & 255, j = r >> 5, qj = (r >> 6) * 32 + (r & 31);
  uchar sc, mn;
  scale_min_k4(j, w + o + 4, sc, mn);
  uchar q = w[o + 16 + qj], v = (r & 32) ? (q >> 4) : (q & 15);
  y[t * 4096 + col] = half(float(h16(w + o)) * float(sc) * float(v) - float(h16(w + o + 2)) * float(mn));
}

#define PREFILL_ARGS device half*, device const half*, device const uchar*, constant long&, uint3, uint, uint, uint3
template [[host_name("q4_k_k4096_n1024_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 1024>(PREFILL_ARGS);
template [[host_name("q4_k_k4096_n1024_prefill_small")]] kernel void prefill_qk_small<q4k_tag, 4096, 1024>(PREFILL_ARGS);
template [[host_name("q4_k_k4096_n4096_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 4096>(PREFILL_ARGS);
template [[host_name("q4_k_k4096_n4096_prefill_small")]] kernel void prefill_qk_small<q4k_tag, 4096, 4096>(PREFILL_ARGS);
template [[host_name("q4_k_k12288_n4096_prefill")]] kernel void prefill_qk<q4k_tag, 12288, 4096>(PREFILL_ARGS);
template [[host_name("q4_k_k12288_n4096_prefill_small")]] kernel void prefill_qk_small<q4k_tag, 12288, 4096>(PREFILL_ARGS);
template [[host_name("q4_k_k32768_n4096_prefill")]] kernel void prefill_qk<q4k_tag, 32768, 4096>(PREFILL_ARGS);
template [[host_name("q4_k_k32768_n4096_prefill_small")]] kernel void prefill_qk_small<q4k_tag, 32768, 4096>(PREFILL_ARGS);
template [[host_name("q4_k_k4096_n8192_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 8192>(PREFILL_ARGS);
template [[host_name("q4_k_k4096_n8192_prefill_small")]] kernel void prefill_qk_small<q4k_tag, 4096, 8192>(PREFILL_ARGS);
template [[host_name("q4_k_k4096_n12288_prefill")]] kernel void prefill_qk<q4k_tag, 4096, 12288>(PREFILL_ARGS);
template [[host_name("q4_k_k4096_n12288_prefill_small")]] kernel void prefill_qk_small<q4k_tag, 4096, 12288>(PREFILL_ARGS);
#define DECODE_ARGS device half*, device const half*, device const uchar*, ushort, ushort, uint3
#define DECODE_ADD_ARGS device half*, device const half*, device const uchar*, device const half*, ushort, ushort, uint3

template [[host_name("q4_k_k4096_n1024_decode")]] kernel void decode_q4k<4096, 1024>(DECODE_ARGS);
template [[host_name("q4_k_k4096_n4096_decode")]] kernel void decode_q4k<4096, 4096, 1>(DECODE_ARGS);
template [[host_name("q4_k_k4096_n4096_decode_add")]] kernel void decode_q4k_add<4096, 4096, 1>(DECODE_ADD_ARGS);
template [[host_name("q4_k_k12288_n4096_decode")]] kernel void decode_q4k<12288, 4096>(DECODE_ARGS);
template [[host_name("q4_k_k12288_n4096_decode_add")]] kernel void decode_q4k_add<12288, 4096>(DECODE_ADD_ARGS);
template [[host_name("q4_k_k32768_n4096_decode")]] kernel void decode_q4k<32768, 4096, 1>(DECODE_ARGS);
template [[host_name("q4_k_k32768_n4096_decode_add")]] kernel void decode_q4k_add<32768, 4096, 1>(DECODE_ADD_ARGS);
template [[host_name("q4_k_k4096_n8192_decode")]] kernel void decode_q4k<4096, 8192>(DECODE_ARGS);
template [[host_name("q4_k_k4096_n12288_decode")]] kernel void decode_q4k<4096, 12288>(DECODE_ARGS);

template [[host_name("q5_k_k4096_n1024_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 1024>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n1024_prefill_small")]] kernel void prefill_qk_small<q5k_tag, 4096, 1024>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n4096_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 4096>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n4096_prefill_small")]] kernel void prefill_qk_small<q5k_tag, 4096, 4096>(PREFILL_ARGS);
template [[host_name("q5_k_k12288_n4096_prefill")]] kernel void prefill_qk<q5k_tag, 12288, 4096>(PREFILL_ARGS);
template [[host_name("q5_k_k12288_n4096_prefill_small")]] kernel void prefill_qk_small<q5k_tag, 12288, 4096>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n8192_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 8192>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n8192_prefill_small")]] kernel void prefill_qk_small<q5k_tag, 4096, 8192>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n12288_prefill")]] kernel void prefill_qk<q5k_tag, 4096, 12288>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n12288_prefill_small")]] kernel void prefill_qk_small<q5k_tag, 4096, 12288>(PREFILL_ARGS);
template [[host_name("q5_k_k4096_n1024_decode")]] kernel void decode_q5k<4096, 1024>(DECODE_ARGS);
template [[host_name("q5_k_k4096_n4096_decode")]] kernel void decode_q5k<4096, 4096>(DECODE_ARGS);
template [[host_name("q5_k_k4096_n4096_decode_add")]] kernel void decode_q5k_add<4096, 4096>(DECODE_ADD_ARGS);
template [[host_name("q5_k_k12288_n4096_decode")]] kernel void decode_q5k<12288, 4096>(DECODE_ARGS);
template [[host_name("q5_k_k12288_n4096_decode_add")]] kernel void decode_q5k_add<12288, 4096>(DECODE_ADD_ARGS);
template [[host_name("q5_k_k4096_n8192_decode")]] kernel void decode_q5k<4096, 8192>(DECODE_ARGS);
template [[host_name("q5_k_k4096_n12288_decode")]] kernel void decode_q5k<4096, 12288>(DECODE_ARGS);

template [[host_name("q6_k_k4096_n1024_prefill")]] kernel void prefill_qk<q6k_tag, 4096, 1024>(PREFILL_ARGS);
template [[host_name("q6_k_k4096_n1024_prefill_small")]] kernel void prefill_qk_small<q6k_tag, 4096, 1024>(PREFILL_ARGS);
template [[host_name("q6_k_k4096_n8192_prefill")]] kernel void prefill_qk<q6k_tag, 4096, 8192>(PREFILL_ARGS);
template [[host_name("q6_k_k4096_n8192_prefill_small")]] kernel void prefill_qk_small<q6k_tag, 4096, 8192>(PREFILL_ARGS);
template [[host_name("q6_k_k12288_n4096_prefill")]] kernel void prefill_qk<q6k_tag, 12288, 4096>(PREFILL_ARGS);
template [[host_name("q6_k_k12288_n4096_prefill_small")]] kernel void prefill_qk_small<q6k_tag, 12288, 4096>(PREFILL_ARGS);
template [[host_name("q6_k_k4096_n248320_prefill")]] kernel void prefill_qk<q6k_tag, 4096, 248320>(PREFILL_ARGS);
template [[host_name("q6_k_k4096_n248320_prefill_small")]] kernel void prefill_qk_small<q6k_tag, 4096, 248320>(PREFILL_ARGS);
template [[host_name("q6_k_k4096_n1024_decode")]] kernel void decode_q6k<4096, 1024>(DECODE_ARGS);
template [[host_name("q6_k_k4096_n8192_decode")]] kernel void decode_q6k<4096, 8192>(DECODE_ARGS);
template [[host_name("q6_k_k12288_n4096_decode")]] kernel void decode_q6k<12288, 4096>(DECODE_ARGS);
template [[host_name("q6_k_k12288_n4096_decode_add")]] kernel void decode_q6k_add<12288, 4096>(DECODE_ADD_ARGS);
template [[host_name("q6_k_k4096_n248320_decode")]] kernel void decode_q6k<4096, 248320>(DECODE_ARGS);

template [[host_name("q8_0_k4096_n4096_prefill")]] kernel void prefill_qk<q8_0_tag, 4096, 4096>(PREFILL_ARGS);
template [[host_name("q8_0_k4096_n4096_prefill_small")]] kernel void prefill_qk_small<q8_0_tag, 4096, 4096>(PREFILL_ARGS);
template [[host_name("q8_0_k4096_n4096_decode")]] kernel void decode_q8_0<4096, 4096>(DECODE_ARGS);
template [[host_name("q8_0_k4096_n4096_decode_add")]] kernel void decode_q8_0_add<4096, 4096>(DECODE_ADD_ARGS);
template [[host_name("q8_0_k8192_n4096_prefill")]] kernel void prefill_qk<q8_0_tag, 8192, 4096>(PREFILL_ARGS);
template [[host_name("q8_0_k8192_n4096_prefill_small")]] kernel void prefill_qk_small<q8_0_tag, 8192, 4096>(PREFILL_ARGS);
template [[host_name("q8_0_k8192_n4096_decode")]] kernel void decode_q8_0<8192, 4096>(DECODE_ARGS);
template [[host_name("q8_0_k8192_n4096_decode_add")]] kernel void decode_q8_0_add<8192, 4096>(DECODE_ADD_ARGS);

template [[host_name("iq4_xs_k4096_n12288_prefill")]] kernel void prefill_qk<iq4xs_tag, 4096, 12288>(PREFILL_ARGS);
template [[host_name("iq4_xs_k4096_n12288_prefill_small")]] kernel void prefill_qk_small<iq4xs_tag, 4096, 12288>(PREFILL_ARGS);
template [[host_name("iq4_xs_k4096_n12288_decode")]] kernel void decode_iq4xs<4096, 12288>(DECODE_ARGS);
