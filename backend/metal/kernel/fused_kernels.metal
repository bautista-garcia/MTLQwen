#include <metal_stdlib>
using namespace metal;

constant uint GDN_C = 8192;
constant uint GDN_D = 128;
constant uint MLP_K = 4096;
constant uint MLP_N = 12288;

constant float iq4nl[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                              1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f};

static inline __attribute__((always_inline)) void q4k_accumulate(
        device const uchar* weights, uint nb, uint ib, uint first_row, ushort iq, ushort ir,
        thread const float* yl, thread const float* yh, float4 sumy, thread float* sumf) {
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    for (ushort row = 0; row < 2; ++row) {
        long o = long(first_row + row) * nb * 144 + ib * 144;
        device const ushort* sc = reinterpret_cast<device const ushort*>(weights + o + 4) + iq;
        device const ushort* q1 = reinterpret_cast<device const ushort*>(weights + o + 16) + 16 * iq + 4 * ir;
        device const ushort* q2 = q1 + 32;
        device const half* dh = reinterpret_cast<device const half*>(weights + o);
        ushort sc16[4];
        thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);
        sc16[0] = sc[0] & kmask1;
        sc16[1] = sc[2] & kmask1;
        sc16[2] = (sc[4] & kmask2) | ((sc[0] & kmask3) >> 2);
        sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

        float4 acc1 = 0.0f, acc2 = 0.0f;
        for (ushort i = 0; i < 4; ++i) {
            acc1[0] += yl[2 * i] * float(q1[i] & 0x000f);
            acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0f00);
            acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00f0);
            acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xf000);
            acc2[0] += yh[2 * i] * float(q2[i] & 0x000f);
            acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0f00);
            acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00f0);
            acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xf000);
        }
        sumf[row] += float(dh[0]) * ((acc1[0] + acc1[1] / 256.0f) * sc8[0] +
                                     (acc1[2] + acc1[3] / 256.0f) * sc8[1] / 16.0f +
                                     (acc2[0] + acc2[1] / 256.0f) * sc8[4] +
                                     (acc2[2] + acc2[3] / 256.0f) * sc8[5] / 16.0f) -
                     float(dh[1]) * dot(sumy, float4(sc8[2], sc8[3], sc8[6], sc8[7]));
    }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void mlp_gate_up_q4_k_decode(
        device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
        device const uchar* wg [[buffer(2)]], device const uchar* wu [[buffer(3)]],
        constant long& S [[buffer(4)]], ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group [[threadgroup_position_in_grid]]) {
    ushort ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
    uint nb = MLP_K / 256, first_row = (group.x * 2 + simd_group) * 2;
    long s = long(group.y);
    float yl[16], yh[16], gate[2] = {0.0f, 0.0f}, up[2] = {0.0f, 0.0f};
    device const half* src4 = x + s * MLP_K + ix * 256 + 64 * iq + 8 * ir;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = 0.0f;
        for (ushort i = 0; i < 8; ++i) {
            yl[i] = float(src4[i]);           sumy[0] += yl[i];
            yl[i + 8] = float(src4[i + 32]);  sumy[1] += yl[i + 8];
            yh[i] = float(src4[i + 128]);     sumy[2] += yh[i];
            yh[i + 8] = float(src4[i + 160]); sumy[3] += yh[i + 8];
        }
        q4k_accumulate(wg, nb, ib, first_row, iq, ir, yl, yh, sumy, gate);
        q4k_accumulate(wu, nb, ib, first_row, iq, ir, yl, yh, sumy, up);
        src4 += 4 * 256;
    }

    for (ushort row = 0; row < 2; ++row) {
        float gate_sum = simd_sum(gate[row]), up_sum = simd_sum(up[row]);
        if (lane == 0 && s < S && first_row + row < MLP_N) {
            float gh = float(half(gate_sum)), uh = float(half(up_sum));
            y[s * MLP_N + first_row + row] = half((gh / (1.0f + exp(-gh))) * uh);
        }
    }
}

static inline __attribute__((always_inline)) void q5k_accumulate(
        device const uchar* weights, uint nb, uint ib, uint row, ushort q_offset, ushort l0, ushort iq,
        uchar hm1, uchar hm2, uchar hm3, uchar hm4, thread const float* yl, thread const float* yh,
        float4 sumy, thread float& sumf) {
    constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
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
    float4 acc1 = 0.0f, acc2 = 0.0f;
    for (ushort l = 0; l < 8; ++l) {
        uchar h = qh[l];
        acc1[0] += yl[l] * float(q1[l] & 0x0f);
        acc1[1] += yl[l + 8] * float(q1[l] & 0xf0);
        acc1[2] += yh[l] * float(q2[l] & 0x0f);
        acc1[3] += yh[l + 8] * float(q2[l] & 0xf0);
        acc2[0] += h & hm1 ? yl[l] : 0.0f;
        acc2[1] += h & hm2 ? yl[l + 8] : 0.0f;
        acc2[2] += h & hm3 ? yh[l] : 0.0f;
        acc2[3] += h & hm4 ? yh[l + 8] : 0.0f;
    }
    sumf += float(dh[0]) * (sc8[0] * (acc1[0] + 16.0f * acc2[0]) +
                            sc8[1] * (acc1[1] / 16.0f + 16.0f * acc2[1]) +
                            sc8[4] * (acc1[2] + 16.0f * acc2[2]) +
                            sc8[5] * (acc1[3] / 16.0f + 16.0f * acc2[3])) -
            float(dh[1]) * dot(sumy, float4(sc8[2], sc8[3], sc8[6], sc8[7]));
}

[[max_total_threads_per_threadgroup(64)]]
kernel void mlp_gate_up_q5_k_decode(
        device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
        device const uchar* wg [[buffer(2)]], device const uchar* wu [[buffer(3)]],
        constant long& S [[buffer(4)]], ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group [[threadgroup_position_in_grid]]) {
    uint nb = MLP_K / 256, row = group.x * 2 + simd_group;
    ushort tid = lane / 4, ix = lane % 4, iq = tid / 4, ir = tid % 4, l0 = 8 * ir;
    ushort q_offset = 32 * iq + l0, y_offset = 64 * iq + l0;
    uchar hm1 = 1u << (2 * iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
    long s = long(group.y);
    float yl[16], yh[16], gate = 0.0f, up = 0.0f;
    device const half* src1 = x + s * MLP_K + ix * 256 + y_offset;
    for (uint ib = ix; ib < nb; ib += 4) {
        device const half* src2 = src1 + 128;
        float4 sumy = 0.0f;
        for (ushort l = 0; l < 8; ++l) {
            yl[l] = float(src1[l]); yl[l + 8] = float(src1[l + 32]);
            yh[l] = float(src2[l]); yh[l + 8] = float(src2[l + 32]);
            sumy += float4(yl[l], yl[l + 8], yh[l], yh[l + 8]);
        }
        q5k_accumulate(wg, nb, ib, row, q_offset, l0, iq, hm1, hm2, hm3, hm4,
                             yl, yh, sumy, gate);
        q5k_accumulate(wu, nb, ib, row, q_offset, l0, iq, hm1, hm2, hm3, hm4,
                             yl, yh, sumy, up);
        src1 += 4 * 256;
    }
    float gate_sum = simd_sum(gate), up_sum = simd_sum(up);
    if (lane == 0 && s < S && row < MLP_N) {
        float gh = float(half(gate_sum)), uh = float(half(up_sum));
        y[s * MLP_N + row] = half((gh / (1.0f + exp(-gh))) * uh);
    }
}

static inline __attribute__((always_inline)) void iq4xs_accumulate(
        device const uchar* weights, uint nb, uint ib, uint first_row, ushort part, ushort il,
        threadgroup const float* lookup, thread const float4* yl, thread float* sumf) {
    for (ushort row = 0; row < 2; ++row) {
        long o = long(first_row + row) * nb * 136 + ib * 136;
        device const uint* q4 = reinterpret_cast<device const uint*>(weights + o + 8 + 16 * part + 8 * il);
        uint aux[2];
        thread const uchar* q8 = reinterpret_cast<thread const uchar*>(aux);
        float4 acc1 = 0.0f, acc2 = 0.0f;
        aux[0] = q4[0] & 0x0f0f0f0f; aux[1] = (q4[0] >> 4) & 0x0f0f0f0f;
        acc1 += yl[0] * float4(lookup[q8[0]], lookup[q8[1]], lookup[q8[2]], lookup[q8[3]]);
        acc2 += yl[1] * float4(lookup[q8[4]], lookup[q8[5]], lookup[q8[6]], lookup[q8[7]]);
        aux[0] = q4[1] & 0x0f0f0f0f; aux[1] = (q4[1] >> 4) & 0x0f0f0f0f;
        acc1 += yl[2] * float4(lookup[q8[0]], lookup[q8[1]], lookup[q8[2]], lookup[q8[3]]);
        acc2 += yl[3] * float4(lookup[q8[4]], lookup[q8[5]], lookup[q8[6]], lookup[q8[7]]);
        acc1 += acc2;
        ushort sh = *reinterpret_cast<device const ushort*>(weights + o + 2);
        int ls = int((weights[o + 4 + part / 2] >> (4 * (part % 2))) & 15) |
                 int(((sh >> (2 * part)) & 3) << 4);
        float d = float(*reinterpret_cast<device const half*>(weights + o));
        sumf[row] += d * float(ls - 32) * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
    }
}

[[max_total_threads_per_threadgroup(64)]]
kernel void mlp_gate_up_iq4_xs_decode(
        device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
        device const uchar* wg [[buffer(2)]], device const uchar* wu [[buffer(3)]],
        constant long& S [[buffer(4)]], ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        uint3 group [[threadgroup_position_in_grid]]) {
    uint nb = MLP_K / 256, first_row = (group.x * 2 + simd_group) * 2;
    ushort ix = lane / 16, it = lane % 16, part = it / 2, il = it % 2;
    threadgroup float lookup[16];
    if (lane < 16) lookup[lane] = iq4nl[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long s = long(group.y);
    float4 yl[4];
    float gate[2] = {0.0f, 0.0f}, up[2] = {0.0f, 0.0f};
    device const half* src = x + s * MLP_K + ix * 256 + part * 32 + il * 8;
    for (uint ib = ix; ib < nb; ib += 2) {
        yl[0] = float4(float(src[0]), float(src[1]), float(src[2]), float(src[3]));
        yl[1] = float4(float(src[16]), float(src[17]), float(src[18]), float(src[19]));
        yl[2] = float4(float(src[4]), float(src[5]), float(src[6]), float(src[7]));
        yl[3] = float4(float(src[20]), float(src[21]), float(src[22]), float(src[23]));
        iq4xs_accumulate(wg, nb, ib, first_row, part, il, lookup, yl, gate);
        iq4xs_accumulate(wu, nb, ib, first_row, part, il, lookup, yl, up);
        src += 2 * 256;
    }
    for (ushort row = 0; row < 2; ++row) {
        float gate_sum = simd_sum(gate[row]), up_sum = simd_sum(up[row]);
        if (lane == 0 && s < S && first_row + row < MLP_N) {
            float gh = float(half(gate_sum)), uh = float(half(up_sum));
            y[s * MLP_N + first_row + row] = half((gh / (1.0f + exp(-gh))) * uh);
        }
    }
}

static inline half gdn_load_context(device const half* x, device const half* prev, long b, long c, long L,
                                    long pos, bool has_prev) {
    if (has_prev) return pos < 4 ? prev[c * 4 + pos] : x[(b * L + pos - 4) * GDN_C + c];
    return pos < 0 || pos >= L ? half(0.0) : x[(b * L + pos) * GDN_C + c];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void gdn_causal_conv_silu(device half* y [[buffer(0)]], device half* state0 [[buffer(1)]],
                                 device half* state1 [[buffer(2)]], device const half* x [[buffer(3)]],
                                 device const half* w [[buffer(4)]], device const uint* slots [[buffer(5)]],
                                 device const uint* banks [[buffer(6)]], device const uint* valid [[buffer(7)]],
                                 constant long& B [[buffer(8)]], constant long& L [[buffer(9)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    long span = L > 4 ? L : 4, idx = gid.x, b = idx / (GDN_C * span), rem = idx - b * GDN_C * span;
    if (b >= B) return;
    long p = rem / GDN_C, c = rem - p * GDN_C;
    uint slot = slots[b], bank = banks[b]; bool has_prev = valid[b];
    device half* state = (bank ? state0 : state1) + slot * GDN_C * 4;
    device const half* prev = (bank ? state1 : state0) + slot * GDN_C * 4;
    if (p < L) {
        float acc = 0.0f;
        for (long r = 0; r < 4; ++r) acc = fma(float(gdn_load_context(x, prev, b, c, L, has_prev ? p + 1 + r : p + r - 3, has_prev)),
                                               float(w[c * 4 + r]), acc);
        float hv = float(half(acc));
        y[(b * L + p) * GDN_C + c] = half(hv / (1.0f + exp(-hv)));
    }
    if (p < 4) {
        long pos = (has_prev ? L + p : L - 4 + p);
        state[c * 4 + p] = gdn_load_context(x, prev, b, c, L, pos, has_prev);
    }
}

// Speculative verification writes one immutable convolution-state column per query token.
[[max_total_threads_per_threadgroup(256)]]
kernel void gdn_causal_conv_candidates(
        device half* y [[buffer(0)]], device const half* state0 [[buffer(1)]],
        device const half* state1 [[buffer(2)]], device half* candidates [[buffer(3)]],
        device const half* x [[buffer(4)]], device const half* w [[buffer(5)]],
        device const uint* slots [[buffer(6)]], device const uint* banks [[buffer(7)]],
        device const uint* valid [[buffer(8)]], constant long& B [[buffer(9)]], constant long& L [[buffer(10)]],
        uint3 gid [[thread_position_in_grid]]) {
    long idx = gid.x, b = idx / (GDN_C * L), rem = idx - b * GDN_C * L;
    if (b >= B) return;
    long p = rem / GDN_C, c = rem - p * GDN_C; uint slot = slots[b], bank = banks[b];
    bool has_prev = valid[b]; device const half* prev = (bank ? state1 : state0) + slot * GDN_C * 4;
    float acc = 0.0f;
    for (long r = 0; r < 4; ++r) {
        long position = p - 3 + r; half value = position >= 0 ? x[(b * L + position) * GDN_C + c]
            : has_prev ? prev[c * 4 + position + 4] : half(0.0f);
        acc = fma(float(value), float(w[c * 4 + r]), acc);
    }
    float hv = float(half(acc)); y[(b * L + p) * GDN_C + c] = half(hv / (1.0f + exp(-hv)));
    device half* state = candidates + (b * L + p) * GDN_C * 4 + c * 4;
    for (long r = 0; r < 4; ++r) {
        long position = p - 3 + r;
        state[r] = position >= 0 ? x[(b * L + position) * GDN_C + c]
                 : has_prev ? prev[c * 4 + position + 4] : half(0.0f);
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void rmsnorm_gated_128(device half* y [[buffer(0)]], device const half* x [[buffer(1)]],
                              device const half* gate [[buffer(2)]], device const float* w [[buffer(3)]],
                              constant float& eps [[buffer(4)]], uint3 lane3 [[thread_position_in_threadgroup]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]]) {
    uint d = lane3.x, row = group.y;
    threadgroup float scratch[5];
    float xv = float(x[row * GDN_D + d]), ss = simd_sum(xv * xv);
    if (simd_lane == 0) scratch[simd_group] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float s = simd_lane < 4 ? scratch[simd_lane] : 0.0f;
        s = simd_sum(s);
        if (simd_lane == 0) scratch[4] = rsqrt(s / float(GDN_D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gv = float(gate[row * GDN_D + d]);
    y[row * GDN_D + d] = half(xv * scratch[4] * w[d] * (gv / (1.0f + exp(-gv))));
}
