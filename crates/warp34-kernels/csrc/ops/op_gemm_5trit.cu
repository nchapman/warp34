// Native PTQ1_0 5-Trit (1.75 bpw) W1.75A16 GEMV / GEMM & Prefill Dequant kernels
// for PrismML Bonsai 2/2.5.
//
// Packed GPU weight layout: [N / 8, K / 128, 7, 8] of uint32_t (28 bytes per 128 weights,
// exact 1.75 bpw, zero padding). For each (n_blk, g, r) with n = n_blk * 8 + r:
//   w[0..5] : 24 bytes of 5-trit codes (seg 0: 16 bytes @ stride 16 = 80 trits;
//                                       seg 1:  8 bytes @ stride  8 = 40 trits)
//   w[6]    : low 16 bits = 2 bytes of 4-trit codes (seg 2: 2 bytes @ stride 2 = 8 trits)
//             high 16 bits = FP16 group scale
//
// Each trit byte q in [0, 242] decodes to 5 ternary weights t_n in {-1, 0, +1}:
//   u_n = (((uint8_t)(q * 3^n) * 3u) >> 8u)  in {0, 1, 2},  t_n = u_n - 1.
// A 1 KiB shared-memory LUT stores u0..u4 in bits 0..9 of a uint32_t (1 bank per access).
// Paired bytes (q0, q1) merge via 1-instruction __byte_perm(lut[q0], lut[q1], 0x5410)
// and decode 5 contiguous half2 pairs via Marlin-style 0x64006400 LOP3 + HSUB2.

#include "../warp34_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t encode_5trit_lut10(uint32_t q) {
    constexpr uint32_t kPow3[5] = {1u, 3u, 9u, 27u, 81u};
    uint32_t packed = 0u;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint32_t u = (((q * kPow3[i]) & 0xFFu) * 3u) >> 8u;
        packed |= (u << (2 * i));
    }
    return packed;
}

__device__ __forceinline__ __half2 u2_pair_to_h2(uint32_t shifted, __half2 c1025) {
    const uint32_t raw = (shifted & 0x00030003u) | 0x64006400u;
    return __hsub2(*reinterpret_cast<const __half2*>(&raw), c1025);
}

__device__ __forceinline__ void accum_pair_5trit_h2(
    uint32_t q0,
    uint32_t q1,
    const uint32_t* __restrict__ s_lut,
    const __half* __restrict__ x_g,
    int i0,
    int stride,
    __half2 c1025,
    __half2& g_acc2
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_g + i0);
    const int s2 = stride >> 1;
    g_acc2 = __hfma2(u2_pair_to_h2(p, c1025), x2[0], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 2, c1025), x2[s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 4, c1025), x2[2 * s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 6, c1025), x2[3 * s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 8, c1025), x2[4 * s2], g_acc2);
}

__device__ __forceinline__ void accum_pair_4trit_h2(
    uint32_t q0,
    uint32_t q1,
    const uint32_t* __restrict__ s_lut,
    const __half* __restrict__ x_g,
    int i0,
    int stride,
    __half2 c1025,
    __half2& g_acc2
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_g + i0);
    const int s2 = stride >> 1;
    g_acc2 = __hfma2(u2_pair_to_h2(p, c1025), x2[0], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 2, c1025), x2[s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 4, c1025), x2[2 * s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 6, c1025), x2[3 * s2], g_acc2);
}

// Fast M=1 decode kernel: 2 warps/CTA (16 rows/CTA -> 256 CTAs for N=4096),
// 1 KiB single-bank s_lut, coalesced uint4 -> SMEM s_x, 1-instruction __byte_perm
// paired half2 decode + contiguous LDS.32 broadcast loads + half2 FMA.
__global__ __launch_bounds__(64) void gemv_5trit_m1_f16_kernel(
    const __half* __restrict__ a,
    const uint32_t* __restrict__ packed_w,
    __half* __restrict__ out,
    int N,
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int r = lane & 7;
    const int kg = lane >> 3;

    const int n_blk = static_cast<int>(blockIdx.x) * 2 + warp_id;
    const bool active_warp = (n_blk * 8 < N);
    const int n = n_blk * 8 + r;
    const int G = K >> 7;
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));
    const __half2 h2_zero = __halves2half2(__ushort_as_half(0u), __ushort_as_half(0u));

    __shared__ uint32_t s_lut[256];
    __shared__ __half s_x[4][128];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 64]  = encode_5trit_lut10(static_cast<uint32_t>(tid + 64));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    s_lut[tid + 192] = encode_5trit_lut10(static_cast<uint32_t>(tid + 192));
    __syncthreads();

    float acc = 0.0f;
    for (int g_base = 0; g_base < G; g_base += 4) {
        const int g_valid = min(4, G - g_base);
        const int vec_count = g_valid * 16;
        if (tid < vec_count) {
            const int g_off = tid >> 4;
            const int k_vec = tid & 15;
            const __half* src = a + static_cast<size_t>(g_base + g_off) * 128 + k_vec * 8;
            *reinterpret_cast<uint4*>(&s_x[g_off][k_vec * 8]) = *reinterpret_cast<const uint4*>(src);
        }
        __syncthreads();

        const int g = g_base + kg;
        if (active_warp && g < G) {
            const size_t base = (static_cast<size_t>(n_blk) * G + g) * 56u + r;
            const uint32_t w0 = __ldg(packed_w + base + 0 * 8);
            const uint32_t w1 = __ldg(packed_w + base + 1 * 8);
            const uint32_t w2 = __ldg(packed_w + base + 2 * 8);
            const uint32_t w3 = __ldg(packed_w + base + 3 * 8);
            const uint32_t w4 = __ldg(packed_w + base + 4 * 8);
            const uint32_t w5 = __ldg(packed_w + base + 5 * 8);
            const uint32_t w6 = __ldg(packed_w + base + 6 * 8);
            const float scale = __half2float(__ushort_as_half(static_cast<uint16_t>(w6 >> 16)));
            const __half* x_g = s_x[kg];

            __half2 g_acc2 = h2_zero;

            // Segment 0: 8 byte-pairs (bytes 0..15) -> 80 trits at stride 16.
            accum_pair_5trit_h2((w0 >>  0) & 0xFFu, (w0 >>  8) & 0xFFu, s_lut, x_g,  0, 16, c1025, g_acc2);
            accum_pair_5trit_h2((w0 >> 16) & 0xFFu, (w0 >> 24) & 0xFFu, s_lut, x_g,  2, 16, c1025, g_acc2);
            accum_pair_5trit_h2((w1 >>  0) & 0xFFu, (w1 >>  8) & 0xFFu, s_lut, x_g,  4, 16, c1025, g_acc2);
            accum_pair_5trit_h2((w1 >> 16) & 0xFFu, (w1 >> 24) & 0xFFu, s_lut, x_g,  6, 16, c1025, g_acc2);
            accum_pair_5trit_h2((w2 >>  0) & 0xFFu, (w2 >>  8) & 0xFFu, s_lut, x_g,  8, 16, c1025, g_acc2);
            accum_pair_5trit_h2((w2 >> 16) & 0xFFu, (w2 >> 24) & 0xFFu, s_lut, x_g, 10, 16, c1025, g_acc2);
            accum_pair_5trit_h2((w3 >>  0) & 0xFFu, (w3 >>  8) & 0xFFu, s_lut, x_g, 12, 16, c1025, g_acc2);
            accum_pair_5trit_h2((w3 >> 16) & 0xFFu, (w3 >> 24) & 0xFFu, s_lut, x_g, 14, 16, c1025, g_acc2);

            // Segment 1: 4 byte-pairs (bytes 16..23) -> 40 trits at stride 8 (offset 80).
            accum_pair_5trit_h2((w4 >>  0) & 0xFFu, (w4 >>  8) & 0xFFu, s_lut, x_g, 80,  8, c1025, g_acc2);
            accum_pair_5trit_h2((w4 >> 16) & 0xFFu, (w4 >> 24) & 0xFFu, s_lut, x_g, 82,  8, c1025, g_acc2);
            accum_pair_5trit_h2((w5 >>  0) & 0xFFu, (w5 >>  8) & 0xFFu, s_lut, x_g, 84,  8, c1025, g_acc2);
            accum_pair_5trit_h2((w5 >> 16) & 0xFFu, (w5 >> 24) & 0xFFu, s_lut, x_g, 86,  8, c1025, g_acc2);

            // Segment 2: 1 byte-pair (bytes 24..25) -> 8 trits at stride 2 (offset 120).
            accum_pair_4trit_h2((w6 >>  0) & 0xFFu, (w6 >>  8) & 0xFFu, s_lut, x_g, 120, 2, c1025, g_acc2);

            const float2 f2 = __half22float2(g_acc2);
            acc = fmaf(scale, f2.x + f2.y, acc);
        }
        __syncthreads();
    }

    if (!active_warp) return;

    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 8);
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 16);
    if (kg == 0) {
        out[n] = __float2half(acc);
    }
}

template <int M_TILE>
__device__ __forceinline__ void accum_pair_5trit_tile(
    uint32_t q0,
    uint32_t q1,
    const uint32_t* __restrict__ s_lut,
    const __half (* __restrict__ s_x)[4][128],
    int kg,
    int i0,
    int stride,
    __half2 c1025,
    __half2 g_acc2[M_TILE]
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2 w0 = u2_pair_to_h2(p, c1025);
    const __half2 w1 = u2_pair_to_h2(p >> 2, c1025);
    const __half2 w2 = u2_pair_to_h2(p >> 4, c1025);
    const __half2 w3 = u2_pair_to_h2(p >> 6, c1025);
    const __half2 w4 = u2_pair_to_h2(p >> 8, c1025);
    const int s2 = stride >> 1;
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const __half2* x2 = reinterpret_cast<const __half2*>(s_x[m][kg] + i0);
        __half2 a = g_acc2[m];
        a = __hfma2(w0, x2[0], a);
        a = __hfma2(w1, x2[s2], a);
        a = __hfma2(w2, x2[2 * s2], a);
        a = __hfma2(w3, x2[3 * s2], a);
        a = __hfma2(w4, x2[4 * s2], a);
        g_acc2[m] = a;
    }
}

template <int M_TILE>
__device__ __forceinline__ void accum_pair_4trit_tile(
    uint32_t q0,
    uint32_t q1,
    const uint32_t* __restrict__ s_lut,
    const __half (* __restrict__ s_x)[4][128],
    int kg,
    int i0,
    int stride,
    __half2 c1025,
    __half2 g_acc2[M_TILE]
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2 w0 = u2_pair_to_h2(p, c1025);
    const __half2 w1 = u2_pair_to_h2(p >> 2, c1025);
    const __half2 w2 = u2_pair_to_h2(p >> 4, c1025);
    const __half2 w3 = u2_pair_to_h2(p >> 6, c1025);
    const int s2 = stride >> 1;
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const __half2* x2 = reinterpret_cast<const __half2*>(s_x[m][kg] + i0);
        __half2 a = g_acc2[m];
        a = __hfma2(w0, x2[0], a);
        a = __hfma2(w1, x2[s2], a);
        a = __hfma2(w2, x2[2 * s2], a);
        a = __hfma2(w3, x2[3 * s2], a);
        g_acc2[m] = a;
    }
}

template <int M_TILE>
__global__ __launch_bounds__(128) void gemv_5trit_f16_kernel(
    const __half* __restrict__ a,
    const uint32_t* __restrict__ packed_w,
    __half* __restrict__ out,
    int M,
    int N,
    int K
) {
    constexpr int kWarps = 4;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int r = lane & 7;
    const int kg = lane >> 3;

    const int n_blk = static_cast<int>(blockIdx.x) * kWarps + warp_id;
    const int n = n_blk * 8 + r;
    const bool active_warp = (n_blk * 8 < N);
    const int m_base = static_cast<int>(blockIdx.z) * M_TILE;
    const int G = K >> 7;
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));
    const __half2 h2_zero = __halves2half2(__ushort_as_half(0u), __ushort_as_half(0u));

    __shared__ uint32_t s_lut[256];
    __shared__ __half s_x[M_TILE][4][128];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    __syncthreads();

    float acc[M_TILE];
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        acc[m] = 0.0f;
    }

    for (int g_base = 0; g_base < G; g_base += 4) {
        const int g_valid = min(4, G - g_base);
        const int vec_count = g_valid * 16;
#pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int row_m = m_base + m;
            if (tid < vec_count) {
                const int g_off = tid >> 4;
                const int k_vec = tid & 15;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (row_m < M) {
                    const __half* src = a + static_cast<size_t>(row_m) * K
                                        + static_cast<size_t>(g_base + g_off) * 128
                                        + k_vec * 8;
                    v = *reinterpret_cast<const uint4*>(src);
                }
                *reinterpret_cast<uint4*>(&s_x[m][g_off][k_vec * 8]) = v;
            }
        }
        __syncthreads();

        const int g = g_base + kg;
        if (active_warp && g < G) {
            const size_t base = (static_cast<size_t>(n_blk) * G + g) * 56u + r;
            const uint32_t w0 = __ldg(packed_w + base + 0 * 8);
            const uint32_t w1 = __ldg(packed_w + base + 1 * 8);
            const uint32_t w2 = __ldg(packed_w + base + 2 * 8);
            const uint32_t w3 = __ldg(packed_w + base + 3 * 8);
            const uint32_t w4 = __ldg(packed_w + base + 4 * 8);
            const uint32_t w5 = __ldg(packed_w + base + 5 * 8);
            const uint32_t w6 = __ldg(packed_w + base + 6 * 8);
            const float scale = __half2float(__ushort_as_half(static_cast<uint16_t>(w6 >> 16)));

            __half2 g_acc2[M_TILE];
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                g_acc2[m] = h2_zero;
            }

            accum_pair_5trit_tile<M_TILE>((w0 >>  0) & 0xFFu, (w0 >>  8) & 0xFFu, s_lut, s_x, kg,  0, 16, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w0 >> 16) & 0xFFu, (w0 >> 24) & 0xFFu, s_lut, s_x, kg,  2, 16, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w1 >>  0) & 0xFFu, (w1 >>  8) & 0xFFu, s_lut, s_x, kg,  4, 16, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w1 >> 16) & 0xFFu, (w1 >> 24) & 0xFFu, s_lut, s_x, kg,  6, 16, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w2 >>  0) & 0xFFu, (w2 >>  8) & 0xFFu, s_lut, s_x, kg,  8, 16, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w2 >> 16) & 0xFFu, (w2 >> 24) & 0xFFu, s_lut, s_x, kg, 10, 16, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w3 >>  0) & 0xFFu, (w3 >>  8) & 0xFFu, s_lut, s_x, kg, 12, 16, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w3 >> 16) & 0xFFu, (w3 >> 24) & 0xFFu, s_lut, s_x, kg, 14, 16, c1025, g_acc2);

            accum_pair_5trit_tile<M_TILE>((w4 >>  0) & 0xFFu, (w4 >>  8) & 0xFFu, s_lut, s_x, kg, 80,  8, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w4 >> 16) & 0xFFu, (w4 >> 24) & 0xFFu, s_lut, s_x, kg, 82,  8, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w5 >>  0) & 0xFFu, (w5 >>  8) & 0xFFu, s_lut, s_x, kg, 84,  8, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w5 >> 16) & 0xFFu, (w5 >> 24) & 0xFFu, s_lut, s_x, kg, 86,  8, c1025, g_acc2);

            accum_pair_4trit_tile<M_TILE>((w6 >>  0) & 0xFFu, (w6 >>  8) & 0xFFu, s_lut, s_x, kg, 120, 2, c1025, g_acc2);

#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                const float2 f2 = __half22float2(g_acc2[m]);
                acc[m] = fmaf(scale, f2.x + f2.y, acc[m]);
            }
        }
        __syncthreads();
    }

    if (!active_warp) return;

#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        acc[m] += __shfl_xor_sync(0xFFFFFFFFu, acc[m], 8);
        acc[m] += __shfl_xor_sync(0xFFFFFFFFu, acc[m], 16);
    }

    if (kg == 0) {
#pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            const int row_m = m_base + m;
            if (row_m < M) {
                out[static_cast<size_t>(row_m) * N + n] = __float2half(acc[m]);
            }
        }
    }
}

// Fast coalesced strip dequantizer for prefill (M > 8):
// 4 warps/CTA; each warp owns one (n_blk, g) tile (8 rows x 128 halves = 2 KiB in SMEM).
// Uses single-bank 1 KiB s_lut + paired half2 writes + 100%-coalesced uint4 global stores.
__global__ __launch_bounds__(128) void dequant_5trit_to_f16_kernel(
    const uint32_t* __restrict__ packed_w,
    __half* __restrict__ dst_f16,
    int n_start,
    int n_chunk,
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int r = lane & 7;
    const int part = lane >> 3; // 0..3

    __shared__ uint32_t s_lut[256];
    __shared__ __half s_tile[4][8][136];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    __syncthreads();

    const int G = K >> 7;
    const int n_blk_start = n_start >> 3;
    const int n_blks_chunk = n_chunk >> 3;
    const int total_tiles = n_blks_chunk * G;
    const int tile_idx = static_cast<int>(blockIdx.x) * 4 + warp_id;
    if (tile_idx >= total_tiles) return;

    const int local_n_blk = tile_idx / G;
    const int g = tile_idx - local_n_blk * G;
    const int n_blk = n_blk_start + local_n_blk;
    const size_t base = (static_cast<size_t>(n_blk) * G + g) * 56u + r;

    const uint32_t w6 = __ldg(packed_w + base + 6 * 8);
    const __half scale_h = __ushort_as_half(static_cast<uint16_t>(w6 >> 16));
    const __half2 scale_h2 = __halves2half2(scale_h, scale_h);
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));
    __half* row_s = s_tile[warp_id][r];

    auto write_pair_5 = [&](uint32_t q0, uint32_t q1, int i0, int stride) {
        const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
        __half2* d2 = reinterpret_cast<__half2*>(row_s + i0);
        const int s2 = stride >> 1;
        d2[0]      = __hmul2(u2_pair_to_h2(p,      c1025), scale_h2);
        d2[s2]     = __hmul2(u2_pair_to_h2(p >> 2, c1025), scale_h2);
        d2[2 * s2] = __hmul2(u2_pair_to_h2(p >> 4, c1025), scale_h2);
        d2[3 * s2] = __hmul2(u2_pair_to_h2(p >> 6, c1025), scale_h2);
        d2[4 * s2] = __hmul2(u2_pair_to_h2(p >> 8, c1025), scale_h2);
    };

    if (part == 0) {
        const uint32_t w0 = __ldg(packed_w + base + 0 * 8);
        const uint32_t w1 = __ldg(packed_w + base + 1 * 8);
        write_pair_5((w0 >>  0) & 0xFFu, (w0 >>  8) & 0xFFu, 0, 16);
        write_pair_5((w0 >> 16) & 0xFFu, (w0 >> 24) & 0xFFu, 2, 16);
        write_pair_5((w1 >>  0) & 0xFFu, (w1 >>  8) & 0xFFu, 4, 16);
        write_pair_5((w1 >> 16) & 0xFFu, (w1 >> 24) & 0xFFu, 6, 16);
    } else if (part == 1) {
        const uint32_t w2 = __ldg(packed_w + base + 2 * 8);
        const uint32_t w3 = __ldg(packed_w + base + 3 * 8);
        write_pair_5((w2 >>  0) & 0xFFu, (w2 >>  8) & 0xFFu,  8, 16);
        write_pair_5((w2 >> 16) & 0xFFu, (w2 >> 24) & 0xFFu, 10, 16);
        write_pair_5((w3 >>  0) & 0xFFu, (w3 >>  8) & 0xFFu, 12, 16);
        write_pair_5((w3 >> 16) & 0xFFu, (w3 >> 24) & 0xFFu, 14, 16);
    } else if (part == 2) {
        const uint32_t w4 = __ldg(packed_w + base + 4 * 8);
        const uint32_t w5 = __ldg(packed_w + base + 5 * 8);
        write_pair_5((w4 >>  0) & 0xFFu, (w4 >>  8) & 0xFFu, 80, 8);
        write_pair_5((w4 >> 16) & 0xFFu, (w4 >> 24) & 0xFFu, 82, 8);
        write_pair_5((w5 >>  0) & 0xFFu, (w5 >>  8) & 0xFFu, 84, 8);
        write_pair_5((w5 >> 16) & 0xFFu, (w5 >> 24) & 0xFFu, 86, 8);
    } else {
        const uint32_t p = __byte_perm(s_lut[(w6 >> 0) & 0xFFu], s_lut[(w6 >> 8) & 0xFFu], 0x5410u);
        __half2* d2 = reinterpret_cast<__half2*>(row_s + 120);
        d2[0] = __hmul2(u2_pair_to_h2(p,      c1025), scale_h2);
        d2[1] = __hmul2(u2_pair_to_h2(p >> 2, c1025), scale_h2);
        d2[2] = __hmul2(u2_pair_to_h2(p >> 4, c1025), scale_h2);
        d2[3] = __hmul2(u2_pair_to_h2(p >> 6, c1025), scale_h2);
    }
    __syncwarp();

    const int k_vec = lane & 15;
    const int r_base = lane >> 4;
#pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        const int row_r = iter * 2 + r_base;
        __half* dst = dst_f16 + static_cast<size_t>(local_n_blk * 8 + row_r) * K
                    + static_cast<size_t>(g) * 128 + k_vec * 8;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(&s_tile[warp_id][row_r][k_vec * 8]);
    }
}

} // namespace

extern "C" int warp34_gemm_5trit_f16(
    const warp34_exec* ex,
    const void* a_ptr,
    const uint32_t* packed_w,
    void* out_ptr,
    int M,
    int N,
    int K
) {
    if (!ex || !a_ptr || !packed_w || !out_ptr || M < 0 || N <= 0 || K <= 0 || (N % 8) != 0 || (K % 128) != 0) {
        return -1;
    }
    if (M == 0) {
        return 0;
    }
    const __half* a = static_cast<const __half*>(a_ptr);
    __half* out = static_cast<__half*>(out_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    if (M == 1) {
        const int n_blocks = (N + 15) / 16;
        gemv_5trit_m1_f16_kernel<<<dim3(n_blocks, 1, 1), 64, 0, stream>>>(a, packed_w, out, N, K);
    } else if (M == 2) {
        const int n_blocks = (N + 31) / 32;
        gemv_5trit_f16_kernel<2><<<dim3(n_blocks, 1, 1), 128, 0, stream>>>(a, packed_w, out, M, N, K);
    } else if (M <= 4) {
        const int n_blocks = (N + 31) / 32;
        gemv_5trit_f16_kernel<4><<<dim3(n_blocks, 1, 1), 128, 0, stream>>>(a, packed_w, out, M, N, K);
    } else {
        const int n_blocks = (N + 31) / 32;
        const int m_blocks = (M + 7) / 8;
        gemv_5trit_f16_kernel<8><<<dim3(n_blocks, 1, m_blocks), 128, 0, stream>>>(a, packed_w, out, M, N, K);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int warp34_dequant_5trit_to_f16(
    const warp34_exec* ex,
    const uint32_t* packed_w,
    void* dst_f16,
    int n_start,
    int n_chunk,
    int K
) {
    if (!ex || !packed_w || !dst_f16 || n_start < 0 || n_chunk <= 0 || K <= 0
        || (n_start % 8) != 0 || (n_chunk % 8) != 0 || (K % 128) != 0) {
        return -1;
    }
    const int G = K >> 7;
    const int total_tiles = (n_chunk >> 3) * G;
    const int blocks = (total_tiles + 3) / 4;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    dequant_5trit_to_f16_kernel<<<dim3(blocks, 1, 1), 128, 0, stream>>>(
        packed_w,
        static_cast<__half*>(dst_f16),
        n_start,
        n_chunk,
        K
    );
    return static_cast<int>(cudaGetLastError());
}
