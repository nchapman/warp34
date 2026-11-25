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
// and decode 5 contiguous half2 pairs via hoisted-constant lop3.b32 + HSUB2.

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

__device__ __forceinline__ uint32_t ldg_cg_u32(const uint32_t* ptr) {
    uint32_t val;
    asm("ld.global.cg.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ __half2 u2_pair_to_h2(
    uint32_t shifted,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025
) {
    uint32_t raw;
    // (shifted & mask) | bias  =>  lop3 truth table 0xEA
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(raw) : "r"(shifted), "r"(mask), "r"(bias));
    return __hsub2(*reinterpret_cast<const __half2*>(&raw), c1025);
}

__device__ __forceinline__ void accum_pair_5trit_h2_smem(
    uint32_t q0,
    uint32_t q1,
    const uint32_t* __restrict__ s_lut,
    const __half* __restrict__ x_g,
    int i0,
    int stride,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025,
    __half2& g_acc2
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_g + i0);
    const int s2 = stride >> 1;
    g_acc2 = __hfma2(u2_pair_to_h2(p,      mask, bias, c1025), x2[0],      g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 2, mask, bias, c1025), x2[s2],     g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 4, mask, bias, c1025), x2[2 * s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 6, mask, bias, c1025), x2[3 * s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 8, mask, bias, c1025), x2[4 * s2], g_acc2);
}

__device__ __forceinline__ void accum_pair_4trit_h2_smem(
    uint32_t q0,
    uint32_t q1,
    const uint32_t* __restrict__ s_lut,
    const __half* __restrict__ x_g,
    int i0,
    int stride,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025,
    __half2& g_acc2
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_g + i0);
    const int s2 = stride >> 1;
    g_acc2 = __hfma2(u2_pair_to_h2(p,      mask, bias, c1025), x2[0],      g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 2, mask, bias, c1025), x2[s2],     g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 4, mask, bias, c1025), x2[2 * s2], g_acc2);
    g_acc2 = __hfma2(u2_pair_to_h2(p >> 6, mask, bias, c1025), x2[3 * s2], g_acc2);
}

__device__ __forceinline__ float flush_h2_to_f32(__half2& g_acc2, __half2 h2_zero) {
    const float2 f2 = __half22float2(g_acc2);
    g_acc2 = h2_zero;
    return f2.x + f2.y;
}

// Fast M=1 decode kernel:
// - Bank-conflict-free padded SMEM s_x[4*SPLIT_K][136] loaded via coalesced uint4
// - L2-only streaming weight loads via ld.global.cg.u32
// - Hoisted lop3.b32 constants (eliminates 47 IMAD.MOV.U32 rematerializations per group)
// - Periodic FP32 flushes every 40-48 trits (prevents FP16 overflow & reduces rounding drift)
// - SPLIT_K=2 uses 2 warps/CTA on 1 n_blk (8 rows/CTA -> 512 CTAs for N=4096, doubling warps/SM)
//   with deterministic 32B SMEM cross-warp reduction; SPLIT_K=1 uses 2 n_blks/CTA (16 rows/CTA).
template <int SPLIT_K>
__global__ __launch_bounds__(64) void gemv_5trit_m1_f16_kernel(
    const __half* __restrict__ a,
    const uint32_t* __restrict__ packed_w,
    __half* __restrict__ out,
    int N,
    int K
) {
    constexpr int kGroupsPerStep = 4 * SPLIT_K;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int r = lane & 7;
    const int kg = lane >> 3;

    const int n_blk = (SPLIT_K == 2)
        ? static_cast<int>(blockIdx.x)
        : (static_cast<int>(blockIdx.x) * 2 + warp_id);
    const bool active_warp = (n_blk * 8 < N);
    const int n = n_blk * 8 + r;
    const int G = K >> 7;

    __shared__ uint32_t s_lut[256];
    __shared__ __half s_x[kGroupsPerStep][136];
    __shared__ float s_reduce[8];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 64]  = encode_5trit_lut10(static_cast<uint32_t>(tid + 64));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    s_lut[tid + 192] = encode_5trit_lut10(static_cast<uint32_t>(tid + 192));
    __syncthreads();

    uint32_t mask = 0x00030003u;
    uint32_t bias = 0x64006400u;
    asm("" : "+r"(mask));
    asm("" : "+r"(bias));
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));
    const __half2 h2_zero = __halves2half2(__ushort_as_half(0u), __ushort_as_half(0u));

    const int x_slot = (SPLIT_K == 2) ? (warp_id * 4 + kg) : kg;
    const size_t row_base = static_cast<size_t>(n_blk) * G * 56u + r;

    float acc = 0.0f;
    for (int g_base = 0; g_base < G; g_base += kGroupsPerStep) {
        const int g_valid = min(kGroupsPerStep, G - g_base);
        const int vec_count = g_valid * 16;
        for (int idx = tid; idx < vec_count; idx += 64) {
            const int g_off = idx >> 4;
            const int k_vec = idx & 15;
            const __half* src = a + static_cast<size_t>(g_base + g_off) * 128 + k_vec * 8;
            *reinterpret_cast<uint4*>(&s_x[g_off][k_vec * 8]) = *reinterpret_cast<const uint4*>(src);
        }
        __syncthreads();

        const int g = g_base + x_slot;
        if (active_warp && g < G) {
            const size_t base = row_base + static_cast<size_t>(g) * 56u;
            const __half* x_g = s_x[x_slot];
            float g_sum = 0.0f;
            __half2 g_acc2 = h2_zero;

            {
                const uint32_t w0 = ldg_cg_u32(packed_w + base + 0 * 8);
                const uint32_t w1 = ldg_cg_u32(packed_w + base + 1 * 8);
                accum_pair_5trit_h2_smem((w0 >>  0) & 0xFFu, (w0 >>  8) & 0xFFu, s_lut, x_g,  0, 16, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w0 >> 16) & 0xFFu, (w0 >> 24) & 0xFFu, s_lut, x_g,  2, 16, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w1 >>  0) & 0xFFu, (w1 >>  8) & 0xFFu, s_lut, x_g,  4, 16, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w1 >> 16) & 0xFFu, (w1 >> 24) & 0xFFu, s_lut, x_g,  6, 16, mask, bias, c1025, g_acc2);
                g_sum += flush_h2_to_f32(g_acc2, h2_zero);
            }
            {
                const uint32_t w2 = ldg_cg_u32(packed_w + base + 2 * 8);
                const uint32_t w3 = ldg_cg_u32(packed_w + base + 3 * 8);
                accum_pair_5trit_h2_smem((w2 >>  0) & 0xFFu, (w2 >>  8) & 0xFFu, s_lut, x_g,  8, 16, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w2 >> 16) & 0xFFu, (w2 >> 24) & 0xFFu, s_lut, x_g, 10, 16, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w3 >>  0) & 0xFFu, (w3 >>  8) & 0xFFu, s_lut, x_g, 12, 16, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w3 >> 16) & 0xFFu, (w3 >> 24) & 0xFFu, s_lut, x_g, 14, 16, mask, bias, c1025, g_acc2);
                g_sum += flush_h2_to_f32(g_acc2, h2_zero);
            }
            {
                const uint32_t w4 = ldg_cg_u32(packed_w + base + 4 * 8);
                const uint32_t w5 = ldg_cg_u32(packed_w + base + 5 * 8);
                const uint32_t w6 = ldg_cg_u32(packed_w + base + 6 * 8);
                accum_pair_5trit_h2_smem((w4 >>  0) & 0xFFu, (w4 >>  8) & 0xFFu, s_lut, x_g, 80,  8, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w4 >> 16) & 0xFFu, (w4 >> 24) & 0xFFu, s_lut, x_g, 82,  8, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w5 >>  0) & 0xFFu, (w5 >>  8) & 0xFFu, s_lut, x_g, 84,  8, mask, bias, c1025, g_acc2);
                accum_pair_5trit_h2_smem((w5 >> 16) & 0xFFu, (w5 >> 24) & 0xFFu, s_lut, x_g, 86,  8, mask, bias, c1025, g_acc2);
                accum_pair_4trit_h2_smem((w6 >>  0) & 0xFFu, (w6 >>  8) & 0xFFu, s_lut, x_g, 120, 2, mask, bias, c1025, g_acc2);
                g_sum += flush_h2_to_f32(g_acc2, h2_zero);
                const float scale = __half2float(__ushort_as_half(static_cast<uint16_t>(w6 >> 16)));
                acc = fmaf(scale, g_sum, acc);
            }
        }
        __syncthreads();
    }

    if (!active_warp) return;

    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 8);
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 16);

    if (SPLIT_K == 2) {
        if (warp_id == 1 && kg == 0) {
            s_reduce[r] = acc;
        }
        __syncthreads();
        if (warp_id == 0 && kg == 0) {
            out[n] = __float2half(acc + s_reduce[r]);
        }
    } else {
        if (kg == 0) {
            out[n] = __float2half(acc);
        }
    }
}

template <int M_TILE>
__device__ __forceinline__ void accum_pair_5trit_tile(
    uint32_t q0,
    uint32_t q1,
    const uint32_t* __restrict__ s_lut,
    const __half (* __restrict__ s_x)[4][136],
    int kg,
    int i0,
    int stride,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025,
    __half2 g_acc2[M_TILE]
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2 w0 = u2_pair_to_h2(p,      mask, bias, c1025);
    const __half2 w1 = u2_pair_to_h2(p >> 2, mask, bias, c1025);
    const __half2 w2 = u2_pair_to_h2(p >> 4, mask, bias, c1025);
    const __half2 w3 = u2_pair_to_h2(p >> 6, mask, bias, c1025);
    const __half2 w4 = u2_pair_to_h2(p >> 8, mask, bias, c1025);
    const int s2 = stride >> 1;
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const __half2* x2 = reinterpret_cast<const __half2*>(s_x[m][kg] + i0);
        __half2 a = g_acc2[m];
        a = __hfma2(w0, x2[0],      a);
        a = __hfma2(w1, x2[s2],     a);
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
    const __half (* __restrict__ s_x)[4][136],
    int kg,
    int i0,
    int stride,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025,
    __half2 g_acc2[M_TILE]
) {
    const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
    const __half2 w0 = u2_pair_to_h2(p,      mask, bias, c1025);
    const __half2 w1 = u2_pair_to_h2(p >> 2, mask, bias, c1025);
    const __half2 w2 = u2_pair_to_h2(p >> 4, mask, bias, c1025);
    const __half2 w3 = u2_pair_to_h2(p >> 6, mask, bias, c1025);
    const int s2 = stride >> 1;
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const __half2* x2 = reinterpret_cast<const __half2*>(s_x[m][kg] + i0);
        __half2 a = g_acc2[m];
        a = __hfma2(w0, x2[0],      a);
        a = __hfma2(w1, x2[s2],     a);
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

    uint32_t mask = 0x00030003u;
    uint32_t bias = 0x64006400u;
    asm("" : "+r"(mask));
    asm("" : "+r"(bias));
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));
    const __half2 h2_zero = __halves2half2(__ushort_as_half(0u), __ushort_as_half(0u));

    __shared__ uint32_t s_lut[256];
    // Padded to 136 halves (272 B = 68 banks = 4 mod 32) to eliminate 4-way bank conflicts.
    __shared__ __half s_x[M_TILE][4][136];

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
        const int total_vecs = M_TILE * g_valid * 16;
        for (int idx = tid; idx < total_vecs; idx += 128) {
            const int k_vec = idx & 15;
            const int mg = idx >> 4;
            const int g_off = mg % g_valid;
            const int m = mg / g_valid;
            const int row_m = m_base + m;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (row_m < M) {
                const __half* src = a + static_cast<size_t>(row_m) * K
                                    + static_cast<size_t>(g_base + g_off) * 128
                                    + k_vec * 8;
                v = *reinterpret_cast<const uint4*>(src);
            }
            *reinterpret_cast<uint4*>(&s_x[m][g_off][k_vec * 8]) = v;
        }
        __syncthreads();

        const int g = g_base + kg;
        if (active_warp && g < G) {
            const size_t base = (static_cast<size_t>(n_blk) * G + g) * 56u + r;
            const uint32_t w0 = ldg_cg_u32(packed_w + base + 0 * 8);
            const uint32_t w1 = ldg_cg_u32(packed_w + base + 1 * 8);
            const uint32_t w2 = ldg_cg_u32(packed_w + base + 2 * 8);
            const uint32_t w3 = ldg_cg_u32(packed_w + base + 3 * 8);
            const uint32_t w4 = ldg_cg_u32(packed_w + base + 4 * 8);
            const uint32_t w5 = ldg_cg_u32(packed_w + base + 5 * 8);
            const uint32_t w6 = ldg_cg_u32(packed_w + base + 6 * 8);
            const float scale = __half2float(__ushort_as_half(static_cast<uint16_t>(w6 >> 16)));

            float g_sum[M_TILE];
            __half2 g_acc2[M_TILE];
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                g_sum[m] = 0.0f;
                g_acc2[m] = h2_zero;
            }

            accum_pair_5trit_tile<M_TILE>((w0 >>  0) & 0xFFu, (w0 >>  8) & 0xFFu, s_lut, s_x, kg,  0, 16, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w0 >> 16) & 0xFFu, (w0 >> 24) & 0xFFu, s_lut, s_x, kg,  2, 16, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w1 >>  0) & 0xFFu, (w1 >>  8) & 0xFFu, s_lut, s_x, kg,  4, 16, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w1 >> 16) & 0xFFu, (w1 >> 24) & 0xFFu, s_lut, s_x, kg,  6, 16, mask, bias, c1025, g_acc2);
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                g_sum[m] += flush_h2_to_f32(g_acc2[m], h2_zero);
            }

            accum_pair_5trit_tile<M_TILE>((w2 >>  0) & 0xFFu, (w2 >>  8) & 0xFFu, s_lut, s_x, kg,  8, 16, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w2 >> 16) & 0xFFu, (w2 >> 24) & 0xFFu, s_lut, s_x, kg, 10, 16, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w3 >>  0) & 0xFFu, (w3 >>  8) & 0xFFu, s_lut, s_x, kg, 12, 16, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w3 >> 16) & 0xFFu, (w3 >> 24) & 0xFFu, s_lut, s_x, kg, 14, 16, mask, bias, c1025, g_acc2);
#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                g_sum[m] += flush_h2_to_f32(g_acc2[m], h2_zero);
            }

            accum_pair_5trit_tile<M_TILE>((w4 >>  0) & 0xFFu, (w4 >>  8) & 0xFFu, s_lut, s_x, kg, 80,  8, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w4 >> 16) & 0xFFu, (w4 >> 24) & 0xFFu, s_lut, s_x, kg, 82,  8, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w5 >>  0) & 0xFFu, (w5 >>  8) & 0xFFu, s_lut, s_x, kg, 84,  8, mask, bias, c1025, g_acc2);
            accum_pair_5trit_tile<M_TILE>((w5 >> 16) & 0xFFu, (w5 >> 24) & 0xFFu, s_lut, s_x, kg, 86,  8, mask, bias, c1025, g_acc2);
            accum_pair_4trit_tile<M_TILE>((w6 >>  0) & 0xFFu, (w6 >>  8) & 0xFFu, s_lut, s_x, kg, 120, 2, mask, bias, c1025, g_acc2);

#pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                g_sum[m] += flush_h2_to_f32(g_acc2[m], h2_zero);
                acc[m] = fmaf(scale, g_sum[m], acc[m]);
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

// Fast coalesced strip dequantizer:
// 4 warps/CTA; each warp owns one (n_blk, g) tile (8 rows x 128 halves in padded SMEM).
// Uses single-bank 1 KiB s_lut + hoisted lop3.b32 + paired half2 writes + 100%-coalesced uint4 stores.
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
    const int part = lane >> 3;

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

    uint32_t mask = 0x00030003u;
    uint32_t bias = 0x64006400u;
    asm("" : "+r"(mask));
    asm("" : "+r"(bias));

    const uint32_t w6 = ldg_cg_u32(packed_w + base + 6 * 8);
    const __half scale_h = __ushort_as_half(static_cast<uint16_t>(w6 >> 16));
    const __half2 scale_h2 = __halves2half2(scale_h, scale_h);
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));
    __half* row_s = s_tile[warp_id][r];

    auto write_pair_5 = [&](uint32_t q0, uint32_t q1, int i0, int stride) {
        const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
        __half2* d2 = reinterpret_cast<__half2*>(row_s + i0);
        const int s2 = stride >> 1;
        d2[0]      = __hmul2(u2_pair_to_h2(p,      mask, bias, c1025), scale_h2);
        d2[s2]     = __hmul2(u2_pair_to_h2(p >> 2, mask, bias, c1025), scale_h2);
        d2[2 * s2] = __hmul2(u2_pair_to_h2(p >> 4, mask, bias, c1025), scale_h2);
        d2[3 * s2] = __hmul2(u2_pair_to_h2(p >> 6, mask, bias, c1025), scale_h2);
        d2[4 * s2] = __hmul2(u2_pair_to_h2(p >> 8, mask, bias, c1025), scale_h2);
    };

    if (part == 0) {
        const uint32_t w0 = ldg_cg_u32(packed_w + base + 0 * 8);
        const uint32_t w1 = ldg_cg_u32(packed_w + base + 1 * 8);
        write_pair_5((w0 >>  0) & 0xFFu, (w0 >>  8) & 0xFFu, 0, 16);
        write_pair_5((w0 >> 16) & 0xFFu, (w0 >> 24) & 0xFFu, 2, 16);
        write_pair_5((w1 >>  0) & 0xFFu, (w1 >>  8) & 0xFFu, 4, 16);
        write_pair_5((w1 >> 16) & 0xFFu, (w1 >> 24) & 0xFFu, 6, 16);
    } else if (part == 1) {
        const uint32_t w2 = ldg_cg_u32(packed_w + base + 2 * 8);
        const uint32_t w3 = ldg_cg_u32(packed_w + base + 3 * 8);
        write_pair_5((w2 >>  0) & 0xFFu, (w2 >>  8) & 0xFFu,  8, 16);
        write_pair_5((w2 >> 16) & 0xFFu, (w2 >> 24) & 0xFFu, 10, 16);
        write_pair_5((w3 >>  0) & 0xFFu, (w3 >>  8) & 0xFFu, 12, 16);
        write_pair_5((w3 >> 16) & 0xFFu, (w3 >> 24) & 0xFFu, 14, 16);
    } else if (part == 2) {
        const uint32_t w4 = ldg_cg_u32(packed_w + base + 4 * 8);
        const uint32_t w5 = ldg_cg_u32(packed_w + base + 5 * 8);
        write_pair_5((w4 >>  0) & 0xFFu, (w4 >>  8) & 0xFFu, 80, 8);
        write_pair_5((w4 >> 16) & 0xFFu, (w4 >> 24) & 0xFFu, 82, 8);
        write_pair_5((w5 >>  0) & 0xFFu, (w5 >>  8) & 0xFFu, 84, 8);
        write_pair_5((w5 >> 16) & 0xFFu, (w5 >> 24) & 0xFFu, 86, 8);
    } else {
        const uint32_t p = __byte_perm(s_lut[(w6 >> 0) & 0xFFu], s_lut[(w6 >> 8) & 0xFFu], 0x5410u);
        __half2* d2 = reinterpret_cast<__half2*>(row_s + 120);
        d2[0] = __hmul2(u2_pair_to_h2(p,      mask, bias, c1025), scale_h2);
        d2[1] = __hmul2(u2_pair_to_h2(p >> 2, mask, bias, c1025), scale_h2);
        d2[2] = __hmul2(u2_pair_to_h2(p >> 4, mask, bias, c1025), scale_h2);
        d2[3] = __hmul2(u2_pair_to_h2(p >> 6, mask, bias, c1025), scale_h2);
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

// Coalesced zero-divergence 5-Trit -> Marlin U2B2 (kU2B2, g128) repacker for M > 8 prefill.
// Grid: (G = K/128, n_tiles = N/64), Block: 128 threads (4 warps).
// Outputs both Marlin-tiled 2-bit weights (q = u + 1 in {1,2,3}, 2.0 bpw) and
// Marlin-permuted FP16 scales in a single pass so prefill runs on warp34_marlin_gemm_f16u2
// (unlocking W2A8 INT8 IMMA 337/660 TOPS on sm_86/sm_89 and full-rate HMMA on sm_80).
__global__ __launch_bounds__(128) void repack_5trit_to_marlin_u2_kernel(
    const uint32_t* __restrict__ packed_w,
    uint32_t* __restrict__ dst_marlin,
    uint16_t* __restrict__ dst_scales,
    int N,
    int K
) {
    const int tid = threadIdx.x;
    const int g = static_cast<int>(blockIdx.x);
    const int n_tile_id = static_cast<int>(blockIdx.y);
    const int G = K >> 7;
    const int n_tiles = N >> 6;

    __shared__ uint32_t s_lut[256];
    __shared__ uint8_t s_q[64][132];
    __shared__ uint16_t sh_repack_u2[128];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    __syncthreads();

    // Phase 1: unpack 64 rows x 128 trits into s_q[64][128] with zero intra-warp divergence.
    // Warps 0..1 (half_sel == 0) unpack w0..w3 (seg 0: 80 trits);
    // Warps 2..3 (half_sel == 1) unpack w4..w6 (seg 1+2: 48 trits) + write permuted scale.
    const int p = tid & 63;
    const int half_sel = tid >> 6;
    const int n_blk = n_tile_id * 8 + (p >> 3);
    const int r = p & 7;
    const size_t base = (static_cast<size_t>(n_blk) * G + g) * 56u + r;
    uint8_t* row_q = s_q[p];

    auto write_5_u2 = [&](uint32_t q_byte, int i0, int stride) {
        const uint32_t v = s_lut[q_byte];
        row_q[i0]              = static_cast<uint8_t>(((v >> 0) & 3u) + 1u);
        row_q[i0 + stride]     = static_cast<uint8_t>(((v >> 2) & 3u) + 1u);
        row_q[i0 + 2 * stride] = static_cast<uint8_t>(((v >> 4) & 3u) + 1u);
        row_q[i0 + 3 * stride] = static_cast<uint8_t>(((v >> 6) & 3u) + 1u);
        row_q[i0 + 4 * stride] = static_cast<uint8_t>(((v >> 8) & 3u) + 1u);
    };

    if (half_sel == 0) {
        const uint32_t w0 = ldg_cg_u32(packed_w + base + 0 * 8);
        const uint32_t w1 = ldg_cg_u32(packed_w + base + 1 * 8);
        const uint32_t w2 = ldg_cg_u32(packed_w + base + 2 * 8);
        const uint32_t w3 = ldg_cg_u32(packed_w + base + 3 * 8);
        write_5_u2((w0 >>  0) & 0xFFu,  0, 16);
        write_5_u2((w0 >>  8) & 0xFFu,  1, 16);
        write_5_u2((w0 >> 16) & 0xFFu,  2, 16);
        write_5_u2((w0 >> 24) & 0xFFu,  3, 16);
        write_5_u2((w1 >>  0) & 0xFFu,  4, 16);
        write_5_u2((w1 >>  8) & 0xFFu,  5, 16);
        write_5_u2((w1 >> 16) & 0xFFu,  6, 16);
        write_5_u2((w1 >> 24) & 0xFFu,  7, 16);
        write_5_u2((w2 >>  0) & 0xFFu,  8, 16);
        write_5_u2((w2 >>  8) & 0xFFu,  9, 16);
        write_5_u2((w2 >> 16) & 0xFFu, 10, 16);
        write_5_u2((w2 >> 24) & 0xFFu, 11, 16);
        write_5_u2((w3 >>  0) & 0xFFu, 12, 16);
        write_5_u2((w3 >>  8) & 0xFFu, 13, 16);
        write_5_u2((w3 >> 16) & 0xFFu, 14, 16);
        write_5_u2((w3 >> 24) & 0xFFu, 15, 16);
    } else {
        const uint32_t w4 = ldg_cg_u32(packed_w + base + 4 * 8);
        const uint32_t w5 = ldg_cg_u32(packed_w + base + 5 * 8);
        const uint32_t w6 = ldg_cg_u32(packed_w + base + 6 * 8);
        write_5_u2((w4 >>  0) & 0xFFu, 80, 8);
        write_5_u2((w4 >>  8) & 0xFFu, 81, 8);
        write_5_u2((w4 >> 16) & 0xFFu, 82, 8);
        write_5_u2((w4 >> 24) & 0xFFu, 83, 8);
        write_5_u2((w5 >>  0) & 0xFFu, 84, 8);
        write_5_u2((w5 >>  8) & 0xFFu, 85, 8);
        write_5_u2((w5 >> 16) & 0xFFu, 86, 8);
        write_5_u2((w5 >> 24) & 0xFFu, 87, 8);
        const uint32_t v0 = s_lut[(w6 >> 0) & 0xFFu];
        const uint32_t v1 = s_lut[(w6 >> 8) & 0xFFu];
        row_q[120] = static_cast<uint8_t>(((v0 >> 0) & 3u) + 1u);
        row_q[122] = static_cast<uint8_t>(((v0 >> 2) & 3u) + 1u);
        row_q[124] = static_cast<uint8_t>(((v0 >> 4) & 3u) + 1u);
        row_q[126] = static_cast<uint8_t>(((v0 >> 6) & 3u) + 1u);
        row_q[121] = static_cast<uint8_t>(((v1 >> 0) & 3u) + 1u);
        row_q[123] = static_cast<uint8_t>(((v1 >> 2) & 3u) + 1u);
        row_q[125] = static_cast<uint8_t>(((v1 >> 4) & 3u) + 1u);
        row_q[127] = static_cast<uint8_t>(((v1 >> 6) & 3u) + 1u);
        // Marlin 8x8 scale permutation for group_size=128: out[i*8+j] = src[i+8*j].
        const int dst_s_idx = (p & 7) * 8 + (p >> 3);
        dst_scales[static_cast<size_t>(g) * N + static_cast<size_t>(n_tile_id) * 64 + dst_s_idx] =
            static_cast<uint16_t>(w6 >> 16);
    }
    __syncthreads();

    // Phase 2: repack 8 k-tiles (16x64 each) into Marlin U2B2 layout (exact match to op_marlin.cu).
    const int warp_id = tid >> 5;
    const int th_id = tid & 31;
    const int tc_col = th_id >> 2;
    const int tc_row = (th_id & 3) * 2;
    const int cur_n = warp_id * 16 + tc_col;
    const uint8_t* q_b1 = s_q[cur_n];
    const uint8_t* q_b2 = s_q[cur_n + 8];

#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
        const int k_base = sub * 16 + tc_row;
        const uint32_t v0 = q_b1[k_base + 0];
        const uint32_t v1 = q_b1[k_base + 1];
        const uint32_t v2 = q_b1[k_base + 8];
        const uint32_t v3 = q_b1[k_base + 9];
        const uint32_t v4 = q_b2[k_base + 0];
        const uint32_t v5 = q_b2[k_base + 1];
        const uint32_t v6 = q_b2[k_base + 8];
        const uint32_t v7 = q_b2[k_base + 9];

        // pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7}
        const uint32_t lo_byte = v0 | (v2 << 2) | (v4 << 4) | (v6 << 6);
        const uint32_t hi_byte = v1 | (v3 << 2) | (v5 << 4) | (v7 << 6);
        sh_repack_u2[th_id * 4 + warp_id] = static_cast<uint16_t>(lo_byte | (hi_byte << 8));
        __syncthreads();

        if (tid < 64) {
            const uint16_t even = sh_repack_u2[tid * 2 + 0];
            const uint16_t odd  = sh_repack_u2[tid * 2 + 1];
            const uint32_t w_even = static_cast<uint32_t>(even & 0xFFu) | (static_cast<uint32_t>(even >> 8) << 16);
            const uint32_t w_odd  = static_cast<uint32_t>(odd  & 0xFFu) | (static_cast<uint32_t>(odd  >> 8) << 16);
            const size_t k_tile_id = static_cast<size_t>(g) * 8 + sub;
            const size_t out_offset = (k_tile_id * static_cast<size_t>(n_tiles) + n_tile_id) * 64u;
            dst_marlin[out_offset + tid] = w_even | (w_odd << 8);
        }
        __syncthreads();
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
    if (!ex || !a_ptr || !packed_w || !out_ptr || M < 0 || N <= 0 || K <= 0
        || (N % 8) != 0 || (K % 128) != 0
        || (reinterpret_cast<uintptr_t>(a_ptr) & 15u) != 0
        || (reinterpret_cast<uintptr_t>(packed_w) & 3u) != 0
        || (reinterpret_cast<uintptr_t>(out_ptr) & 1u) != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (M == 0) {
        return 0;
    }
    const __half* a = static_cast<const __half*>(a_ptr);
    __half* out = static_cast<__half*>(out_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    if (M == 1) {
        if (N <= 5376 && K >= 4096) {
            const int n_blocks = N / 8;
            gemv_5trit_m1_f16_kernel<2><<<dim3(n_blocks, 1, 1), 64, 0, stream>>>(a, packed_w, out, N, K);
        } else {
            const int n_blocks = (N + 15) / 16;
            gemv_5trit_m1_f16_kernel<1><<<dim3(n_blocks, 1, 1), 64, 0, stream>>>(a, packed_w, out, N, K);
        }
    } else if (M == 2) {
        const int n_blocks = (N + 31) / 32;
        gemv_5trit_f16_kernel<2><<<dim3(n_blocks, 1, 1), 128, 0, stream>>>(a, packed_w, out, M, N, K);
    } else {
        const int n_blocks = (N + 31) / 32;
        const int m_blocks = (M + 3) / 4;
        if (m_blocks > 65535) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        gemv_5trit_f16_kernel<4><<<dim3(n_blocks, 1, m_blocks), 128, 0, stream>>>(a, packed_w, out, M, N, K);
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
        || (n_start % 8) != 0 || (n_chunk % 8) != 0 || (K % 128) != 0
        || (reinterpret_cast<uintptr_t>(packed_w) & 3u) != 0
        || (reinterpret_cast<uintptr_t>(dst_f16) & 15u) != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
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

extern "C" int warp34_repack_5trit_to_marlin_u2(
    const warp34_exec* ex,
    const uint32_t* packed_w,
    uint32_t* dst_marlin,
    uint16_t* dst_scales,
    int N,
    int K
) {
    if (!ex || !packed_w || !dst_marlin || !dst_scales || N <= 0 || K <= 0
        || (N % 64) != 0 || (K % 128) != 0
        || (reinterpret_cast<uintptr_t>(packed_w) & 3u) != 0
        || (reinterpret_cast<uintptr_t>(dst_marlin) & 15u) != 0
        || (reinterpret_cast<uintptr_t>(dst_scales) & 15u) != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int G = K >> 7;
    const int n_tiles = N >> 6;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    repack_5trit_to_marlin_u2_kernel<<<dim3(G, n_tiles, 1), 128, 0, stream>>>(
        packed_w,
        dst_marlin,
        dst_scales,
        N,
        K
    );
    return static_cast<int>(cudaGetLastError());
}
