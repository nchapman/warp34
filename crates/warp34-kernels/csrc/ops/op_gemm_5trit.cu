// Native PTQ1_0 5-Trit (1.75 bpw) W1.75A16 GEMV / Tensor-Core GEMM & Prefill Repack kernels
// for PrismML Bonsai 2/2.5.
//
// Packed GPU weight layout: [N / 8, K / 128, 7, 8] of uint32_t (28 bytes per 128 weights,
// exact 1.75 bpw, zero padding). For each (n_blk, g, r) with n = n_blk * 8 + r:
//   w[0..5] : 24 bytes of 5-trit codes (seg 0: 16 bytes @ stride 16 = 80 trits;
//                                       seg 1:  8 bytes @ stride  8 = 40 trits)
//   w[6]    : low 16 bits = 2 bytes of 4-trit codes (seg 2: 2 bytes @ stride 2 = 8 trits)
//             high 16 bits = FP16 group scale
//
// Each trit byte q in [0, 255] (243 distinct fixed-point base-3 codes) decodes to 5 ternary
// weights t_n in {-1, 0, +1}:
//   u_n = (((uint8_t)(q * 3^n) * 3u) >> 8u)  in {0, 1, 2},  t_n = u_n - 1.
// A 1 KiB shared-memory LUT stores u0..u4 in bits 0..9 of a uint32_t (1 bank per access).
// Paired bytes (q0, q1) merge via 1-instruction __byte_perm(lut[q0], lut[q1], 0x5410)
// and decode 5 contiguous half2 pairs via hoisted-constant lop3.b32 + HSUB2.
//
// Execution regimes:
//   - M == 1: `gemv_5trit_m1_f16_kernel<SPLIT_K>` (SPLIT_K in {1, 2, 4}) with periodic 20/24-term
//     FP32 flushes (exact/overflow-free for |a| <= 2729) and deterministic in-CTA SMEM reduction.
//   - M >= 2: `tc_gemm_5trit_f16_kernel<SPLIT_K>` (SPLIT_K in {1, 2, 4}, M_TILE = 16) feeding
//     `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` directly from coalesced register-prefetched
//     1.75 bpw `packed_w` (zero weight SMEM, zero global scratch, full FP32 accumulation).
//   - M > 16 prefill (when N % 64 == 0 and scratch fits): `repack_5trit_to_marlin_u2_kernel` +
//     `warp34_marlin_gemm_f16u2` (unlocking W2A8 INT8 IMMA on sm_86/sm_89).

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
// - Periodic FP32 flushes every 20-24 half2 pairs (prevents FP16 overflow & reduces drift)
// - SPLIT_K=1: 2 warps/CTA, 2 n_blks/CTA (16 rows/CTA) for wide N
// - SPLIT_K=2: 2 warps/CTA, 1 n_blk/CTA (8 rows/CTA) for N<=5376, K>=1024
// - SPLIT_K=4: 4 warps/CTA, 1 n_blk/CTA (8 rows/CTA) for N<=5376, K>=6144 (doubles warps/SM
//   and halves K-loop barrier drains on down_proj / 27B o_proj & out_proj)
template <int SPLIT_K>
__global__ __launch_bounds__((SPLIT_K == 4) ? 128 : 64) void gemv_5trit_m1_f16_kernel(
    const __half* __restrict__ a,
    const uint32_t* __restrict__ packed_w,
    __half* __restrict__ out,
    int N,
    int K
) {
    static_assert(SPLIT_K == 1 || SPLIT_K == 2 || SPLIT_K == 4, "unsupported SPLIT_K");
    constexpr int kThreads = (SPLIT_K == 4) ? 128 : 64;
    constexpr int kGroupsPerStep = 4 * SPLIT_K;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int r = lane & 7;
    const int kg = lane >> 3;

    const int n_blk = (SPLIT_K == 1)
        ? (static_cast<int>(blockIdx.x) * 2 + warp_id)
        : static_cast<int>(blockIdx.x);
    const bool active_warp = (n_blk * 8 < N);
    const int n = n_blk * 8 + r;
    const int G = K >> 7;

    __shared__ uint32_t s_lut[256];
    __shared__ __half s_x[kGroupsPerStep][136];
    __shared__ float s_reduce[(SPLIT_K > 1) ? (SPLIT_K - 1) : 1][8];

#pragma unroll
    for (int i = tid; i < 256; i += kThreads) {
        s_lut[i] = encode_5trit_lut10(static_cast<uint32_t>(i));
    }
    __syncthreads();

    uint32_t mask = 0x00030003u;
    uint32_t bias = 0x64006400u;
    asm("" : "+r"(mask));
    asm("" : "+r"(bias));
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));
    const __half2 h2_zero = __halves2half2(__ushort_as_half(0u), __ushort_as_half(0u));

    const int x_slot = (SPLIT_K == 1) ? kg : (warp_id * 4 + kg);
    const size_t row_base = static_cast<size_t>(n_blk) * G * 56u + r;

    float acc = 0.0f;
    for (int g_base = 0; g_base < G; g_base += kGroupsPerStep) {
        const int g_valid = min(kGroupsPerStep, G - g_base);
        const int vec_count = g_valid * 16;
        for (int idx = tid; idx < vec_count; idx += kThreads) {
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

    if (SPLIT_K == 4) {
        if (warp_id > 0 && kg == 0) {
            s_reduce[warp_id - 1][r] = acc;
        }
        __syncthreads();
        if (warp_id == 0 && kg == 0) {
            out[n] = __float2half((acc + s_reduce[0][r]) + (s_reduce[1][r] + s_reduce[2][r]));
        }
    } else if (SPLIT_K == 2) {
        if (warp_id == 1 && kg == 0) {
            s_reduce[0][r] = acc;
        }
        __syncthreads();
        if (warp_id == 0 && kg == 0) {
            out[n] = __float2half(acc + s_reduce[0][r]);
        }
    } else {
        if (kg == 0) {
            out[n] = __float2half(acc);
        }
    }
}

// Per-warp 5-trit (8 rows x 128 cols) unpack into padded FP16 SMEM `dst_tile[8][136]`.
// Lanes 0..23 (`part < 3`) execute in lockstep across w0..w5; lanes 24..31 (`part == 3`)
// unpack the trailing 8 trits from w6.
__device__ __forceinline__ void unpack_5trit_tile_to_f16_smem(
    const uint32_t* __restrict__ packed_w,
    size_t base,
    const uint32_t* __restrict__ s_lut,
    __half (* __restrict__ dst_tile)[136],
    int r,
    int part,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025
) {
    const uint32_t w6 = ldg_cg_u32(packed_w + base + 6 * 8);
    const __half scale_h = __ushort_as_half(static_cast<uint16_t>(w6 >> 16));
    const __half2 scale_h2 = __halves2half2(scale_h, scale_h);
    __half* row_s = dst_tile[r];

    if (part < 3) {
        const uint32_t wa = ldg_cg_u32(packed_w + base + (part * 2 + 0) * 8);
        const uint32_t wb = ldg_cg_u32(packed_w + base + (part * 2 + 1) * 8);
        const int i0_base = (part == 0) ? 0 : ((part == 1) ? 8 : 80);
        const int stride = (part < 2) ? 16 : 8;
        const int s2 = stride >> 1;
        auto write_5 = [&](uint32_t q0, uint32_t q1, int off) {
            const uint32_t p = __byte_perm(s_lut[q0], s_lut[q1], 0x5410u);
            __half2* d2 = reinterpret_cast<__half2*>(row_s + i0_base + off);
            d2[0]      = __hmul2(u2_pair_to_h2(p,      mask, bias, c1025), scale_h2);
            d2[s2]     = __hmul2(u2_pair_to_h2(p >> 2, mask, bias, c1025), scale_h2);
            d2[2 * s2] = __hmul2(u2_pair_to_h2(p >> 4, mask, bias, c1025), scale_h2);
            d2[3 * s2] = __hmul2(u2_pair_to_h2(p >> 6, mask, bias, c1025), scale_h2);
            d2[4 * s2] = __hmul2(u2_pair_to_h2(p >> 8, mask, bias, c1025), scale_h2);
        };
        write_5((wa >>  0) & 0xFFu, (wa >>  8) & 0xFFu, 0);
        write_5((wa >> 16) & 0xFFu, (wa >> 24) & 0xFFu, 2);
        write_5((wb >>  0) & 0xFFu, (wb >>  8) & 0xFFu, 4);
        write_5((wb >> 16) & 0xFFu, (wb >> 24) & 0xFFu, 6);
    } else {
        const uint32_t p = __byte_perm(s_lut[(w6 >> 0) & 0xFFu], s_lut[(w6 >> 8) & 0xFFu], 0x5410u);
        __half2* d2 = reinterpret_cast<__half2*>(row_s + 120);
        d2[0] = __hmul2(u2_pair_to_h2(p,      mask, bias, c1025), scale_h2);
        d2[1] = __hmul2(u2_pair_to_h2(p >> 2, mask, bias, c1025), scale_h2);
        d2[2] = __hmul2(u2_pair_to_h2(p >> 4, mask, bias, c1025), scale_h2);
        d2[3] = __hmul2(u2_pair_to_h2(p >> 6, mask, bias, c1025), scale_h2);
    }
}

struct PackedTileRegs {
    uint32_t lo;
    uint32_t hi;
};

struct UnpackedGroupRegs {
    uint32_t p01;
    uint32_t p23;
    uint32_t p45;
    uint32_t p6;
    float sc0;
    float sc1;
};

__device__ __forceinline__ PackedTileRegs prefetch_5trit_tile(
    const uint32_t* __restrict__ packed_w,
    size_t row_blk_base,
    int g,
    int G,
    bool active,
    int lane
) {
    PackedTileRegs r{0u, 0u};
    if (active && g < G) {
        const uint32_t* tile_ptr = packed_w + row_blk_base + static_cast<size_t>(g) * 56u;
        r.lo = ldg_cg_u32(tile_ptr + lane);
        r.hi = (lane < 24) ? ldg_cg_u32(tile_ptr + 32 + lane) : 0u;
    }
    return r;
}

__device__ __forceinline__ UnpackedGroupRegs decode_5trit_tile_regs(
    PackedTileRegs raw,
    const uint32_t* __restrict__ s_lut,
    int lane
) {
    const int n_in_blk = lane >> 2;
    const int t_in_grp = lane & 3;
    const int src_half = (t_in_grp >> 1) * 8 + n_in_blk;
    const int byte_shift = (t_in_grp & 1) * 16;
    const int col_pair = t_in_grp * 2;

    const uint32_t s01 = __shfl_sync(0xFFFFFFFFu, raw.lo, src_half) >> byte_shift;
    const uint32_t s23 = __shfl_sync(0xFFFFFFFFu, raw.lo, 16 + src_half) >> byte_shift;
    const uint32_t s45 = __shfl_sync(0xFFFFFFFFu, raw.hi, src_half) >> byte_shift;
    const uint32_t s6  = __shfl_sync(0xFFFFFFFFu, raw.hi, 16 + n_in_blk);
    const uint32_t sw0 = __shfl_sync(0xFFFFFFFFu, raw.hi, 16 + col_pair + 0);
    const uint32_t sw1 = __shfl_sync(0xFFFFFFFFu, raw.hi, 16 + col_pair + 1);

    UnpackedGroupRegs u;
    u.p01 = __byte_perm(s_lut[s01 & 0xFFu], s_lut[(s01 >> 8) & 0xFFu], 0x5410u);
    u.p23 = __byte_perm(s_lut[s23 & 0xFFu], s_lut[(s23 >> 8) & 0xFFu], 0x5410u);
    u.p45 = __byte_perm(s_lut[s45 & 0xFFu], s_lut[(s45 >> 8) & 0xFFu], 0x5410u);
    u.p6  = __byte_perm(s_lut[s6  & 0xFFu], s_lut[(s6  >> 8) & 0xFFu], 0x5410u)
          >> (2 * t_in_grp);
    u.sc0 = __half2float(__ushort_as_half(static_cast<uint16_t>(sw0 >> 16)));
    u.sc1 = __half2float(__ushort_as_half(static_cast<uint16_t>(sw1 >> 16)));
    return u;
}

__device__ __forceinline__ uint32_t u2_pair_bits_unscaled(
    uint32_t shifted,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025
) {
    const __half2 h2 = u2_pair_to_h2(shifted, mask, bias, c1025);
    return *reinterpret_cast<const uint32_t*>(&h2);
}

// Evaluates 8 `m16n8k16` steps across TWO 8-col N-subtiles (`u0`, `u1`) sharing a single
// `ldmatrix.m8n8.x4` of `s_a`. Uses unscaled `{-1, 0, +1}` B-fragments with full-rate
// 64-term `.f16` MMA chunks on GeForce (`sm_86`/`sm_89`) and `.f32` MMA on A100 (`sm_80`),
// followed by exact FP32 per-column group-scale FMA.
__device__ __forceinline__ void mma_5trit_2n_from_regs(
    PackedTileRegs raw0,
    PackedTileRegs raw1,
    bool active0,
    bool active1,
    const uint32_t* __restrict__ s_lut,
    const __half (* __restrict__ s_a_slot)[136],
    int lane,
    int a_ldm_row,
    int a_ldm_col_off,
    uint32_t mask,
    uint32_t bias,
    __half2 c1025,
    float (&acc0)[4],
    float (&acc1)[4]
) {
    const UnpackedGroupRegs u0 = decode_5trit_tile_regs(raw0, s_lut, lane);
    const UnpackedGroupRegs u1 = decode_5trit_tile_regs(raw1, s_lut, lane);

    auto load_a = [&](int ks, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
        const uint32_t a_smem = static_cast<uint32_t>(
            __cvta_generic_to_shared(&s_a_slot[a_ldm_row][ks * 16 + a_ldm_col_off]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "r"(a_smem));
    };

    auto b_pair = [&](const UnpackedGroupRegs& u, int ks, uint32_t& b0, uint32_t& b1) {
        if (ks < 5) {
            b0 = u2_pair_bits_unscaled(u.p01 >> (2 * ks), mask, bias, c1025);
            b1 = u2_pair_bits_unscaled(u.p23 >> (2 * ks), mask, bias, c1025);
        } else if (ks == 5) {
            b0 = u2_pair_bits_unscaled(u.p45 >> 0, mask, bias, c1025);
            b1 = u2_pair_bits_unscaled(u.p45 >> 2, mask, bias, c1025);
        } else if (ks == 6) {
            b0 = u2_pair_bits_unscaled(u.p45 >> 4, mask, bias, c1025);
            b1 = u2_pair_bits_unscaled(u.p45 >> 6, mask, bias, c1025);
        } else {
            b0 = u2_pair_bits_unscaled(u.p45 >> 8, mask, bias, c1025);
            b1 = u2_pair_bits_unscaled(u.p6,       mask, bias, c1025);
        }
    };

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
    float g0[4] = {0.f, 0.f, 0.f, 0.f};
    float g1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        uint32_t a0, a1, a2, a3, b0, b1;
        load_a(ks, a0, a1, a2, a3);
        if (active0) {
            b_pair(u0, ks, b0, b1);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(g0[0]), "+f"(g0[1]), "+f"(g0[2]), "+f"(g0[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        if (active1) {
            b_pair(u1, ks, b0, b1);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(g1[0]), "+f"(g1[1]), "+f"(g1[2]), "+f"(g1[3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
#else
    float g0[4] = {0.f, 0.f, 0.f, 0.f};
    float g1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {
        uint32_t c0_01 = 0u, c0_23 = 0u;
        uint32_t c1_01 = 0u, c1_23 = 0u;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int ks = half_idx * 4 + s;
            uint32_t a0, a1, a2, a3, b0, b1;
            load_a(ks, a0, a1, a2, a3);
            if (active0) {
                b_pair(u0, ks, b0, b1);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                    : "+r"(c0_01), "+r"(c0_23)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            if (active1) {
                b_pair(u1, ks, b0, b1);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                    : "+r"(c1_01), "+r"(c1_23)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        if (active0) {
            const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&c0_01));
            const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&c0_23));
            g0[0] += f01.x; g0[1] += f01.y; g0[2] += f23.x; g0[3] += f23.y;
        }
        if (active1) {
            const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&c1_01));
            const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&c1_23));
            g1[0] += f01.x; g1[1] += f01.y; g1[2] += f23.x; g1[3] += f23.y;
        }
    }
#endif
    if (active0) {
        acc0[0] = fmaf(u0.sc0, g0[0], acc0[0]);
        acc0[1] = fmaf(u0.sc1, g0[1], acc0[1]);
        acc0[2] = fmaf(u0.sc0, g0[2], acc0[2]);
        acc0[3] = fmaf(u0.sc1, g0[3], acc0[3]);
    }
    if (active1) {
        acc1[0] = fmaf(u1.sc0, g1[0], acc1[0]);
        acc1[1] = fmaf(u1.sc1, g1[1], acc1[1]);
        acc1[2] = fmaf(u1.sc0, g1[2], acc1[2]);
        acc1[3] = fmaf(u1.sc1, g1[3], acc1[3]);
    }
}

// Native 5-Trit Tensor-Core GEMM for M >= 2 (`M_TILE = 16`, `2` 8-col N-subtiles per warp):
// - Coalesced 2-instruction warp loads (`7` 32B sectors = 100% sector efficiency) with
//   cross-iteration register prefetching across BOTH N-subtiles (`448` B/warp in flight).
// - Shares each `ldmatrix.m8n8.x4` of `s_a` across `16` N-columns per warp (`64/SPLIT_K` per CTA)
//   and loads only `m_row < M` into pre-zeroed `s_a`.
template <int SPLIT_K>
__global__ __launch_bounds__(128) void tc_gemm_5trit_f16_kernel(
    const __half* __restrict__ a,
    const uint32_t* __restrict__ packed_w,
    __half* __restrict__ out,
    int M,
    int N,
    int K
) {
    static_assert(SPLIT_K == 1 || SPLIT_K == 2 || SPLIT_K == 4, "unsupported SPLIT_K");
    constexpr int kWarpNSlots = 4 / SPLIT_K;
    constexpr int kNBlksPerCta = kWarpNSlots * 2;
    constexpr int kGroupsPerWarp = (SPLIT_K == 1) ? 2 : 1;
    constexpr int kGroupsPerStep = SPLIT_K * kGroupsPerWarp;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int warp_n = (SPLIT_K == 4) ? 0 : ((SPLIT_K == 2) ? (warp_id & 1) : warp_id);
    const int warp_k = (SPLIT_K == 4) ? warp_id : ((SPLIT_K == 2) ? (warp_id >> 1) : 0);
    const int n_blk0 = static_cast<int>(blockIdx.x) * kNBlksPerCta + warp_n * 2;
    const int n_blk1 = n_blk0 + 1;
    const bool active0 = (n_blk0 * 8 < N);
    const bool active1 = (n_blk1 * 8 < N);
    const int m_base = static_cast<int>(blockIdx.z) * 16;
    const int m_valid = min(16, M - m_base);
    const int a_vecs = m_valid * 16;
    const int G = K >> 7;

    uint32_t mask = 0x00030003u;
    uint32_t bias = 0x64006400u;
    asm("" : "+r"(mask));
    asm("" : "+r"(bias));
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));

    __shared__ uint32_t s_lut[256];
    __shared__ alignas(16) __half s_a[kGroupsPerStep][16][136];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    if (m_valid < 16) {
#pragma unroll
        for (int sk = 0; sk < kGroupsPerStep; ++sk) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int idx = tid + u * 128;
                *reinterpret_cast<uint4*>(&s_a[sk][idx >> 4][(idx & 15) * 8]) =
                    make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    __syncthreads();

    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t row_blk_base0 = static_cast<size_t>(n_blk0) * G * 56u;
    const size_t row_blk_base1 = static_cast<size_t>(n_blk1) * G * 56u;
    const int a_ldm_row = lane & 15;
    const int a_ldm_col_off = (lane >> 4) << 3;

    PackedTileRegs cur0[kGroupsPerWarp];
    PackedTileRegs cur1[kGroupsPerWarp];
#pragma unroll
    for (int sub = 0; sub < kGroupsPerWarp; ++sub) {
        const int g0 = warp_k * kGroupsPerWarp + sub;
        cur0[sub] = prefetch_5trit_tile(packed_w, row_blk_base0, g0, G, active0, lane);
        cur1[sub] = prefetch_5trit_tile(packed_w, row_blk_base1, g0, G, active1, lane);
    }

    for (int g_base = 0; g_base < G; g_base += kGroupsPerStep) {
        PackedTileRegs next0[kGroupsPerWarp];
        PackedTileRegs next1[kGroupsPerWarp];
        const int next_g_base = g_base + kGroupsPerStep;
#pragma unroll
        for (int sub = 0; sub < kGroupsPerWarp; ++sub) {
            const int gn = next_g_base + warp_k * kGroupsPerWarp + sub;
            next0[sub] = prefetch_5trit_tile(packed_w, row_blk_base0, gn, G, active0, lane);
            next1[sub] = prefetch_5trit_tile(packed_w, row_blk_base1, gn, G, active1, lane);
        }

#pragma unroll
        for (int sk = 0; sk < kGroupsPerStep; ++sk) {
            if (g_base + sk < G) {
                for (int idx = tid; idx < a_vecs; idx += 128) {
                    const int m_row = idx >> 4;
                    const int k_vec = idx & 15;
                    const __half* src = a + static_cast<size_t>(m_base + m_row) * K
                                        + static_cast<size_t>(g_base + sk) * 128
                                        + k_vec * 8;
                    *reinterpret_cast<uint4*>(&s_a[sk][m_row][k_vec * 8]) =
                        *reinterpret_cast<const uint4*>(src);
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int sub = 0; sub < kGroupsPerWarp; ++sub) {
            const int a_slot = warp_k * kGroupsPerWarp + sub;
            const int g = g_base + a_slot;
            if (active0 && g < G) {
                mma_5trit_2n_from_regs(
                    cur0[sub], cur1[sub], active0, active1, s_lut, s_a[a_slot], lane,
                    a_ldm_row, a_ldm_col_off, mask, bias, c1025, acc0, acc1);
            }
            cur0[sub] = next0[sub];
            cur1[sub] = next1[sub];
        }
        __syncthreads();
    }

    const int group_id = lane >> 2;
    const int col_pair = (lane & 3) * 2;

    if (SPLIT_K == 4) {
        float (*red)[2][16][8] = reinterpret_cast<float (*)[2][16][8]>(&s_a[0][0][0]);
        if (warp_k > 0) {
            if (active0) {
                red[warp_k - 1][0][group_id][col_pair + 0]     = acc0[0];
                red[warp_k - 1][0][group_id][col_pair + 1]     = acc0[1];
                red[warp_k - 1][0][group_id + 8][col_pair + 0] = acc0[2];
                red[warp_k - 1][0][group_id + 8][col_pair + 1] = acc0[3];
            }
            if (active1) {
                red[warp_k - 1][1][group_id][col_pair + 0]     = acc1[0];
                red[warp_k - 1][1][group_id][col_pair + 1]     = acc1[1];
                red[warp_k - 1][1][group_id + 8][col_pair + 0] = acc1[2];
                red[warp_k - 1][1][group_id + 8][col_pair + 1] = acc1[3];
            }
        }
        __syncthreads();
        if (warp_k != 0 || !active0) return;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int rr = group_id + ((i >= 2) ? 8 : 0);
            const int cc = col_pair + (i & 1);
            acc0[i] = (acc0[i] + red[0][0][rr][cc]) + (red[1][0][rr][cc] + red[2][0][rr][cc]);
            if (active1) {
                acc1[i] = (acc1[i] + red[0][1][rr][cc]) + (red[1][1][rr][cc] + red[2][1][rr][cc]);
            }
        }
    } else if (SPLIT_K == 2) {
        float (*red)[2][16][8] = reinterpret_cast<float (*)[2][16][8]>(&s_a[0][0][0]);
        if (warp_k == 1) {
            if (active0) {
                red[warp_n][0][group_id][col_pair + 0]     = acc0[0];
                red[warp_n][0][group_id][col_pair + 1]     = acc0[1];
                red[warp_n][0][group_id + 8][col_pair + 0] = acc0[2];
                red[warp_n][0][group_id + 8][col_pair + 1] = acc0[3];
            }
            if (active1) {
                red[warp_n][1][group_id][col_pair + 0]     = acc1[0];
                red[warp_n][1][group_id][col_pair + 1]     = acc1[1];
                red[warp_n][1][group_id + 8][col_pair + 0] = acc1[2];
                red[warp_n][1][group_id + 8][col_pair + 1] = acc1[3];
            }
        }
        __syncthreads();
        if (warp_k != 0 || !active0) return;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int rr = group_id + ((i >= 2) ? 8 : 0);
            const int cc = col_pair + (i & 1);
            acc0[i] += red[warp_n][0][rr][cc];
            if (active1) {
                acc1[i] += red[warp_n][1][rr][cc];
            }
        }
    } else {
        if (!active0) return;
    }

    const bool h2_aligned = (reinterpret_cast<uintptr_t>(out) & 3u) == 0u;
    auto store_subtile = [&](int nb, const float (&ac)[4]) {
        const int out_col = nb * 8 + col_pair;
        const int row0 = m_base + group_id;
        if (row0 < M) {
            __half* dst0 = out + static_cast<size_t>(row0) * N + out_col;
            const __half2 v01 = __floats2half2_rn(ac[0], ac[1]);
            if (h2_aligned) {
                *reinterpret_cast<__half2*>(dst0) = v01;
            } else {
                dst0[0] = __low2half(v01);
                dst0[1] = __high2half(v01);
            }
        }
        const int row1 = row0 + 8;
        if (row1 < M) {
            __half* dst1 = out + static_cast<size_t>(row1) * N + out_col;
            const __half2 v23 = __floats2half2_rn(ac[2], ac[3]);
            if (h2_aligned) {
                *reinterpret_cast<__half2*>(dst1) = v23;
            } else {
                dst1[0] = __low2half(v23);
                dst1[1] = __high2half(v23);
            }
        }
    };
    store_subtile(n_blk0, acc0);
    if (active1) {
        store_subtile(n_blk1, acc1);
    }
}

// Fast coalesced strip dequantizer:
// 4 warps/CTA; each warp owns one (n_blk, g) tile (8 rows x 128 halves in padded SMEM).
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
    __shared__ alignas(16) __half s_tile[4][8][136];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    __syncthreads();

    const int G = K >> 7;
    const int n_blk_start = n_start >> 3;
    const int n_blks_chunk = n_chunk >> 3;
    const int64_t total_tiles = static_cast<int64_t>(n_blks_chunk) * G;
    const int64_t tile_idx = static_cast<int64_t>(blockIdx.x) * 4 + warp_id;
    if (tile_idx >= total_tiles) return;

    const int local_n_blk = static_cast<int>(tile_idx / G);
    const int g = static_cast<int>(tile_idx - static_cast<int64_t>(local_n_blk) * G);
    const int n_blk = n_blk_start + local_n_blk;
    const size_t base = (static_cast<size_t>(n_blk) * G + g) * 56u + r;

    uint32_t mask = 0x00030003u;
    uint32_t bias = 0x64006400u;
    asm("" : "+r"(mask));
    asm("" : "+r"(bias));
    const __half2 c1025 = __halves2half2(__ushort_as_half(0x6401u), __ushort_as_half(0x6401u));

    unpack_5trit_tile_to_f16_smem(
        packed_w, base, s_lut, s_tile[warp_id], r, part, mask, bias, c1025);
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

// Coalesced zero-divergence 5-Trit -> Marlin U2B2 (kU2B2, g128) repacker for M > 32 prefill.
// Grid: (G = K/128, n_tiles = N/64), Block: 128 threads (4 warps).
// Outputs both Marlin-tiled 2-bit weights (q = u + 1 in {1,2,3}, 2.0 bpw) and
// Marlin-permuted FP16 scales in a single pass (2 CTA barriers total).
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
    __shared__ alignas(4) uint8_t s_q[64][132];
    __shared__ alignas(4) uint16_t sh_repack_u2[8][128];

    s_lut[tid]       = encode_5trit_lut10(static_cast<uint32_t>(tid));
    s_lut[tid + 128] = encode_5trit_lut10(static_cast<uint32_t>(tid + 128));
    __syncthreads();

    // Phase 1: unpack 64 rows x 128 trits into s_q[64][128] with zero intra-warp divergence.
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

    // Phase 2: repack all 8 k-tiles (16x64 each) into sh_repack_u2[8][128], sync once,
    // then coalesce all 512 uint32_t Marlin words across all 128 threads via PRMT.
    const int warp_id = tid >> 5;
    const int th_id = tid & 31;
    const int tc_col = th_id >> 2;
    const int tc_row = (th_id & 3) * 2;
    const int cur_n = warp_id * 16 + tc_col;
    const uint8_t* q_b1 = s_q[cur_n];
    const uint8_t* q_b2 = s_q[cur_n + 8];
    const int sh_slot = th_id * 4 + warp_id;

#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
        const int k_base = sub * 16 + tc_row;
        const uint32_t p01 = *reinterpret_cast<const uint16_t*>(&q_b1[k_base + 0]);
        const uint32_t p23 = *reinterpret_cast<const uint16_t*>(&q_b1[k_base + 8]);
        const uint32_t p45 = *reinterpret_cast<const uint16_t*>(&q_b2[k_base + 0]);
        const uint32_t p67 = *reinterpret_cast<const uint16_t*>(&q_b2[k_base + 8]);
        const uint32_t lo_byte = (p01 & 0xFFu) | ((p23 & 0xFFu) << 2)
                               | ((p45 & 0xFFu) << 4) | ((p67 & 0xFFu) << 6);
        const uint32_t hi_byte = (p01 >> 8) | ((p23 >> 8) << 2)
                               | ((p45 >> 8) << 4) | ((p67 >> 8) << 6);
        sh_repack_u2[sub][sh_slot] = static_cast<uint16_t>(lo_byte | (hi_byte << 8));
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int idx = tid + u * 128;
        const int sub = idx >> 6;
        const int elem = idx & 63;
        const uint32_t pair = *reinterpret_cast<const uint32_t*>(&sh_repack_u2[sub][elem * 2]);
        const size_t k_tile_id = static_cast<size_t>(g) * 8 + sub;
        const size_t out_offset = (k_tile_id * static_cast<size_t>(n_tiles) + n_tile_id) * 64u;
        dst_marlin[out_offset + elem] = __byte_perm(pair, 0u, 0x3120u);
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
        if (N <= 5376 && K >= 6144) {
            const int n_blocks = N / 8;
            gemv_5trit_m1_f16_kernel<4><<<dim3(n_blocks, 1, 1), 128, 0, stream>>>(a, packed_w, out, N, K);
        } else if (N <= 5376 && K >= 1024) {
            const int n_blocks = N / 8;
            gemv_5trit_m1_f16_kernel<2><<<dim3(n_blocks, 1, 1), 64, 0, stream>>>(a, packed_w, out, N, K);
        } else {
            const int n_blocks = (N + 15) / 16;
            gemv_5trit_m1_f16_kernel<1><<<dim3(n_blocks, 1, 1), 64, 0, stream>>>(a, packed_w, out, N, K);
        }
    } else {
        const int m_blocks = (M + 15) / 16;
        if (m_blocks > 65535) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        if (N <= 5376 && K >= 4096) {
            const int n_blocks = (N + 15) / 16;
            tc_gemm_5trit_f16_kernel<4><<<dim3(n_blocks, 1, m_blocks), 128, 0, stream>>>(
                a, packed_w, out, M, N, K);
        } else if (K >= 1024) {
            const int n_blocks = (N + 31) / 32;
            tc_gemm_5trit_f16_kernel<2><<<dim3(n_blocks, 1, m_blocks), 128, 0, stream>>>(
                a, packed_w, out, M, N, K);
        } else {
            const int n_blocks = (N + 63) / 64;
            tc_gemm_5trit_f16_kernel<1><<<dim3(n_blocks, 1, m_blocks), 128, 0, stream>>>(
                a, packed_w, out, M, N, K);
        }
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
    const int64_t total_tiles = static_cast<int64_t>(n_chunk >> 3) * G;
    const int64_t blocks = (total_tiles + 3) / 4;
    if (blocks > 2147483647LL) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ex->stream);
    dequant_5trit_to_f16_kernel<<<dim3(static_cast<unsigned>(blocks), 1, 1), 128, 0, stream>>>(
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
