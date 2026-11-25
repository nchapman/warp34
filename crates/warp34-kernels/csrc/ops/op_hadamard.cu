#include "../warp34_cuda.h"
#include "hadamard_inline.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

template <typename T>
struct HadScalar;

template <>
struct HadScalar<__half> {
    using Vec2 = __half2;
    __device__ __forceinline__ static float2 to_float2(uint32_t raw) {
        return __half22float2(*reinterpret_cast<const __half2*>(&raw));
    }
    __device__ __forceinline__ static float to_float(__half v) { return __half2float(v); }
    __device__ __forceinline__ static __half2 from_floats2(float a, float b) {
        return __floats2half2_rn(a, b);
    }
    __device__ __forceinline__ static __half low(__half2 v) { return __low2half(v); }
    __device__ __forceinline__ static __half high(__half2 v) { return __high2half(v); }
};

template <>
struct HadScalar<__nv_bfloat16> {
    using Vec2 = __nv_bfloat162;
    __device__ __forceinline__ static float2 to_float2(uint32_t raw) {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
    }
    __device__ __forceinline__ static float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    __device__ __forceinline__ static __nv_bfloat162 from_floats2(float a, float b) {
        return __floats2bfloat162_rn(a, b);
    }
    __device__ __forceinline__ static __nv_bfloat16 low(__nv_bfloat162 v) {
        return __low2bfloat16(v);
    }
    __device__ __forceinline__ static __nv_bfloat16 high(__nv_bfloat162 v) {
        return __high2bfloat16(v);
    }
};

// Each 4-lane subgroup owns H16 (4 values/lane); each 16-lane subgroup owns H64 (4 values/lane);
// each 32-lane warp owns H128 (4 values/lane).
// Executing bits 0..1 via an in-register H4 butterfly followed by bits 2..log2(B)-1 via warp
// shuffles preserves strict bit-order FP32 summation while coalescing 8-byte (uint2) loads/stores.
template <typename T, int B>
__global__ void block_hadamard_16bit(const T* src, T* dst,
                                     int k, int padded_k, int64_t groups) {
    static_assert(B == 16 || B == 64 || B == 128,
                  "block_hadamard_16bit supports B=16, B=64, or B=128");
    using S = HadScalar<T>;
    constexpr int lanes = B / 4;
    constexpr float scale = (B == 16) ? 0.25f
                                      : (B == 64) ? 0.125f
                                                  : 0.08838834764831845f; // 1 / sqrt(128)
    const int lane = threadIdx.x & (lanes - 1);
    const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / lanes;
    const int64_t step = (int64_t(gridDim.x) * blockDim.x) / lanes;
    const int blocks_per_row = padded_k / B;
    const bool vec_aligned =
        ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 7u) == 0u;

    for (int64_t group = first; group < groups; group += step) {
        const int64_t row = group / blocks_per_row;
        const int col = int(group % blocks_per_row) * B + lane * 4;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if (col < k) {
            const T* sp = src + row * k + col;
            if (vec_aligned) {
                const uint2 raw = *reinterpret_cast<const uint2*>(sp);
                const float2 f01 = S::to_float2(raw.x);
                const float2 f23 = S::to_float2(raw.y);
                v0 = f01.x;
                v1 = f01.y;
                v2 = f23.x;
                v3 = f23.y;
            } else {
                v0 = S::to_float(sp[0]);
                v1 = S::to_float(sp[1]);
                v2 = S::to_float(sp[2]);
                v3 = S::to_float(sp[3]);
            }
        }

        // Stage h=1, 2 in registers (H4 butterfly)
        {
            const float a0 = v0 + v1;
            const float a1 = v0 - v1;
            const float a2 = v2 + v3;
            const float a3 = v2 - v3;
            v0 = a0 + a2;
            v1 = a1 + a3;
            v2 = a0 - a2;
            v3 = a1 - a3;
        }

        // Stage h=4..B/2 across subgroup lanes
        #pragma unroll
        for (int h = 1; h < lanes; h *= 2) {
            const float o0 = __shfl_xor_sync(0xffffffffu, v0, h, lanes);
            const float o1 = __shfl_xor_sync(0xffffffffu, v1, h, lanes);
            const float o2 = __shfl_xor_sync(0xffffffffu, v2, h, lanes);
            const float o3 = __shfl_xor_sync(0xffffffffu, v3, h, lanes);
            const bool sub = (lane & h) != 0;
            v0 = sub ? (o0 - v0) : (v0 + o0);
            v1 = sub ? (o1 - v1) : (v1 + o1);
            v2 = sub ? (o2 - v2) : (v2 + o2);
            v3 = sub ? (o3 - v3) : (v3 + o3);
        }

        T* dp = dst + row * padded_k + col;
        const typename S::Vec2 out01 = S::from_floats2(v0 * scale, v1 * scale);
        const typename S::Vec2 out23 = S::from_floats2(v2 * scale, v3 * scale);
        if (vec_aligned) {
            uint2 packed_out;
            packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
            packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
            *reinterpret_cast<uint2*>(dp) = packed_out;
        } else {
            dp[0] = S::low(out01);
            dp[1] = S::high(out01);
            dp[2] = S::low(out23);
            dp[3] = S::high(out23);
        }
    }
}

// Each 128-thread (4-warp) subgroup owns H512 (4 values/lane, 2 groups per 256-thread CTA).
// Bits 0..1 execute via an in-register H4 butterfly, bits 2..6 via full-warp shuffles (H128),
// and bits 7..8 across the 4 warps through 4 KiB shared memory, preserving exact bit-order
// FP32 summation and coalesced 8-byte (uint2) loads/stores (race-free for in-place src == dst).
__device__ __forceinline__ float xor_sign_byte_f32(float v, uint32_t s_byte) {
    return __uint_as_float(__float_as_uint(v) ^ ((s_byte & 0x80u) << 24));
}

template <typename T>
__global__ void block_hadamard_512_16bit(const T* __restrict__ src,
                                         const int8_t* __restrict__ pre_signs,
                                         T* __restrict__ dst,
                                         int k, int padded_k, int64_t groups) {
    using S = HadScalar<T>;
    constexpr float scale = 0.04419417382415922f; // 1 / sqrt(512)
    __shared__ float4 smem[2][4][32];

    const int grp_in_cta = threadIdx.x >> 7;        // 0..1
    const int warp_in_grp = (threadIdx.x >> 5) & 3; // 0..3
    const int lane = threadIdx.x & 31;              // 0..31
    const int blocks_per_row = padded_k / 512;
    const bool vec_aligned =
        ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 7u) == 0u;
    const bool signs_u32_aligned =
        pre_signs != nullptr && ((reinterpret_cast<uintptr_t>(pre_signs) & 3u) == 0u);
    const bool sub7 = (warp_in_grp & 1) != 0;
    const bool sub8 = (warp_in_grp & 2) != 0;

    const int64_t num_cta_iters = (groups + 1) / 2;
    for (int64_t iter = blockIdx.x; iter < num_cta_iters; iter += gridDim.x) {
        const int64_t group = iter * 2 + grp_in_cta;
        const bool active = group < groups;
        int64_t row = 0;
        int col = 0;

        if (active) {
            if (blocks_per_row == 1) {
                row = group;
                col = warp_in_grp * 128 + lane * 4;
            } else if (static_cast<uint64_t>(group) <= 0xFFFFFFFFull) {
                const uint32_t g32 = static_cast<uint32_t>(group);
                const uint32_t bpr = static_cast<uint32_t>(blocks_per_row);
                row = static_cast<int64_t>(g32 / bpr);
                col = static_cast<int>(g32 % bpr) * 512 + warp_in_grp * 128 + lane * 4;
            } else {
                row = group / blocks_per_row;
                col = int(group % blocks_per_row) * 512 + warp_in_grp * 128 + lane * 4;
            }
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (col < k) {
                const T* sp = src + row * k + col;
                if (vec_aligned) {
                    const uint2 raw = *reinterpret_cast<const uint2*>(sp);
                    const float2 f01 = S::to_float2(raw.x);
                    const float2 f23 = S::to_float2(raw.y);
                    v0 = f01.x;
                    v1 = f01.y;
                    v2 = f23.x;
                    v3 = f23.y;
                } else {
                    v0 = S::to_float(sp[0]);
                    v1 = S::to_float(sp[1]);
                    v2 = S::to_float(sp[2]);
                    v3 = S::to_float(sp[3]);
                }
                if (pre_signs) {
                    if (signs_u32_aligned) {
                        const uint32_t s4 = *reinterpret_cast<const uint32_t*>(pre_signs + col);
                        v0 = xor_sign_byte_f32(v0, s4 >>  0);
                        v1 = xor_sign_byte_f32(v1, s4 >>  8);
                        v2 = xor_sign_byte_f32(v2, s4 >> 16);
                        v3 = xor_sign_byte_f32(v3, s4 >> 24);
                    } else {
                        v0 = xor_sign_byte_f32(v0, static_cast<uint8_t>(pre_signs[col + 0]));
                        v1 = xor_sign_byte_f32(v1, static_cast<uint8_t>(pre_signs[col + 1]));
                        v2 = xor_sign_byte_f32(v2, static_cast<uint8_t>(pre_signs[col + 2]));
                        v3 = xor_sign_byte_f32(v3, static_cast<uint8_t>(pre_signs[col + 3]));
                    }
                }
            }

            // Stage h=1, 2 in registers (H4 butterfly)
            {
                const float a0 = v0 + v1;
                const float a1 = v0 - v1;
                const float a2 = v2 + v3;
                const float a3 = v2 - v3;
                v0 = a0 + a2;
                v1 = a1 + a3;
                v2 = a0 - a2;
                v3 = a1 - a3;
            }

            // Stage h=4..64 across warp lanes (H128)
            #pragma unroll
            for (int h = 1; h < 32; h *= 2) {
                const float o0 = __shfl_xor_sync(0xffffffffu, v0, h, 32);
                const float o1 = __shfl_xor_sync(0xffffffffu, v1, h, 32);
                const float o2 = __shfl_xor_sync(0xffffffffu, v2, h, 32);
                const float o3 = __shfl_xor_sync(0xffffffffu, v3, h, 32);
                const bool sub = (lane & h) != 0;
                v0 = sub ? (o0 - v0) : (v0 + o0);
                v1 = sub ? (o1 - v1) : (v1 + o1);
                v2 = sub ? (o2 - v2) : (v2 + o2);
                v3 = sub ? (o3 - v3) : (v3 + o3);
            }

            smem[grp_in_cta][warp_in_grp][lane] = make_float4(v0, v1, v2, v3);
        }

        __syncthreads();

        if (active) {
            const float4 w0 = smem[grp_in_cta][0][lane];
            const float4 w1 = smem[grp_in_cta][1][lane];
            const float4 w2 = smem[grp_in_cta][2][lane];
            const float4 w3 = smem[grp_in_cta][3][lane];

            auto combine = [&](float c0, float c1, float c2, float c3) -> float {
                const float a = sub7 ? (c0 - c1) : (c0 + c1);
                const float b = sub7 ? (c2 - c3) : (c2 + c3);
                return (sub8 ? (a - b) : (a + b)) * scale;
            };

            const float r0 = combine(w0.x, w1.x, w2.x, w3.x);
            const float r1 = combine(w0.y, w1.y, w2.y, w3.y);
            const float r2 = combine(w0.z, w1.z, w2.z, w3.z);
            const float r3 = combine(w0.w, w1.w, w2.w, w3.w);

            T* dp = dst + row * padded_k + col;
            const typename S::Vec2 out01 = S::from_floats2(r0, r1);
            const typename S::Vec2 out23 = S::from_floats2(r2, r3);
            if (vec_aligned) {
                uint2 packed_out;
                packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
                packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
                *reinterpret_cast<uint2*>(dp) = packed_out;
            } else {
                dp[0] = S::low(out01);
                dp[1] = S::high(out01);
                dp[2] = S::low(out23);
                dp[3] = S::high(out23);
            }
        }

        if (iter + gridDim.x < num_cta_iters) {
            __syncthreads();
        }
    }
}

// H_1024 factored as H_8 (8 warps in SMEM) ⊗ H_128 (in-warp): 256 threads, 4 KiB SMEM, 1 barrier.
template <typename T>
__global__ __launch_bounds__(256) void block_hadamard_1024_16bit(
    const T* __restrict__ src,
    const int8_t* __restrict__ pre_signs,
    T* __restrict__ dst,
    int k,
    int padded_k,
    int64_t groups
) {
    using S = HadScalar<T>;
    constexpr float scale = 0.03125f; // 1 / sqrt(1024) = 1 / 32
    __shared__ float4 smem[8][32];

    const int warp_id = threadIdx.x >> 5; // 0..7
    const int lane    = threadIdx.x & 31; // 0..31
    const int blocks_per_row = padded_k >> 10;
    const bool vec_aligned =
        ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 7u) == 0u;
    const bool signs_u32_aligned =
        pre_signs != nullptr && ((reinterpret_cast<uintptr_t>(pre_signs) & 3u) == 0u);
    const bool sub7 = (warp_id & 1) != 0;
    const bool sub8 = (warp_id & 2) != 0;
    const bool sub9 = (warp_id & 4) != 0;

    for (int64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        int64_t row;
        int base_col;
        if (blocks_per_row == 1) {
            row = group;
            base_col = 0;
        } else if (static_cast<uint64_t>(group) <= 0xFFFFFFFFull) {
            const uint32_t g32 = static_cast<uint32_t>(group);
            const uint32_t bpr = static_cast<uint32_t>(blocks_per_row);
            row = static_cast<int64_t>(g32 / bpr);
            base_col = static_cast<int>(g32 % bpr) << 10;
        } else {
            row = group / blocks_per_row;
            base_col = static_cast<int>(group % blocks_per_row) << 10;
        }
        const int col = base_col + warp_id * 128 + lane * 4;

        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if (col < k) {
            const T* sp = src + row * k + col;
            if (vec_aligned) {
                const uint2 raw = *reinterpret_cast<const uint2*>(sp);
                const float2 f01 = S::to_float2(raw.x);
                const float2 f23 = S::to_float2(raw.y);
                v0 = f01.x;
                v1 = f01.y;
                v2 = f23.x;
                v3 = f23.y;
            } else {
                v0 = S::to_float(sp[0]);
                v1 = S::to_float(sp[1]);
                v2 = S::to_float(sp[2]);
                v3 = S::to_float(sp[3]);
            }
            if (pre_signs) {
                if (signs_u32_aligned) {
                    const uint32_t s4 = *reinterpret_cast<const uint32_t*>(pre_signs + col);
                    v0 = xor_sign_byte_f32(v0, s4 >>  0);
                    v1 = xor_sign_byte_f32(v1, s4 >>  8);
                    v2 = xor_sign_byte_f32(v2, s4 >> 16);
                    v3 = xor_sign_byte_f32(v3, s4 >> 24);
                } else {
                    v0 = xor_sign_byte_f32(v0, static_cast<uint8_t>(pre_signs[col + 0]));
                    v1 = xor_sign_byte_f32(v1, static_cast<uint8_t>(pre_signs[col + 1]));
                    v2 = xor_sign_byte_f32(v2, static_cast<uint8_t>(pre_signs[col + 2]));
                    v3 = xor_sign_byte_f32(v3, static_cast<uint8_t>(pre_signs[col + 3]));
                }
            }
        }

        // H_4 in registers + H_32 across warp lanes = H_128.
        {
            const float a0 = v0 + v1;
            const float a1 = v0 - v1;
            const float a2 = v2 + v3;
            const float a3 = v2 - v3;
            v0 = a0 + a2;
            v1 = a1 + a3;
            v2 = a0 - a2;
            v3 = a1 - a3;
        }
        #pragma unroll
        for (int h = 1; h < 32; h *= 2) {
            const float o0 = __shfl_xor_sync(0xffffffffu, v0, h, 32);
            const float o1 = __shfl_xor_sync(0xffffffffu, v1, h, 32);
            const float o2 = __shfl_xor_sync(0xffffffffu, v2, h, 32);
            const float o3 = __shfl_xor_sync(0xffffffffu, v3, h, 32);
            const bool sub = (lane & h) != 0;
            v0 = sub ? (o0 - v0) : (v0 + o0);
            v1 = sub ? (o1 - v1) : (v1 + o1);
            v2 = sub ? (o2 - v2) : (v2 + o2);
            v3 = sub ? (o3 - v3) : (v3 + o3);
        }

        smem[warp_id][lane] = make_float4(v0, v1, v2, v3);
        __syncthreads();

        const float4 w0 = smem[0][lane];
        const float4 w1 = smem[1][lane];
        const float4 w2 = smem[2][lane];
        const float4 w3 = smem[3][lane];
        const float4 w4 = smem[4][lane];
        const float4 w5 = smem[5][lane];
        const float4 w6 = smem[6][lane];
        const float4 w7 = smem[7][lane];

        auto combine8 = [&](float c0, float c1, float c2, float c3,
                            float c4, float c5, float c6, float c7) -> float {
            const float a0 = sub7 ? (c0 - c1) : (c0 + c1);
            const float a1 = sub7 ? (c2 - c3) : (c2 + c3);
            const float a2 = sub7 ? (c4 - c5) : (c4 + c5);
            const float a3 = sub7 ? (c6 - c7) : (c6 + c7);
            const float b0 = sub8 ? (a0 - a1) : (a0 + a1);
            const float b1 = sub8 ? (a2 - a3) : (a2 + a3);
            return (sub9 ? (b0 - b1) : (b0 + b1)) * scale;
        };

        const float r0 = combine8(w0.x, w1.x, w2.x, w3.x, w4.x, w5.x, w6.x, w7.x);
        const float r1 = combine8(w0.y, w1.y, w2.y, w3.y, w4.y, w5.y, w6.y, w7.y);
        const float r2 = combine8(w0.z, w1.z, w2.z, w3.z, w4.z, w5.z, w6.z, w7.z);
        const float r3 = combine8(w0.w, w1.w, w2.w, w3.w, w4.w, w5.w, w6.w, w7.w);

        T* dp = dst + row * padded_k + col;
        const typename S::Vec2 out01 = S::from_floats2(r0, r1);
        const typename S::Vec2 out23 = S::from_floats2(r2, r3);
        if (vec_aligned) {
            uint2 packed_out;
            packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
            packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
            *reinterpret_cast<uint2*>(dp) = packed_out;
        } else {
            dp[0] = S::low(out01);
            dp[1] = S::high(out01);
            dp[2] = S::low(out23);
            dp[3] = S::high(out23);
        }
        if (group + gridDim.x < groups) {
            __syncthreads();
        }
    }
}

// H_4096 factored as H_64 ⊗ H_64 in shared memory (`64 × 65` float = 16.25 KiB).
// 512 threads (16 warps = 32 half-warps × 16 lanes, p = 0..1) halves single-CTA M=1 latency
// while preserving 100% occupancy (3 CTAs/SM = 1536 threads/SM) on prefill.
// Bank-conflict-free across all 3 passes:
//   - Two half-warps in each warp own rows/cols separated by 16 (16*65 ≡ 16 mod 32 banks).
//   - Within each half-warp, lane=0..15 writes/reads contiguous stride-1 indices lane+{0,16,32,48}
//     (bijection on 0..63; Pass 2 evaluates H_64 = H_4(bits 4..5) ⊗ H_16(bits 0..3) identically).
template <typename T>
__global__ __launch_bounds__(512) void block_hadamard_4096_16bit(
    const T* __restrict__ src,
    const int8_t* __restrict__ pre_signs,
    T* __restrict__ dst,
    int k,
    int padded_k,
    int64_t groups
) {
    using S = HadScalar<T>;
    constexpr float scale = 0.015625f; // 1 / sqrt(4096) = 1 / 64
    __shared__ float smem[64][65];

    const int warp_id = threadIdx.x >> 5;       // 0..15
    const int sub_hw  = (threadIdx.x >> 4) & 1; // 0..1
    const int lane    = threadIdx.x & 15;       // 0..15
    const int rc_base = warp_id + sub_hw * 16;  // 0..31 (half-warps in a warp differ by 16)
    const int blocks_per_row = padded_k >> 12;
    const bool vec_aligned =
        ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 7u) == 0u;
    const bool signs_u32_aligned =
        pre_signs != nullptr && ((reinterpret_cast<uintptr_t>(pre_signs) & 3u) == 0u);

    for (int64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        int64_t row;
        int base_col;
        if (blocks_per_row == 1) {
            row = group;
            base_col = 0;
        } else if (static_cast<uint64_t>(group) <= 0xFFFFFFFFull) {
            const uint32_t g32 = static_cast<uint32_t>(group);
            const uint32_t bpr = static_cast<uint32_t>(blocks_per_row);
            row = static_cast<int64_t>(g32 / bpr);
            base_col = static_cast<int>(g32 % bpr) << 12;
        } else {
            row = group / blocks_per_row;
            base_col = static_cast<int>(group % blocks_per_row) << 12;
        }

        // Pass 1: H_64 along each of the 64 rows (low 6 bits: h = 1..32).
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int r = rc_base + p * 32;
            const int c = lane * 4;
            const int in_blk = r * 64 + c;
            const int col = base_col + in_blk;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (col < k) {
                const T* sp = src + row * k + col;
                if (vec_aligned) {
                    const uint2 raw = *reinterpret_cast<const uint2*>(sp);
                    const float2 f01 = S::to_float2(raw.x);
                    const float2 f23 = S::to_float2(raw.y);
                    v0 = f01.x;
                    v1 = f01.y;
                    v2 = f23.x;
                    v3 = f23.y;
                } else {
                    v0 = S::to_float(sp[0]);
                    v1 = S::to_float(sp[1]);
                    v2 = S::to_float(sp[2]);
                    v3 = S::to_float(sp[3]);
                }
                if (pre_signs) {
                    if (signs_u32_aligned) {
                        const uint32_t s4 = *reinterpret_cast<const uint32_t*>(pre_signs + col);
                        v0 = xor_sign_byte_f32(v0, s4 >>  0);
                        v1 = xor_sign_byte_f32(v1, s4 >>  8);
                        v2 = xor_sign_byte_f32(v2, s4 >> 16);
                        v3 = xor_sign_byte_f32(v3, s4 >> 24);
                    } else {
                        v0 = xor_sign_byte_f32(v0, static_cast<uint8_t>(pre_signs[col + 0]));
                        v1 = xor_sign_byte_f32(v1, static_cast<uint8_t>(pre_signs[col + 1]));
                        v2 = xor_sign_byte_f32(v2, static_cast<uint8_t>(pre_signs[col + 2]));
                        v3 = xor_sign_byte_f32(v3, static_cast<uint8_t>(pre_signs[col + 3]));
                    }
                }
            }
            {
                const float a0 = v0 + v1;
                const float a1 = v0 - v1;
                const float a2 = v2 + v3;
                const float a3 = v2 - v3;
                v0 = a0 + a2;
                v1 = a1 + a3;
                v2 = a0 - a2;
                v3 = a1 - a3;
            }
            #pragma unroll
            for (int h = 1; h < 16; h *= 2) {
                const float o0 = __shfl_xor_sync(0xffffffffu, v0, h, 16);
                const float o1 = __shfl_xor_sync(0xffffffffu, v1, h, 16);
                const float o2 = __shfl_xor_sync(0xffffffffu, v2, h, 16);
                const float o3 = __shfl_xor_sync(0xffffffffu, v3, h, 16);
                const bool sub = (lane & h) != 0;
                v0 = sub ? (o0 - v0) : (v0 + o0);
                v1 = sub ? (o1 - v1) : (v1 + o1);
                v2 = sub ? (o2 - v2) : (v2 + o2);
                v3 = sub ? (o3 - v3) : (v3 + o3);
            }
            // Store at stride-1 across lane (zero bank conflicts for all 32 warp lanes).
            smem[r][lane +  0] = v0;
            smem[r][lane + 16] = v1;
            smem[r][lane + 32] = v2;
            smem[r][lane + 48] = v3;
        }
        __syncthreads();

        // Pass 2: H_64 along each of the 64 columns (H_4 on bits 4..5 ⊗ H_16 on bits 0..3), scaled by 1/64.
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int c = rc_base + p * 32;
            float v0 = smem[lane +  0][c];
            float v1 = smem[lane + 16][c];
            float v2 = smem[lane + 32][c];
            float v3 = smem[lane + 48][c];
            {
                const float a0 = v0 + v1;
                const float a1 = v0 - v1;
                const float a2 = v2 + v3;
                const float a3 = v2 - v3;
                v0 = a0 + a2;
                v1 = a1 + a3;
                v2 = a0 - a2;
                v3 = a1 - a3;
            }
            #pragma unroll
            for (int h = 1; h < 16; h *= 2) {
                const float o0 = __shfl_xor_sync(0xffffffffu, v0, h, 16);
                const float o1 = __shfl_xor_sync(0xffffffffu, v1, h, 16);
                const float o2 = __shfl_xor_sync(0xffffffffu, v2, h, 16);
                const float o3 = __shfl_xor_sync(0xffffffffu, v3, h, 16);
                const bool sub = (lane & h) != 0;
                v0 = sub ? (o0 - v0) : (v0 + o0);
                v1 = sub ? (o1 - v1) : (v1 + o1);
                v2 = sub ? (o2 - v2) : (v2 + o2);
                v3 = sub ? (o3 - v3) : (v3 + o3);
            }
            smem[lane +  0][c] = v0 * scale;
            smem[lane + 16][c] = v1 * scale;
            smem[lane + 32][c] = v2 * scale;
            smem[lane + 48][c] = v3 * scale;
        }
        __syncthreads();

        // Pass 3: coalesced row-major stores back to global memory (zero SMEM bank conflicts).
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int r = rc_base + p * 32;
            const int c = lane * 4;
            T* dp = dst + row * padded_k + base_col + r * 64 + c;
            const typename S::Vec2 out01 = S::from_floats2(smem[r][lane +  0], smem[r][lane + 16]);
            const typename S::Vec2 out23 = S::from_floats2(smem[r][lane + 32], smem[r][lane + 48]);
            if (vec_aligned) {
                uint2 packed_out;
                packed_out.x = *reinterpret_cast<const uint32_t*>(&out01);
                packed_out.y = *reinterpret_cast<const uint32_t*>(&out23);
                *reinterpret_cast<uint2*>(dp) = packed_out;
            } else {
                dp[0] = S::low(out01);
                dp[1] = S::high(out01);
                dp[2] = S::low(out23);
                dp[3] = S::high(out23);
            }
        }
        if (group + gridDim.x < groups) {
            __syncthreads();
        }
    }
}

template <typename T>
int launch_block_hadamard(const warp34_exec* ex, const void* src,
                          const int8_t* pre_signs, void* dst, int rows, int k,
                          int padded_k, int block_size) {
    if (!ex || rows < 0 || k <= 0 || padded_k < k ||
        (block_size != 16 && block_size != 64 && block_size != 128 &&
         block_size != 512 && block_size != 1024 && block_size != 4096) ||
        (pre_signs != nullptr && block_size != 512 && block_size != 1024 && block_size != 4096) ||
        k % block_size != 0 || padded_k % block_size != 0) {
        return int(cudaErrorInvalidValue);
    }
    if (rows == 0) return int(cudaSuccess);
    const uintptr_t s = reinterpret_cast<uintptr_t>(src);
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
    const uint64_t src_bytes = uint64_t(rows) * k * sizeof(T);
    const uint64_t dst_bytes = uint64_t(rows) * padded_k * sizeof(T);
    // Reject overlapping ranges, including partial aliasing: callers must not
    // destroy shared inputs. Validate arithmetic before comparing end pointers.
    if (!s || !d || s % alignof(T) || d % alignof(T) ||
        s > UINTPTR_MAX - src_bytes || d > UINTPTR_MAX - dst_bytes ||
        (s < d + dst_bytes && d < s + src_bytes)) {
        return int(cudaErrorInvalidValue);
    }
    if (pre_signs != nullptr) {
        const uintptr_t ps = reinterpret_cast<uintptr_t>(pre_signs);
        const uint64_t ps_bytes = uint64_t(k);
        if (ps > UINTPTR_MAX - ps_bytes || (ps < d + dst_bytes && d < ps + ps_bytes)) {
            return int(cudaErrorInvalidValue);
        }
    }
    const int64_t groups = int64_t(rows) * (padded_k / block_size);
    const int64_t needed = (block_size == 1024 || block_size == 4096)
        ? groups
        : ((groups + (256 / (block_size / 4)) - 1) / (256 / (block_size / 4)));
    const int grid = int(needed < 65535 ? needed : 65535);
    const auto stream = reinterpret_cast<cudaStream_t>(ex->stream);
    if (block_size == 16) {
        block_hadamard_16bit<T, 16><<<grid, 256, 0, stream>>>(
            static_cast<const T*>(src), static_cast<T*>(dst), k, padded_k, groups);
    } else if (block_size == 64) {
        block_hadamard_16bit<T, 64><<<grid, 256, 0, stream>>>(
            static_cast<const T*>(src), static_cast<T*>(dst), k, padded_k, groups);
    } else if (block_size == 128) {
        block_hadamard_16bit<T, 128><<<grid, 256, 0, stream>>>(
            static_cast<const T*>(src), static_cast<T*>(dst), k, padded_k, groups);
    } else if (block_size == 512) {
        block_hadamard_512_16bit<T><<<grid, 256, 0, stream>>>(
            static_cast<const T*>(src), pre_signs, static_cast<T*>(dst), k, padded_k, groups);
    } else if (block_size == 1024) {
        block_hadamard_1024_16bit<T><<<grid, 256, 0, stream>>>(
            static_cast<const T*>(src), pre_signs, static_cast<T*>(dst), k, padded_k, groups);
    } else {
        block_hadamard_4096_16bit<T><<<grid, 512, 0, stream>>>(
            static_cast<const T*>(src), pre_signs, static_cast<T*>(dst), k, padded_k, groups);
    }
    return int(cudaGetLastError());
}

} // namespace

extern "C" int warp34_block_hadamard_f16(const warp34_exec* ex, const void* src,
                                      void* dst, int rows, int k, int padded_k,
                                      int block_size) {
    return launch_block_hadamard<__half>(ex, src, nullptr, dst, rows, k, padded_k, block_size);
}

extern "C" int warp34_block_hadamard_signed_f16(const warp34_exec* ex, const void* src,
                                             const int8_t* pre_signs, void* dst,
                                             int rows, int k, int padded_k,
                                             int block_size) {
    return launch_block_hadamard<__half>(ex, src, pre_signs, dst, rows, k, padded_k, block_size);
}

extern "C" int warp34_block_hadamard_inplace_f16(const warp34_exec* ex, void* buf,
                                              int rows, int k, int block_size) {
    if (!ex || rows < 0 || k <= 0 ||
        (block_size != 16 && block_size != 64 && block_size != 128 &&
         block_size != 512 && block_size != 1024 && block_size != 4096) ||
        k % block_size != 0) {
        return int(cudaErrorInvalidValue);
    }
    if (rows == 0) return int(cudaSuccess);
    const uintptr_t b = reinterpret_cast<uintptr_t>(buf);
    const uint64_t bytes = uint64_t(rows) * k * sizeof(__half);
    if (!b || b % alignof(__half) || b > UINTPTR_MAX - bytes) {
        return int(cudaErrorInvalidValue);
    }
    const int64_t groups = int64_t(rows) * (k / block_size);
    const int64_t needed = (block_size == 1024 || block_size == 4096)
        ? groups
        : ((groups + (256 / (block_size / 4)) - 1) / (256 / (block_size / 4)));
    const int grid = int(needed < 65535 ? needed : 65535);
    const auto stream = reinterpret_cast<cudaStream_t>(ex->stream);
    if (block_size == 16) {
        block_hadamard_16bit<__half, 16><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(buf), static_cast<__half*>(buf), k, k, groups);
    } else if (block_size == 64) {
        block_hadamard_16bit<__half, 64><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(buf), static_cast<__half*>(buf), k, k, groups);
    } else if (block_size == 128) {
        block_hadamard_16bit<__half, 128><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(buf), static_cast<__half*>(buf), k, k, groups);
    } else if (block_size == 512) {
        block_hadamard_512_16bit<__half><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(buf), nullptr, static_cast<__half*>(buf), k, k, groups);
    } else if (block_size == 1024) {
        block_hadamard_1024_16bit<__half><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(buf), nullptr, static_cast<__half*>(buf), k, k, groups);
    } else {
        block_hadamard_4096_16bit<__half><<<grid, 512, 0, stream>>>(
            static_cast<const __half*>(buf), nullptr, static_cast<__half*>(buf), k, k, groups);
    }
    return int(cudaGetLastError());
}

extern "C" int warp34_block_hadamard_bf16(const warp34_exec* ex, const void* src,
                                       void* dst, int rows, int k, int padded_k,
                                       int block_size) {
    return launch_block_hadamard<__nv_bfloat16>(
        ex, src, nullptr, dst, rows, k, padded_k, block_size);
}

namespace {
__global__ void linear_epilogue_f16(const __half* src, const __half* bias,
                                   __half* dst, int logical_n, int physical_n,
                                   int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        const int col = int(i % logical_n);
        const __half value = src[(i / logical_n) * physical_n + col];
        dst[i] = bias ? __float2half_rn(__half2float(value) + __half2float(bias[col])) : value;
    }
}
}

extern "C" int warp34_linear_epilogue_f16(const warp34_exec* ex, const void* src,
                                       const void* bias, void* dst, int rows,
                                       int logical_n, int physical_n) {
    if (!ex || rows < 0 || logical_n <= 0 || physical_n < logical_n)
        return int(cudaErrorInvalidValue);
    if (rows == 0) return int(cudaSuccess);
    const uintptr_t s = reinterpret_cast<uintptr_t>(src);
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t b = reinterpret_cast<uintptr_t>(bias);
    const uint64_t sb = uint64_t(rows) * physical_n * sizeof(__half);
    const uint64_t db = uint64_t(rows) * logical_n * sizeof(__half);
    const uint64_t bb = uint64_t(logical_n) * sizeof(__half);
    if (!s || !d || s % 2 || d % 2 || b % 2 ||
        s > UINTPTR_MAX - sb || d > UINTPTR_MAX - db || b > UINTPTR_MAX - bb ||
        ((s < d + db && d < s + sb) && !(s == d && logical_n == physical_n)) ||
        (b && b < d + db && d < b + bb))
        return int(cudaErrorInvalidValue);
    const int64_t count = int64_t(rows) * logical_n;
    const int64_t needed = (count + 255) / 256;
    const int grid = int(needed < 65535 ? needed : 65535);
    linear_epilogue_f16<<<grid, 256, 0, reinterpret_cast<cudaStream_t>(ex->stream)>>>(
        static_cast<const __half*>(src), static_cast<const __half*>(bias),
        static_cast<__half*>(dst), logical_n, physical_n, count);
    return int(cudaGetLastError());
}
