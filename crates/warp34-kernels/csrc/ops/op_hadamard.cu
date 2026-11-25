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
template <typename T>
__global__ void block_hadamard_512_16bit(const T* src, T* dst,
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
    const bool sub7 = (warp_in_grp & 1) != 0;
    const bool sub8 = (warp_in_grp & 2) != 0;

    const int64_t num_cta_iters = (groups + 1) / 2;
    for (int64_t iter = blockIdx.x; iter < num_cta_iters; iter += gridDim.x) {
        const int64_t group = iter * 2 + grp_in_cta;
        const bool active = group < groups;
        int64_t row = 0;
        int col = 0;

        if (active) {
            row = group / blocks_per_row;
            col = int(group % blocks_per_row) * 512 + warp_in_grp * 128 + lane * 4;
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

        __syncthreads();
    }
}

// H_4096 factored as H_64 ⊗ H_64 in shared memory (`64 × 65` float = 16.25 KiB, zero bank
// conflicts on transpose). Supports optional per-channel `pre_signs` (`{-1, +1}` in `int8_t`,
// length 4096) fused into the load so `v_o_proj` (`H_4096(x * signs)`) takes a single pass.
// One CTA (256 threads = 16 half-warps × 16 lanes) handles one 4096-element block per iteration.
template <typename T>
__global__ void block_hadamard_4096_16bit(const T* src, const int8_t* pre_signs, T* dst,
                                          int k, int padded_k, int64_t groups) {
    using S = HadScalar<T>;
    constexpr float scale = 0.015625f; // 1 / sqrt(4096) = 1 / 64
    __shared__ float smem[64][65];

    const int hw = threadIdx.x >> 4;   // 0..15 half-warp index
    const int lane = threadIdx.x & 15; // 0..15 lane within half-warp
    const int blocks_per_row = padded_k / 4096;
    const bool vec_aligned =
        ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 7u) == 0u;

    for (int64_t group = blockIdx.x; group < groups; group += gridDim.x) {
        const int64_t row = group / blocks_per_row;
        const int base_col = int(group % blocks_per_row) * 4096;

        // Pass 1: H_64 along each of the 64 rows (low 6 bits: h = 1..32).
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int r = hw + p * 16;
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
                    v0 *= float(pre_signs[in_blk + 0]);
                    v1 *= float(pre_signs[in_blk + 1]);
                    v2 *= float(pre_signs[in_blk + 2]);
                    v3 *= float(pre_signs[in_blk + 3]);
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
            smem[r][c + 0] = v0;
            smem[r][c + 1] = v1;
            smem[r][c + 2] = v2;
            smem[r][c + 3] = v3;
        }
        __syncthreads();

        // Pass 2: H_64 along each of the 64 columns (high 6 bits: h = 64..2048), scaled by 1/64.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int c = hw + p * 16;
            const int r = lane * 4;
            float v0 = smem[r + 0][c];
            float v1 = smem[r + 1][c];
            float v2 = smem[r + 2][c];
            float v3 = smem[r + 3][c];
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
            smem[r + 0][c] = v0 * scale;
            smem[r + 1][c] = v1 * scale;
            smem[r + 2][c] = v2 * scale;
            smem[r + 3][c] = v3 * scale;
        }
        __syncthreads();

        // Pass 3: coalesced row-major stores back to global memory.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int r = hw + p * 16;
            const int c = lane * 4;
            T* dp = dst + row * padded_k + base_col + r * 64 + c;
            const typename S::Vec2 out01 = S::from_floats2(smem[r][c + 0], smem[r][c + 1]);
            const typename S::Vec2 out23 = S::from_floats2(smem[r][c + 2], smem[r][c + 3]);
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
        __syncthreads();
    }
}

template <typename T>
int launch_block_hadamard(const warp34_exec* ex, const void* src,
                          const int8_t* pre_signs, void* dst, int rows, int k,
                          int padded_k, int block_size) {
    if (!ex || rows < 0 || k <= 0 || padded_k < k ||
        (block_size != 16 && block_size != 64 && block_size != 128 &&
         block_size != 512 && block_size != 4096) ||
        (pre_signs != nullptr && block_size != 4096) ||
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
    const int64_t groups = int64_t(rows) * (padded_k / block_size);
    const int64_t needed = (block_size == 4096)
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
            static_cast<const T*>(src), static_cast<T*>(dst), k, padded_k, groups);
    } else {
        block_hadamard_4096_16bit<T><<<grid, 256, 0, stream>>>(
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
         block_size != 512 && block_size != 4096) ||
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
    const int64_t needed = (block_size == 4096)
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
            static_cast<const __half*>(buf), static_cast<__half*>(buf), k, k, groups);
    } else {
        block_hadamard_4096_16bit<__half><<<grid, 256, 0, stream>>>(
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
