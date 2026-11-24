#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace warp34 {

// In-register H_64 butterfly over a 16-lane subgroup (4 contiguous __half values per lane).
// Preserves exact bit-order FP32 summation and * 0.125f RN rounding matching warp34_block_hadamard_f16.
__device__ __forceinline__ uint2 hadamard64_vec4_f16(uint2 raw, int lane = (int)(threadIdx.x & 31)) {
    const __half2 h01 = *reinterpret_cast<const __half2*>(&raw.x);
    const __half2 h23 = *reinterpret_cast<const __half2*>(&raw.y);
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    float v0 = f01.x, v1 = f01.y, v2 = f23.x, v3 = f23.y;

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

    const unsigned mask = __activemask();
    #pragma unroll
    for (int h = 1; h < 16; h <<= 1) {
        const float o0 = __shfl_xor_sync(mask, v0, h, 16);
        const float o1 = __shfl_xor_sync(mask, v1, h, 16);
        const float o2 = __shfl_xor_sync(mask, v2, h, 16);
        const float o3 = __shfl_xor_sync(mask, v3, h, 16);
        const bool sub = (lane & h) != 0;
        v0 = sub ? (o0 - v0) : (v0 + o0);
        v1 = sub ? (o1 - v1) : (v1 + o1);
        v2 = sub ? (o2 - v2) : (v2 + o2);
        v3 = sub ? (o3 - v3) : (v3 + o3);
    }

    const __half2 out01 = __floats2half2_rn(v0 * 0.125f, v1 * 0.125f);
    const __half2 out23 = __floats2half2_rn(v2 * 0.125f, v3 * 0.125f);
    uint2 out;
    out.x = *reinterpret_cast<const uint32_t*>(&out01);
    out.y = *reinterpret_cast<const uint32_t*>(&out23);
    return out;
}

// In-register H_64 butterfly over an 8-lane subgroup (8 contiguous __half values per lane).
// Stages h=1,2,4 execute within each lane's registers; stages h=8,16,32 execute via 3 warp
// shuffles across the 8-lane subgroup. 0-ULP identical to hadamard64_vec4_f16.
__device__ __forceinline__ uint4 hadamard64_vec8_f16(uint4 raw, int lane) {
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
    const float2 f01 = __half22float2(h2[0]);
    const float2 f23 = __half22float2(h2[1]);
    const float2 f45 = __half22float2(h2[2]);
    const float2 f67 = __half22float2(h2[3]);
    float v[8] = {f01.x, f01.y, f23.x, f23.y, f45.x, f45.y, f67.x, f67.y};

    // Stage h=1, 2, 4 (H_8 in-register butterfly)
    {
        const float a0 = v[0] + v[1], a1 = v[0] - v[1];
        const float a2 = v[2] + v[3], a3 = v[2] - v[3];
        const float a4 = v[4] + v[5], a5 = v[4] - v[5];
        const float a6 = v[6] + v[7], a7 = v[6] - v[7];

        const float b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
        const float b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;

        v[0] = b0 + b4;
        v[1] = b1 + b5;
        v[2] = b2 + b6;
        v[3] = b3 + b7;
        v[4] = b0 - b4;
        v[5] = b1 - b5;
        v[6] = b2 - b6;
        v[7] = b3 - b7;
    }

    // Stage h=8, 16, 32 across the 8-lane subgroup
    const unsigned mask = __activemask();
    #pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
        const bool sub = (lane & d) != 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float o = __shfl_xor_sync(mask, v[j], d, 8);
            v[j] = sub ? (o - v[j]) : (v[j] + o);
        }
    }

    uint4 out;
    __half2* out_h2 = reinterpret_cast<__half2*>(&out);
    out_h2[0] = __floats2half2_rn(v[0] * 0.125f, v[1] * 0.125f);
    out_h2[1] = __floats2half2_rn(v[2] * 0.125f, v[3] * 0.125f);
    out_h2[2] = __floats2half2_rn(v[4] * 0.125f, v[5] * 0.125f);
    out_h2[3] = __floats2half2_rn(v[6] * 0.125f, v[7] * 0.125f);
    return out;
}

// Block-cooperative H_512 butterfly over a 64-lane (2-warp) subgroup (8 contiguous __half values
// per lane). Stages h=1,2,4 execute in registers, h=8..128 via full-warp shuffles, and h=256
// across the 2 warps via shared memory (`smem[num_warps][32][8]`). All threads in the CTA must
// reach this helper (pass `active = false` for trailing out-of-range warps) so `__syncthreads()`
// is uniform and race-free. 0-ULP bit-identical to `block_hadamard_512_16bit`.
__device__ __forceinline__ uint4 hadamard512_vec8_f16(
    uint4 raw, bool active, float (*smem)[32][8]) {
    constexpr float kScale512 = 0.04419417382415922f;  // 1 / sqrt(512)
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    float v[8];

    if (active) {
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
        const float2 f01 = __half22float2(h2[0]);
        const float2 f23 = __half22float2(h2[1]);
        const float2 f45 = __half22float2(h2[2]);
        const float2 f67 = __half22float2(h2[3]);
        v[0] = f01.x; v[1] = f01.y; v[2] = f23.x; v[3] = f23.y;
        v[4] = f45.x; v[5] = f45.y; v[6] = f67.x; v[7] = f67.y;

        // Stage h=1, 2, 4 (H_8 in-register butterfly)
        {
            const float a0 = v[0] + v[1], a1 = v[0] - v[1];
            const float a2 = v[2] + v[3], a3 = v[2] - v[3];
            const float a4 = v[4] + v[5], a5 = v[4] - v[5];
            const float a6 = v[6] + v[7], a7 = v[6] - v[7];

            const float b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
            const float b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;

            v[0] = b0 + b4; v[1] = b1 + b5; v[2] = b2 + b6; v[3] = b3 + b7;
            v[4] = b0 - b4; v[5] = b1 - b5; v[6] = b2 - b6; v[7] = b3 - b7;
        }

        // Stage h=8, 16, 32, 64, 128 across all 32 warp lanes (H_32)
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const bool sub = (lane & d) != 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float o = __shfl_xor_sync(0xffffffffu, v[j], d, 32);
                v[j] = sub ? (o - v[j]) : (v[j] + o);
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            smem[warp][lane][j] = v[j];
        }
    }
    __syncthreads();

    uint4 out = make_uint4(0u, 0u, 0u, 0u);
    if (active) {
        const int peer_warp = warp ^ 1;
        const bool sub8 = (warp & 1) != 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float o = smem[peer_warp][lane][j];
            v[j] = sub8 ? (o - v[j]) : (v[j] + o);
        }
        __half2* out_h2 = reinterpret_cast<__half2*>(&out);
        out_h2[0] = __floats2half2_rn(v[0] * kScale512, v[1] * kScale512);
        out_h2[1] = __floats2half2_rn(v[2] * kScale512, v[3] * kScale512);
        out_h2[2] = __floats2half2_rn(v[4] * kScale512, v[5] * kScale512);
        out_h2[3] = __floats2half2_rn(v[6] * kScale512, v[7] * kScale512);
    }
    __syncthreads();
    return out;
}

}  // namespace warp34
