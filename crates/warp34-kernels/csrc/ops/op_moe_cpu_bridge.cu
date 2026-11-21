// op_moe_cpu_bridge.cu — CUDA-Graph-safe GPU mailbox kernels for hybrid CPU+GPU MoE offload.
//
// Device-side half of the hybrid MoE bridge (`crates/warp34-host/src/moe_cpu_bridge.rs`):
//   1. `warp34_moe_hybrid_submit_and_pull`:
//      - `moe_hybrid_plan_and_submit_kernel` splits routed `top_k` experts per token into hot
//        VRAM hits (`0 <= slot < k_static`), optional GPU-staged cold misses (`k_static..k_hot`),
//        and CPU DDR5 cold misses, publishing the cold request to mapped pinned host memory
//        (`MoeBridgeReqSlot`) via a system-scope release store (`st.release.sys.global.b64`).
//      - `warp34_moe_pull_mapped_experts_with_count` (`op_moe_router.cu`) optionally pulls
//        GPU-staged cold misses over zero-copy PCIe when `k_hot > k_static`.
//   2. `warp34_moe_hybrid_wait_and_add`:
//      - `moe_hybrid_wait_and_add_kernel` waits on the Rust host worker pool's system-scope
//        response (`ld.acquire.sys.global.b64` on `MoeBridgeRespSlot::seq`) and adds the FP32
//        cold-expert output (`__ldcv`) into the GPU hot-expert output buffer `[num_tokens, hidden]`.
//
// All host-side thread pooling, core pinning, and AVX-512 `u4b8` expert evaluation live in Rust
// (`crates/warp34-host/src/moe_cpu_bridge.rs`).

#include "../warp34_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace warp34_moe_bridge {

static constexpr int kMaxTokens = 8;
static constexpr int kMaxTopK = 16;
static constexpr int kMaxPairs = kMaxTokens * kMaxTopK; // 128
static constexpr int kMaxHidden = 4096;
static constexpr int kMaxGpuStagePull = 8;
static constexpr int kDefaultSingleTokenMaxGpuPull = 2;
static constexpr float kAvx512RooflineGpuShareThreshold = 0.10f;
static constexpr float kHighGpuShareThreshold = 0.20f;
static constexpr float kMultiTokenDefaultGpuShare = 0.36f;

__host__ __device__ __forceinline__ int hybrid_max_gpu_pull(
    int k_static,
    int k_hot,
    int num_tokens,
    float gpu_share_frac) {
    const int max_gpu_stage = (k_hot > k_static) ? (k_hot - k_static) : 0;
    if (max_gpu_stage <= 0) {
        return 0;
    }
    if (gpu_share_frac > 0.0f && gpu_share_frac <= kAvx512RooflineGpuShareThreshold) {
        return 1;
    }
    const int cap_stage = (max_gpu_stage < kMaxGpuStagePull) ? max_gpu_stage : kMaxGpuStagePull;
    if (gpu_share_frac > kHighGpuShareThreshold || num_tokens > 1) {
        return cap_stage;
    }
    return (max_gpu_stage < kDefaultSingleTokenMaxGpuPull)
        ? max_gpu_stage
        : kDefaultSingleTokenMaxGpuPull;
}

struct alignas(64) MoeBridgeReqSlot {
    int64_t seq;
    int32_t num_tokens;
    int32_t top_k;
    int32_t _reserved0;
    int32_t num_cpu_pairs;
    int32_t hidden;
    int32_t intermediate;
    int32_t local_num_experts;
    int32_t _pad0[7];
    int32_t local_expert_ids[kMaxPairs];
    float topk_weights[kMaxPairs];
    uint16_t x_f16[kMaxTokens * kMaxHidden];
};

static_assert(offsetof(MoeBridgeReqSlot, x_f16) == 1088);
static_assert(sizeof(MoeBridgeReqSlot) == 66624);

struct alignas(64) MoeBridgeRespSlot {
    int64_t seq;
    int32_t status; // nonzero: host failure, published before seq; do not consume y
    int32_t _pad[13];
    float y_f32[kMaxTokens * kMaxHidden];
};
static_assert(offsetof(MoeBridgeRespSlot, status) == 8);
static_assert(offsetof(MoeBridgeRespSlot, y_f32) == 64);
static_assert(sizeof(MoeBridgeRespSlot) == 131136);

__global__ void moe_hybrid_plan_and_submit_kernel(
    const __half* __restrict__ x_f16,
    const int32_t* __restrict__ global_topk_ids,
    const float* __restrict__ topk_weights,
    int32_t* __restrict__ expert_to_slot,
    int32_t* __restrict__ slot_to_expert,
    uint64_t* __restrict__ slot_last_used,
    uint64_t* __restrict__ step_counter,
    uint8_t* __restrict__ expert_freq,
    int32_t* __restrict__ remapped_topk_ids,
    int32_t* __restrict__ miss_local_experts,
    int32_t* __restrict__ miss_victim_slots,
    int32_t* __restrict__ num_misses_out,
    MoeBridgeReqSlot* __restrict__ req_slot,
    int num_tokens,
    int top_k,
    int hidden,
    int intermediate,
    int global_expert_offset,
    int local_num_experts,
    int k_static,
    int k_hot,
    float gpu_share_frac,
    int async_bg_pull) {
    __shared__ int32_t s_ge[kMaxPairs];
    __shared__ int32_t s_le[kMaxPairs];
    __shared__ int32_t s_slot[kMaxPairs];
    __shared__ int32_t s_pair_remap[kMaxPairs];
    __shared__ int32_t s_cpu_le[kMaxPairs];
    __shared__ int32_t distinct_cold[kMaxPairs];
    __shared__ int32_t distinct_ge[kMaxPairs];
    __shared__ int32_t cold_hits[kMaxPairs];
    __shared__ int32_t cold_score[kMaxPairs];
    __shared__ int32_t cold_freq[kMaxPairs];
    __shared__ uint8_t cold_anchor[kMaxPairs];
    __shared__ int32_t cold_assigned_slot[kMaxPairs];
    __shared__ int32_t s_n_cpu_cold;
    __shared__ uint64_t s_next_step;
    __shared__ uint32_t s_slot_vfreq[256];
    __shared__ uint64_t s_slot_ts[256];
    __shared__ int32_t s_slot_ge[256];

    const int total_pairs = num_tokens * top_k;
    if (threadIdx.x == 0) {
        const uint64_t step = (*step_counter) + 1;
        *step_counter = step;
        s_next_step = step;
    }
    __syncthreads();
    const uint64_t next_step = s_next_step;
    if (expert_freq != nullptr && (next_step & 63ULL) == 0ULL) {
        for (int le = threadIdx.x; le < local_num_experts; le += blockDim.x) {
            expert_freq[le] >>= 1;
        }
        __syncthreads();
    }

    // Phase 1 (parallel): issue all `total_pairs` global `global_topk_ids` + `expert_to_slot`
    // lookups in a single memory wave instead of `O(T*k)` serial dependent loads in thread 0.
    if ((int)threadIdx.x < total_pairs) {
        const int p = (int)threadIdx.x;
        const int ge = global_topk_ids[p];
        const int64_t le64 = (int64_t)ge - global_expert_offset;
        s_ge[p] = ge;
        if (le64 < 0 || le64 >= local_num_experts) {
            s_le[p] = -1;
            s_slot[p] = -1;
        } else {
            const int le = (int)le64;
            s_le[p] = le;
            const int slot = expert_to_slot[le];
            s_slot[p] = (slot >= 0 && slot < k_hot) ? slot : -1;
        }
    }

    // Phase 1b (parallel): preload evictable slot metadata (`k_static..k_hot`) into shared memory
    // so multi-victim selection (`want_gpu_cold = 1..4`) runs with zero global memory stalls.
    const int n_evict = (k_hot > k_static) ? (k_hot - k_static) : 0;
    const bool cached_evict_table = (n_evict > 0 && n_evict <= 256);
    if (cached_evict_table && (int)threadIdx.x < n_evict) {
        const int idx = (int)threadIdx.x;
        const int s = k_static + idx;
        const uint64_t ts = slot_last_used[s];
        const int v_ge = slot_to_expert[s];
        uint32_t vfreq = 0u;
        if (expert_freq != nullptr) {
            const int64_t v_le = (int64_t)v_ge - global_expert_offset;
            if (v_le >= 0 && v_le < local_num_experts) {
                vfreq = (uint32_t)expert_freq[v_le];
                if (ts + 32ULL < next_step) {
                    vfreq >>= 1;
                }
            }
        }
        s_slot_ts[idx] = ts;
        s_slot_ge[idx] = v_ge;
        s_slot_vfreq[idx] = vfreq;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const uint64_t step = next_step;
        int n_cold = 0;

        for (int p = 0; p < total_pairs; ++p) {
            const int le = s_le[p];
            if (le < 0) {
                s_pair_remap[p] = -1;
                s_cpu_le[p] = -1;
                continue;
            }
            const int slot = s_slot[p];
            if (slot >= 0) {
                s_pair_remap[p] = slot;
                const uint64_t prev_ts = (cached_evict_table && slot >= k_static)
                    ? s_slot_ts[slot - k_static]
                    : slot_last_used[slot];
                if (expert_freq != nullptr && prev_ts != step) {
                    const int inc = (p < top_k) ? 2 : 1;
                    const int nf = (int)expert_freq[le] + inc;
                    expert_freq[le] = (uint8_t)((nf < 255) ? nf : 255);
                }
                if (slot >= k_static || expert_freq != nullptr) {
                    slot_last_used[slot] = step;
                    if (cached_evict_table && slot >= k_static) {
                        s_slot_ts[slot - k_static] = step;
                    }
                }
                s_cpu_le[p] = -1;
            } else {
                int c_idx = 0;
                while (c_idx < n_cold && distinct_cold[c_idx] != le) ++c_idx;
                if (c_idx == n_cold) {
                    distinct_cold[n_cold] = le;
                    distinct_ge[n_cold] = s_ge[p];
                    cold_hits[n_cold] = 0;
                    cold_score[n_cold] = 0;
                    cold_anchor[n_cold] = 0;
                    n_cold++;
                }
                cold_hits[c_idx] += 1;
                cold_score[c_idx] += (p < top_k) ? 5 : 2;
                if (p < top_k) {
                    cold_anchor[c_idx] = 1;
                }
                // Temporarily store `-2 - c_idx` to mark the cold rank `c_idx`.
                s_pair_remap[p] = -2 - c_idx;
            }
        }

        if (expert_freq != nullptr) {
            for (int c = 0; c < n_cold; ++c) {
                const int le = distinct_cold[c];
                const int inc = cold_anchor[c] ? 2 : ((cold_hits[c] >= 2) ? 2 : 1);
                const int nf = (int)expert_freq[le] + inc;
                const uint8_t clamped = (uint8_t)((nf < 255) ? nf : 255);
                expert_freq[le] = clamped;
                cold_freq[c] = (int)clamped;
            }
        } else {
            for (int c = 0; c < n_cold; ++c) {
                cold_freq[c] = cold_score[c];
            }
        }

        const int max_pull = hybrid_max_gpu_pull(k_static, k_hot, num_tokens, gpu_share_frac);
        int want_gpu_cold = 0;
        const bool use_tinylfu =
            (gpu_share_frac > 0.0f && gpu_share_frac <= kAvx512RooflineGpuShareThreshold &&
             expert_freq != nullptr);
        const bool use_parallel_pcie_split =
            (gpu_share_frac >= 0.20f && gpu_share_frac <= 0.35f && expert_freq != nullptr);
        if (max_pull > 0 && gpu_share_frac > 0.0f) {
            if (use_tinylfu) {
                if (n_cold >= 3) {
                    for (int c = 0; c < n_cold; ++c) {
                        if (cold_freq[c] >= 3 || cold_score[c] >= 5) {
                            want_gpu_cold = 1;
                            break;
                        }
                    }
                }
            } else if (use_parallel_pcie_split) {
                // Parallel CPU DDR5 (~37 GB/s) + GPU PCIe (~12.6 GB/s) cold-expert split:
                // `(n_cold - 1) * 0.25` -> 0 for n_cold<=4, 1 for 5..8, 2 for 9..12,
                // 3 for 13..16, 4 for >=17: balances `k*200us + moe_hot` against
                // `(n_cold-k)*68us` so GPU PCIe pull never outruns CPU eval.
                const int eff_cold = (n_cold > 1) ? (n_cold - 1) : 0;
                want_gpu_cold = static_cast<int>(static_cast<float>(eff_cold) * gpu_share_frac);
            } else if (gpu_share_frac <= kAvx512RooflineGpuShareThreshold) {
                int worthy_promotes = 0;
                for (int c = 0; c < n_cold; ++c) {
                    if (cold_score[c] >= 4) {
                        worthy_promotes++;
                    }
                }
                want_gpu_cold = (worthy_promotes > 0) ? 1 : 0;
            } else {
                if (n_cold >= 7) {
                    want_gpu_cold = 2;
                } else if (n_cold >= 4) {
                    want_gpu_cold = 1;
                } else {
                    for (int c = 0; c < n_cold; ++c) {
                        if (cold_hits[c] >= 2) {
                            want_gpu_cold++;
                        }
                    }
                }
                if (gpu_share_frac > kHighGpuShareThreshold || num_tokens > 1) {
                    const float eff_frac = (gpu_share_frac > kHighGpuShareThreshold)
                        ? gpu_share_frac
                        : kMultiTokenDefaultGpuShare;
                    const int scaled =
                        static_cast<int>(static_cast<float>(n_cold) * eff_frac + 0.5f);
                    if (scaled > want_gpu_cold) {
                        want_gpu_cold = scaled;
                    }
                }
            }
            if (want_gpu_cold > max_pull) want_gpu_cold = max_pull;
            if (want_gpu_cold > n_cold) want_gpu_cold = n_cold;
        }

        for (int c = 0; c < n_cold; ++c) {
            cold_assigned_slot[c] = -1;
        }

        int n_gpu_cold = 0;
        const bool freq_ranked_victims = (use_tinylfu || use_parallel_pcie_split);
        for (int m = 0; m < want_gpu_cold; ++m) {
            int best_c = -1;
            int best_rank = -1;
            for (int c = 0; c < n_cold; ++c) {
                if (cold_assigned_slot[c] != -1) continue;
                if (use_tinylfu && cold_freq[c] < 3 && cold_score[c] < 5) continue;
                const int rank = use_parallel_pcie_split
                    ? (cold_hits[c] * 16 + cold_freq[c] * 4 + (int)cold_anchor[c] * 2)
                    : (use_tinylfu ? (cold_freq[c] * 8 + cold_score[c]) : cold_score[c]);
                if (rank > best_rank) {
                    best_rank = rank;
                    best_c = c;
                }
            }
            if (best_c < 0) break;

            uint64_t best_ts = step;
            uint32_t best_vfreq = 0xffffu;
            int victim_slot = -1;
            int victim_old_ge = -1;
            if (cached_evict_table) {
                for (int idx = 0; idx < n_evict; ++idx) {
                    const uint64_t ts = s_slot_ts[idx];
                    if (ts >= step) continue;
                    const uint32_t vfreq = freq_ranked_victims ? s_slot_vfreq[idx] : 0u;
                    if (vfreq < best_vfreq || (vfreq == best_vfreq && ts < best_ts)) {
                        best_vfreq = vfreq;
                        best_ts = ts;
                        victim_slot = k_static + idx;
                        victim_old_ge = s_slot_ge[idx];
                    }
                }
            } else {
                for (int s = k_static; s < k_hot; ++s) {
                    const uint64_t ts = slot_last_used[s];
                    if (ts >= step) continue;
                    uint32_t vfreq = 0u;
                    const int v_ge = slot_to_expert[s];
                    if (freq_ranked_victims) {
                        const int64_t v_le = (int64_t)v_ge - global_expert_offset;
                        if (v_le >= 0 && v_le < local_num_experts) {
                            vfreq = (uint32_t)expert_freq[v_le];
                            if (ts + 32ULL < step) {
                                vfreq >>= 1;
                            }
                        }
                    }
                    if (vfreq < best_vfreq || (vfreq == best_vfreq && ts < best_ts)) {
                        best_vfreq = vfreq;
                        best_ts = ts;
                        victim_slot = s;
                        victim_old_ge = v_ge;
                    }
                }
            }
            if (victim_slot < 0) break;
            if (use_tinylfu) {
                const uint32_t hyst = (best_ts + 32ULL < step) ? 1u : 2u;
                const uint32_t cand_eff = (cold_score[best_c] >= 5 && cold_freq[best_c] < 3)
                    ? 3u
                    : (uint32_t)cold_freq[best_c];
                if (cand_eff < best_vfreq + hyst) break;
            }

            const int le = distinct_cold[best_c];
            const int ge = distinct_ge[best_c];
            const int64_t old_le = (int64_t)victim_old_ge - global_expert_offset;
            if (old_le >= 0 && old_le < local_num_experts) {
                expert_to_slot[old_le] = -1;
            }
            expert_to_slot[le] = victim_slot;
            slot_to_expert[victim_slot] = ge;
            slot_last_used[victim_slot] = step;
            if (cached_evict_table) {
                const int idx = victim_slot - k_static;
                s_slot_ts[idx] = step;
                s_slot_ge[idx] = ge;
            }
            if (freq_ranked_victims) {
                const uint32_t min_prom_freq =
                    (best_vfreq + 2u < 255u) ? (best_vfreq + 2u) : 255u;
                if ((uint32_t)expert_freq[le] < min_prom_freq) {
                    expert_freq[le] = (uint8_t)min_prom_freq;
                }
            }
            cold_assigned_slot[best_c] = async_bg_pull ? -2 : victim_slot;
            miss_local_experts[n_gpu_cold] = le;
            miss_victim_slots[n_gpu_cold] = victim_slot;
            n_gpu_cold++;
        }

        const int n_cpu_cold = async_bg_pull ? n_cold : (n_cold - n_gpu_cold);
        num_misses_out[0] = n_gpu_cold;
        num_misses_out[1] = n_cpu_cold;
        s_n_cpu_cold = n_cpu_cold;

        if (n_cpu_cold > 0) {
            req_slot->num_tokens = num_tokens;
            req_slot->top_k = top_k;
            req_slot->num_cpu_pairs = total_pairs;
            req_slot->hidden = hidden;
            req_slot->intermediate = intermediate;
            req_slot->local_num_experts = local_num_experts;
        }
    }
    __syncthreads();

    // Phase 3 (parallel): write `remapped_topk_ids` and `s_cpu_le` across all `total_pairs` threads.
    if ((int)threadIdx.x < total_pairs) {
        const int p = (int)threadIdx.x;
        const int r = s_pair_remap[p];
        if (r <= -2) {
            const int c_idx = -2 - r;
            const int v_slot = cold_assigned_slot[c_idx];
            if (v_slot >= 0) {
                remapped_topk_ids[p] = v_slot;
                s_cpu_le[p] = -1;
            } else {
                remapped_topk_ids[p] = -1;
                s_cpu_le[p] = distinct_cold[c_idx];
            }
        } else {
            remapped_topk_ids[p] = r;
        }
    }
    __syncthreads();

    if (s_n_cpu_cold <= 0) return;

    for (int p = threadIdx.x; p < kMaxPairs; p += blockDim.x) {
        if (p < total_pairs) {
            const int32_t cle = s_cpu_le[p];
            req_slot->local_expert_ids[p] = cle;
            req_slot->topk_weights[p] = (cle >= 0) ? topk_weights[p] : 0.0f;
        } else {
            req_slot->local_expert_ids[p] = -1;
            req_slot->topk_weights[p] = 0.0f;
        }
    }

    const int total_elems = num_tokens * hidden;
    if (((total_elems & 7) == 0) && ((((uintptr_t)x_f16 | (uintptr_t)req_slot->x_f16) & 15u) == 0)) {
        const int vec_elems = total_elems >> 3;
        const uint4* src_v = reinterpret_cast<const uint4*>(x_f16);
        uint4* dst_v = reinterpret_cast<uint4*>(req_slot->x_f16);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
            dst_v[i] = src_v[i];
        }
    } else {
        const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(x_f16);
        for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
            req_slot->x_f16[i] = src_u16[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence_system();
        const int64_t next_seq = req_slot->seq + 1;
        asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(&req_slot->seq), "l"(next_seq)
                     : "memory");
    }
}

__global__ void moe_hybrid_wait_and_add_kernel(
    __half* __restrict__ out_f16,
    const int32_t* __restrict__ num_misses_buf,
    const MoeBridgeReqSlot* __restrict__ req_slot,
    const MoeBridgeRespSlot* __restrict__ resp_slot,
    int total_elems) {
    if (num_misses_buf[1] <= 0) return;

    __shared__ int failed;
    if (threadIdx.x == 0) {
        const int64_t expected = *reinterpret_cast<const volatile int64_t*>(&req_slot->seq);
        int64_t v = 0;
        for (;;) {
            asm volatile("ld.acquire.sys.global.b64 %0, [%1];"
                         : "=l"(v)
                         : "l"(&resp_slot->seq)
                         : "memory");
            if (v == expected) break;
            __nanosleep(128);
        }
        failed = __ldcv(&resp_slot->status);
    }
    __syncthreads();
    // The Rust stream/event synchronization reports the sticky host error before
    // outputs can be consumed. Release the graph rather than waiting forever or
    // adding stale mailbox data. Its model state must not be reused after failure.
    if (failed) return;

    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * i4 < total_elems) {
        const float4 y = __ldcv(reinterpret_cast<const float4*>(resp_slot->y_f32) + i4);
        __half* o = out_f16 + 4 * i4;
        o[0] = __float2half(__half2float(o[0]) + y.x);
        o[1] = __float2half(__half2float(o[1]) + y.y);
        o[2] = __float2half(__half2float(o[2]) + y.z);
        o[3] = __float2half(__half2float(o[3]) + y.w);
    }
}

} // namespace warp34_moe_bridge

extern "C" int warp34_moe_hybrid_submit_and_pull_forked(
    const warp34_exec* ex,
    void* d_req_slot,
    const void* x_f16,
    const int32_t* global_topk_ids,
    const float* topk_weights,
    int32_t* expert_to_slot,
    int32_t* slot_to_expert,
    uint64_t* slot_last_used,
    uint64_t* step_counter,
    uint8_t* expert_freq,
    int32_t* remapped_topk_ids,
    int32_t* miss_local_experts,
    int32_t* miss_victim_slots,
    int32_t* num_misses_buf,
    const void* mapped_host_packs,
    uint32_t* gate_up_packed_base,
    void* gate_up_scales_base,
    uint32_t* down_packed_base,
    void* down_scales_base,
    int num_tokens,
    int top_k,
    int global_expert_offset,
    int local_num_experts,
    int k_static,
    int k_hot,
    size_t pack_stride_bytes,
    int n_interm,
    int k_hidden,
    int group_size,
    int bits,
    float gpu_share_frac,
    void* pull_stream_raw,
    void* plan_event_raw,
    void* pull_done_event_raw) {
    if (!ex || !d_req_slot || !x_f16 || !global_topk_ids || !topk_weights ||
        !expert_to_slot || !slot_to_expert || !slot_last_used || !step_counter ||
        !remapped_topk_ids || !miss_local_experts || !miss_victim_slots ||
        !num_misses_buf || !gate_up_packed_base || !gate_up_scales_base ||
        !down_packed_base || !down_scales_base || num_tokens <= 0 ||
        num_tokens > warp34_moe_bridge::kMaxTokens || top_k <= 0 ||
        top_k > warp34_moe_bridge::kMaxTopK || k_static <= 0 || k_hot < k_static || k_hidden <= 0 ||
        k_hidden > warp34_moe_bridge::kMaxHidden) {
        return (int)cudaErrorInvalidValue;
    }

    const float effective_gpu_share = (mapped_host_packs != nullptr) ? gpu_share_frac : 0.0f;
    const int max_pull = warp34_moe_bridge::hybrid_max_gpu_pull(
        k_static, k_hot, num_tokens, effective_gpu_share);
    const bool can_pull = (max_pull > 0 && mapped_host_packs != nullptr && effective_gpu_share > 0.0f);
    const bool use_async_fork =
        (can_pull && pull_stream_raw != nullptr && plan_event_raw != nullptr &&
         pull_done_event_raw != nullptr && expert_freq != nullptr &&
         effective_gpu_share <= warp34_moe_bridge::kAvx512RooflineGpuShareThreshold);

    if (can_pull) {
        const int pack_factor = (bits == 2) ? 16 : 8;
        if (n_interm <= 0 || (bits != 2 && bits != 4) ||
            (((uintptr_t)mapped_host_packs | (uintptr_t)gate_up_packed_base |
              (uintptr_t)gate_up_scales_base | (uintptr_t)down_packed_base |
              (uintptr_t)down_scales_base) & 15) != 0 ||
            group_size <= 0 || max_pull > 65535 ||
            n_interm > INT_MAX / 2 || n_interm % pack_factor != 0 || k_hidden % pack_factor != 0 ||
            n_interm % group_size != 0 || k_hidden % group_size != 0 ||
            (pack_stride_bytes & 15) != 0) {
            return (int)cudaErrorInvalidValue;
        }
        const size_t gu_packed_bytes =
            (size_t)(2 * n_interm) * (size_t)(k_hidden / pack_factor) * sizeof(uint32_t);
        const size_t gu_scales_bytes =
            (size_t)(2 * n_interm) * (size_t)(k_hidden / group_size) * sizeof(uint16_t);
        const size_t dn_packed_bytes =
            (size_t)k_hidden * (size_t)(n_interm / pack_factor) * sizeof(uint32_t);
        const size_t dn_scales_bytes =
            (size_t)k_hidden * (size_t)(n_interm / group_size) * sizeof(uint16_t);
        if ((gu_packed_bytes & 15) != 0 || (gu_scales_bytes & 15) != 0 ||
            (dn_packed_bytes & 15) != 0 || (dn_scales_bytes & 15) != 0 ||
            pack_stride_bytes < gu_packed_bytes + gu_scales_bytes + dn_packed_bytes + dn_scales_bytes) {
            return (int)cudaErrorInvalidValue;
        }
    }
    cudaStream_t stream = (cudaStream_t)ex->stream;
    warp34_moe_bridge::moe_hybrid_plan_and_submit_kernel<<<1, 256, 0, stream>>>(
        static_cast<const __half*>(x_f16),
        global_topk_ids,
        topk_weights,
        expert_to_slot,
        slot_to_expert,
        slot_last_used,
        step_counter,
        expert_freq,
        remapped_topk_ids,
        miss_local_experts,
        miss_victim_slots,
        num_misses_buf,
        static_cast<warp34_moe_bridge::MoeBridgeReqSlot*>(d_req_slot),
        num_tokens,
        top_k,
        k_hidden,
        n_interm,
        global_expert_offset,
        local_num_experts,
        k_static,
        k_hot,
        effective_gpu_share,
        use_async_fork ? 1 : 0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    if (can_pull) {
        if (use_async_fork) {
            cudaStream_t pull_stream = (cudaStream_t)pull_stream_raw;
            cudaEvent_t plan_ev = (cudaEvent_t)plan_event_raw;
            cudaEvent_t pull_done_ev = (cudaEvent_t)pull_done_event_raw;
            err = cudaEventRecord(plan_ev, stream);
            if (err != cudaSuccess) return (int)err;
            err = cudaStreamWaitEvent(pull_stream, plan_ev, 0);
            if (err != cudaSuccess) return (int)err;
            warp34_exec pull_ex = *ex;
            pull_ex.stream = reinterpret_cast<uintptr_t>(pull_stream_raw);
            const int rc = warp34_moe_pull_mapped_experts_with_count(
                &pull_ex,
                mapped_host_packs,
                miss_local_experts,
                miss_victim_slots,
                num_misses_buf,
                gate_up_packed_base,
                gate_up_scales_base,
                down_packed_base,
                down_scales_base,
                max_pull,
                local_num_experts,
                k_hot,
                pack_stride_bytes,
                n_interm,
                k_hidden,
                group_size,
                bits);
            if (rc != 0) return rc;
            err = cudaEventRecord(pull_done_ev, pull_stream);
            if (err != cudaSuccess) return (int)err;
            return 0;
        }
        return warp34_moe_pull_mapped_experts_with_count(
            ex,
            mapped_host_packs,
            miss_local_experts,
            miss_victim_slots,
            num_misses_buf,
            gate_up_packed_base,
            gate_up_scales_base,
            down_packed_base,
            down_scales_base,
            max_pull,
            local_num_experts,
            k_hot,
            pack_stride_bytes,
            n_interm,
            k_hidden,
            group_size,
            bits);
    }

    return 0;
}

extern "C" int warp34_moe_hybrid_submit_and_pull(
    const warp34_exec* ex,
    void* d_req_slot,
    const void* x_f16,
    const int32_t* global_topk_ids,
    const float* topk_weights,
    int32_t* expert_to_slot,
    int32_t* slot_to_expert,
    uint64_t* slot_last_used,
    uint64_t* step_counter,
    uint8_t* expert_freq,
    int32_t* remapped_topk_ids,
    int32_t* miss_local_experts,
    int32_t* miss_victim_slots,
    int32_t* num_misses_buf,
    const void* mapped_host_packs,
    uint32_t* gate_up_packed_base,
    void* gate_up_scales_base,
    uint32_t* down_packed_base,
    void* down_scales_base,
    int num_tokens,
    int top_k,
    int global_expert_offset,
    int local_num_experts,
    int k_static,
    int k_hot,
    size_t pack_stride_bytes,
    int n_interm,
    int k_hidden,
    int group_size,
    int bits,
    float gpu_share_frac) {
    return warp34_moe_hybrid_submit_and_pull_forked(
        ex,
        d_req_slot,
        x_f16,
        global_topk_ids,
        topk_weights,
        expert_to_slot,
        slot_to_expert,
        slot_last_used,
        step_counter,
        expert_freq,
        remapped_topk_ids,
        miss_local_experts,
        miss_victim_slots,
        num_misses_buf,
        mapped_host_packs,
        gate_up_packed_base,
        gate_up_scales_base,
        down_packed_base,
        down_scales_base,
        num_tokens,
        top_k,
        global_expert_offset,
        local_num_experts,
        k_static,
        k_hot,
        pack_stride_bytes,
        n_interm,
        k_hidden,
        group_size,
        bits,
        gpu_share_frac,
        nullptr,
        nullptr,
        nullptr);
}

extern "C" int warp34_moe_hybrid_wait_and_add_joined(
    const warp34_exec* ex,
    const void* d_req_slot,
    const void* d_resp_slot,
    void* out_f16,
    const int32_t* num_misses_buf,
    int num_tokens,
    int k_hidden,
    void* pull_done_event_raw) {
    if (!ex || !d_req_slot || !d_resp_slot || !out_f16 || !num_misses_buf || num_tokens <= 0 ||
        num_tokens > warp34_moe_bridge::kMaxTokens || k_hidden <= 0 ||
        k_hidden > warp34_moe_bridge::kMaxHidden || (k_hidden & 3) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    cudaStream_t stream = (cudaStream_t)ex->stream;
    const int total_elems = num_tokens * k_hidden;
    const int n4 = total_elems / 4;
    const int blocks = (n4 + 255) / 256;
    warp34_moe_bridge::moe_hybrid_wait_and_add_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<__half*>(out_f16),
        num_misses_buf,
        static_cast<const warp34_moe_bridge::MoeBridgeReqSlot*>(d_req_slot),
        static_cast<const warp34_moe_bridge::MoeBridgeRespSlot*>(d_resp_slot),
        total_elems);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    // `moe_hybrid_wait_and_add_kernel` only consumes the CPU bridge response (`d_resp_slot`),
    // while `pull_stream` writes to a victim slot masked out of this step's GPU MoE (`async_bg_pull`).
    // Joining `pull_done_event_raw` after the kernel lets CPU wait and PCIe pull overlap while still
    // guaranteeing the pull completes before any subsequent layer/round reads the slot or reuses the event.
    if (pull_done_event_raw != nullptr) {
        err = cudaStreamWaitEvent(stream, (cudaEvent_t)pull_done_event_raw, 0);
        if (err != cudaSuccess) return (int)err;
    }
    return 0;
}

extern "C" int warp34_moe_hybrid_wait_and_add(
    const warp34_exec* ex,
    const void* d_req_slot,
    const void* d_resp_slot,
    void* out_f16,
    const int32_t* num_misses_buf,
    int num_tokens,
    int k_hidden) {
    return warp34_moe_hybrid_wait_and_add_joined(
        ex, d_req_slot, d_resp_slot, out_f16, num_misses_buf, num_tokens, k_hidden, nullptr);
}
