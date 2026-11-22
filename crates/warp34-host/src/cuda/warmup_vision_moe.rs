use super::*;
use core::ffi::c_void;
use warp34_kernels::ffi::{
    self, Warp34AttentionBatch, Warp34Exec, Warp34GdnBatch, Warp34KvLayerLayout,
};

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
pub(super) fn warmup_vision_moe_and_qwen4(
    arena: &mut DeviceArena,
    stream: &CudaStream,
    ex: &Warp34Exec,
    buf_a: u64,
    buf_b: u64,
    buf_c: u64,
    buf_d: u64,
    buf_ws: u64,
    zero_i32: u64,
    reject_ids: u64,
    reject_probs: u64,
    attn_batch: &Warp34AttentionBatch,
    layout_256: &Warp34KvLayerLayout,
    gdn_batch: &Warp34GdnBatch,
) -> Result<()> {
    macro_rules! warmup_ffi {
        ($call:expr, $label:expr) => {{
            // SAFETY: Warmup arena and fixture buffers are sized for this launch and remain live until `dev.synchronize()`.
            check_rc(unsafe { $call }, $label)
        }};
    }

    // Warm Vision & MoE kernels so CUDA 13 lazy loading
    // resolves all module symbols before any TP=2 collective or graph capture.
    let cu_seqlens = arena.upload_slice(&[0i32, 1i32], stream)?;
    warmup_ffi!(
        ffi::warp34_vision_bias_add_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_a as *const c_void,
            buf_c as *mut c_void,
            1,
            64,
        ),
        "warmup warp34_vision_bias_add_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_vision_layernorm_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            1,
            64,
            1e-6,
        ),
        "warmup warp34_vision_layernorm_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_vision_bias_gelu_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            1,
            64,
            1,
        ),
        "warmup warp34_vision_bias_gelu_f16"
    )?;
    let ex_vision = stream.exec(buf_ws, 64 * 1024);
    for head_dim in [64, 72] {
        warmup_ffi!(
            ffi::warp34_vision_qkv_rope_attn_f16(
                &ex_vision,
                buf_a as *mut c_void,
                buf_b as *const c_void,
                reject_ids as *const i32,
                cu_seqlens as *const i32,
                buf_c as *mut c_void,
                1,
                1,
                2,
                head_dim,
                10000.0,
            ),
            "warmup warp34_vision_qkv_rope_attn_f16"
        )?;
    }
    warmup_ffi!(
        ffi::warp34_embedding_splice_f16(
            ex,
            buf_c as *mut c_void,
            buf_a as *const c_void,
            reject_ids as *const i32,
            1,
            64,
        ),
        "warmup warp34_embedding_splice_f16"
    )?;
    for (n_exp, top_k) in [(8i32, 2i32), (256, 8), (512, 2)] {
        warmup_ffi!(
            ffi::warp34_moe_topk_router_f16(
                ex,
                buf_a as *const c_void,
                buf_b as *mut f32,
                buf_d as *mut i32,
                1,
                n_exp,
                top_k,
                1,
                0,
            ),
            "warmup warp34_moe_topk_router_f16"
        )?;
    }
    warmup_ffi!(
        ffi::warp34_moe_w4a16_gate_up_silu_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const u32,
            buf_a as *const c_void,
            reject_ids as *const i32,
            buf_c as *mut c_void,
            1,
            1,
            2,
            8,
            64,
            64,
        ),
        "warmup warp34_moe_w4a16_gate_up_silu_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_moe_w2a16_gate_up_silu_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const u32,
            buf_a as *const c_void,
            reject_ids as *const i32,
            buf_c as *mut c_void,
            1,
            1,
            2,
            16,
            64,
            64,
        ),
        "warmup warp34_moe_w2a16_gate_up_silu_f16"
    )?;
    for (top_k, n_interm) in [
        (1i32, 64i32),
        (8, 256),
        (8, 512),
        (10, 256),
        (10, 512),
        (10, 640),
    ] {
        warmup_ffi!(
            ffi::warp34_moe_w4a16_down_accum_f16(
                ex,
                buf_c as *const c_void,
                buf_b as *const u32,
                buf_a as *const c_void,
                zero_i32 as *const i32,
                buf_a as *const f32,
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_c as *mut c_void,
                1,
                top_k,
                2,
                64,
                n_interm,
                64,
            ),
            "warmup warp34_moe_w4a16_down_accum_f16"
        )?;
        warmup_ffi!(
            ffi::warp34_moe_w2a16_down_accum_f16(
                ex,
                buf_c as *const c_void,
                buf_b as *const u32,
                buf_a as *const c_void,
                zero_i32 as *const i32,
                buf_a as *const f32,
                buf_a as *const c_void,
                buf_a as *const c_void,
                buf_c as *mut c_void,
                1,
                top_k,
                2,
                64,
                n_interm,
                64,
            ),
            "warmup warp34_moe_w2a16_down_accum_f16"
        )?;
    }
    let moe_ids_m4 = arena.upload_slice(&[0i32, 1, 0, 1], stream)?;
    let moe_w_m4 = arena.upload_slice(&[1.0f32; 4], stream)?;
    let moe_ws = arena.alloc_aligned(64 * 1024)?;
    let ex_moe = stream.exec(moe_ws, 64 * 1024);
    for (m_tok, n_exp) in [(1i32, 2i32), (4, 8), (4, 2)] {
        for down_had in [0i32, 64] {
            warmup_ffi!(
                ffi::warp34_moe_w4a16_mlp_ex_f16(
                    &ex_moe,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    moe_ids_m4 as *const i32,
                    moe_w_m4 as *const f32,
                    buf_a as *const c_void,
                    buf_a as *const c_void,
                    buf_c as *mut c_void,
                    buf_d as *mut c_void,
                    m_tok,
                    1,
                    n_exp,
                    64,
                    64,
                    64,
                    down_had,
                ),
                "warmup warp34_moe_w4a16_mlp_ex_f16"
            )?;
            warmup_ffi!(
                ffi::warp34_moe_w2a16_mlp_ex_f16(
                    &ex_moe,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    moe_ids_m4 as *const i32,
                    moe_w_m4 as *const f32,
                    buf_a as *const c_void,
                    buf_a as *const c_void,
                    buf_c as *mut c_void,
                    buf_d as *mut c_void,
                    m_tok,
                    1,
                    n_exp,
                    64,
                    64,
                    64,
                    down_had,
                ),
                "warmup warp34_moe_w2a16_mlp_ex_f16"
            )?;
            warmup_ffi!(
                ffi::warp34_moe_w4a16_mlp_zp_ex_f16(
                    &ex_moe,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    buf_b as *const u32,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    moe_ids_m4 as *const i32,
                    moe_w_m4 as *const f32,
                    buf_a as *const c_void,
                    buf_a as *const c_void,
                    buf_c as *mut c_void,
                    buf_d as *mut c_void,
                    m_tok,
                    1,
                    n_exp,
                    64,
                    64,
                    64,
                    down_had,
                ),
                "warmup warp34_moe_w4a16_mlp_zp_ex_f16"
            )?;
        }
    }
    // Sparse M4 (BLOCK_M=4 vec4) and M8..128 (BLOCK_M=8 MMA) g64/g128 dispatches have
    // distinct sorting and kernel specializations. Warm both on each rank before
    // any TP collective capture or replay.
    let moe_ids_m8 = arena.upload_slice(&[0i32, 0, 2, 3, 4, 5, 6, 7], stream)?;
    let moe_w_m8 = arena.upload_slice(&[1.0f32; 8], stream)?;
    let moe_sparse_weights = arena.alloc_aligned(16 * 256 * 128 / 2)?;
    for m_sparse in [4i32, 8] {
        for group_size in [64i32, 128] {
            for down_had in [0i32, 64] {
                warmup_ffi!(
                    ffi::warp34_moe_w4a16_mlp_ex_f16(
                        &ex_moe,
                        buf_a as *const c_void,
                        moe_sparse_weights as *const u32,
                        buf_a as *const c_void,
                        moe_sparse_weights as *const u32,
                        buf_a as *const c_void,
                        moe_ids_m8 as *const i32,
                        moe_w_m8 as *const f32,
                        core::ptr::null(),
                        core::ptr::null(),
                        buf_c as *mut c_void,
                        buf_d as *mut c_void,
                        m_sparse,
                        1,
                        16,
                        128,
                        128,
                        group_size,
                        down_had,
                    ),
                    "warmup sparse warp34_moe_w4a16_mlp_ex_f16"
                )?;
                warmup_ffi!(
                    ffi::warp34_moe_w2a16_mlp_ex_f16(
                        &ex_moe,
                        buf_a as *const c_void,
                        moe_sparse_weights as *const u32,
                        buf_a as *const c_void,
                        moe_sparse_weights as *const u32,
                        buf_a as *const c_void,
                        moe_ids_m8 as *const i32,
                        moe_w_m8 as *const f32,
                        core::ptr::null(),
                        core::ptr::null(),
                        buf_c as *mut c_void,
                        buf_d as *mut c_void,
                        m_sparse,
                        1,
                        16,
                        128,
                        128,
                        group_size,
                        down_had,
                    ),
                    "warmup sparse warp34_moe_w2a16_mlp_ex_f16"
                )?;
            }
        }
    }
    warmup_ffi!(
        ffi::warp34_moe_f16_gate_up_silu(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            reject_ids as *const i32,
            buf_c as *mut c_void,
            1,
            1,
            2,
            8,
            64,
        ),
        "warmup warp34_moe_f16_gate_up_silu"
    )?;
    warmup_ffi!(
        ffi::warp34_moe_f16_down_accum(
            ex,
            buf_c as *const c_void,
            buf_b as *const c_void,
            reject_ids as *const i32,
            reject_probs as *const f32,
            buf_a as *const c_void,
            buf_a as *const c_void,
            buf_c as *mut c_void,
            1,
            1,
            2,
            64,
            8,
        ),
        "warmup warp34_moe_f16_down_accum"
    )?;

    // Warm Qwen4Exp (HC, PLE, QSA, and Tiered MoE offload) kernels across TP ranks.
    warmup_ffi!(
        ffi::warp34_hc_init_streams_f16(
            ex,
            buf_a as *const c_void,
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
        ),
        "warmup warp34_hc_init_streams_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_hc_rmsnorm_f16(
            ex,
            buf_c as *const c_void,
            buf_b as *const c_void,
            buf_d as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
            1e-6,
        ),
        "warmup warp34_hc_rmsnorm_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_hc_down_silu_inject_f16(
            ex,
            buf_a as *const c_void,
            buf_c as *mut c_void,
            buf_b as *const c_void,
            buf_d as *mut f32,
            core::ptr::null(),
            1,
            64,
            4,
        ),
        "warmup warp34_hc_down_silu_inject_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_hc_mix_streams_hadamard_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            buf_d as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
        ),
        "warmup warp34_hc_mix_streams_hadamard_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_hc_combine_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_d as *const f32,
            core::ptr::null(),
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
        ),
        "warmup warp34_hc_combine_f16"
    )?;
    let hc_inject_w = buf_d;
    let hc_tap_out = buf_d + 4096;
    let hc_next_norm_w = buf_a + 4096;
    let hc_next_norm_out = buf_d + 8192;
    warmup_ffi!(
        ffi::warp34_hc_combine_tap_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            hc_inject_w as *const f32,
            core::ptr::null(),
            buf_c as *mut c_void,
            hc_tap_out as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
            128,
            0,
        ),
        "warmup warp34_hc_combine_tap_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_hc_combine_tap_rmsnorm_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            hc_inject_w as *const f32,
            core::ptr::null(),
            buf_c as *mut c_void,
            core::ptr::null_mut(),
            hc_next_norm_w as *const c_void,
            hc_next_norm_out as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
            1e-6,
            0,
            0,
        ),
        "warmup warp34_hc_combine_tap_rmsnorm_f16 (notap)"
    )?;
    warmup_ffi!(
        ffi::warp34_hc_combine_tap_rmsnorm_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            hc_inject_w as *const f32,
            core::ptr::null(),
            buf_c as *mut c_void,
            hc_tap_out as *mut c_void,
            hc_next_norm_w as *const c_void,
            hc_next_norm_out as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
            1e-6,
            128,
            0,
        ),
        "warmup warp34_hc_combine_tap_rmsnorm_f16 (tap)"
    )?;
    warmup_ffi!(
        ffi::warp34_qwen4_mtp_broadcast_add_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            1,
            4,
            128,
        ),
        "warmup warp34_qwen4_mtp_broadcast_add_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_ple_dequant_f8_to_f16(
            ex,
            buf_a as *const u8,
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            128,
            1.0,
        ),
        "warmup warp34_ple_dequant_f8_to_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_ple_dequant_f8_hadamard_to_f16(
            ex,
            buf_a as *const u8,
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            128,
            1.0,
        ),
        "warmup warp34_ple_dequant_f8_hadamard_to_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_ple_dequant_lut4_to_f16(
            ex,
            buf_a as *const u8,
            buf_b as *const f32,
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            16,
            160,
            32,
        ),
        "warmup warp34_ple_dequant_lut4_to_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_ple_dequant_lut4_hadamard_to_f16(
            ex,
            buf_a as *const u8,
            buf_b as *const f32,
            buf_c as *mut c_void,
            core::ptr::null(),
            1,
            16,
            160,
            32,
        ),
        "warmup warp34_ple_dequant_lut4_hadamard_to_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_ple_dilated_conv1d_add_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *mut c_void,
            buf_d as *mut c_void,
            buf_d as *mut c_void,
            gdn_batch,
            64,
        ),
        "warmup warp34_ple_dilated_conv1d_add_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_ple_signed_sqrt_gate_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
            1e-6,
        ),
        "warmup warp34_ple_signed_sqrt_gate_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_ple_signed_sqrt_gate_norm_add_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_c as *const c_void,
            buf_d as *mut c_void,
            buf_d as *mut c_void,
            buf_b as *const c_void,
            buf_d as *mut c_void,
            core::ptr::null(),
            1,
            4,
            128,
            1e-6,
            1e-6,
        ),
        "warmup warp34_ple_signed_sqrt_gate_norm_add_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_qsa_indexer_norm_rope_f16(
            ex,
            buf_a as *const c_void,
            buf_b as *const c_void,
            buf_b as *const c_void,
            zero_i32 as *const i32,
            buf_c as *mut c_void,
            buf_d as *mut c_void,
            core::ptr::null(),
            1,
            4,
            1,
            128,
            32,
            10_000_000.0,
            1e-6,
            6,
            11,
            1,
        ),
        "warmup warp34_qsa_indexer_norm_rope_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_qsa_indexer_cache_update_f16(
            ex,
            buf_a as *const c_void,
            buf_c as *mut c_void,
            buf_d as *mut c_void,
            buf_b as *const c_void,
            zero_i32 as *const i32,
            zero_i32 as *const i32,
            core::ptr::null(),
            1,
            128,
            64,
            10_000_000.0,
            1e-6,
            11,
            11,
            1,
        ),
        "warmup warp34_qsa_indexer_cache_update_f16"
    )?;
    let warm_scratch = arena.alloc_aligned(256 * 1024)?;
    let ex_no_ws = stream.exec(0, 0);
    let ex_ws = stream.exec(warm_scratch, 256 * 1024);
    let zeros_16_i32 = arena.upload_slice(&[0i32; 16], stream)?;
    for qsa_ex in [&ex_no_ws, &ex_ws] {
        warmup_ffi!(
            ffi::warp34_qsa_select_blocks_f16(
                qsa_ex,
                buf_a as *const c_void,
                buf_b as *const c_void,
                attn_batch,
                buf_c as *mut i32,
                (buf_c + 256) as *mut i32,
                (buf_c + 512) as *mut i32,
                (buf_c + 768) as *mut i32,
                4,
                128,
                16,
                16,
                2,
                2,
            ),
            "warmup warp34_qsa_select_blocks_f16"
        )?;
        for local_q_heads in [2i32, 4] {
            warmup_ffi!(
                ffi::warp34_qsa_sparse_paged_attn_f16(
                    qsa_ex,
                    buf_a as *const c_void,
                    buf_b as *const c_void,
                    layout_256,
                    zeros_16_i32 as *const i32,
                    zeros_16_i32 as *const i32,
                    zero_i32 as *const i32,
                    zero_i32 as *const i32,
                    core::ptr::null(),
                    buf_d as *mut c_void,
                    1,
                    local_q_heads,
                    16,
                    0.0625,
                ),
                "warmup warp34_qsa_sparse_paged_attn_f16"
            )?;
            warmup_ffi!(
                ffi::warp34_qsa_sparse_paged_attn_hadamard_f16(
                    qsa_ex,
                    buf_a as *const c_void,
                    buf_b as *const c_void,
                    layout_256,
                    zeros_16_i32 as *const i32,
                    zeros_16_i32 as *const i32,
                    zero_i32 as *const i32,
                    zero_i32 as *const i32,
                    core::ptr::null(),
                    buf_d as *mut c_void,
                    1,
                    local_q_heads,
                    16,
                    0.0625,
                ),
                "warmup warp34_qsa_sparse_paged_attn_hadamard_f16"
            )?;
        }
    }
    warmup_ffi!(
        ffi::warp34_moe_unpack_staged_experts(
            ex,
            buf_a as *const c_void,
            zero_i32 as *const i32,
            buf_c as *mut u32,
            (buf_c + 32768) as *mut c_void,
            buf_d as *mut u32,
            (buf_d + 32768) as *mut c_void,
            1,
            1,
            28672,
            128,
            128,
            128,
            4,
        ),
        "warmup warp34_moe_unpack_staged_experts"
    )?;
    warmup_ffi!(
        ffi::warp34_moe_remap_topk_ids_i32(
            ex,
            zero_i32 as *const i32,
            zero_i32 as *const i32,
            buf_c as *mut i32,
            1,
            0,
            16,
        ),
        "warmup warp34_moe_remap_topk_ids_i32"
    )?;
    warmup_ffi!(
        ffi::warp34_add_inplace_f16(ex, buf_c as *mut c_void, buf_a as *const c_void, 128),
        "warmup warp34_add_inplace_f16"
    )?;
    warmup_ffi!(
        ffi::warp34_moe_gpu_lru_update_and_pull(
            ex,
            zero_i32 as *const i32,
            buf_c as *mut i32,
            (buf_c + 256) as *mut i32,
            (buf_c + 512) as *mut u64,
            (buf_c + 1024) as *mut u64,
            (buf_c + 1280) as *mut i32,
            (buf_c + 1536) as *mut i32,
            (buf_c + 1792) as *mut i32,
            (buf_c + 2048) as *mut i32,
            buf_a as *const c_void,
            buf_b as *mut u32,
            (buf_b + 16384) as *mut c_void,
            (buf_b + 32768) as *mut u32,
            (buf_b + 49152) as *mut c_void,
            1,
            0,
            16,
            8,
            1,
            28672,
            128,
            128,
            128,
            4,
        ),
        "warmup warp34_moe_gpu_lru_update_and_pull"
    )?;
    for gs in [64i32, 128] {
        for pulled_fn in [
            ffi::warp34_moe_w4a16_pulled_add_f16,
            ffi::warp34_moe_w2a16_pulled_add_f16,
        ] {
            warmup_ffi!(
                pulled_fn(
                    &ex_ws,
                    buf_a as *const c_void,
                    buf_b as *const u32,
                    (buf_a + 4096) as *const c_void,
                    (buf_b + 32768) as *const u32,
                    (buf_a + 8192) as *const c_void,
                    reject_ids as *const i32,
                    reject_probs as *const f32,
                    zero_i32 as *const i32,
                    buf_c as *mut c_void,
                    buf_d as *mut c_void,
                    1,
                    1,
                    2,
                    128,
                    128,
                    gs,
                    64,
                ),
                "warmup warp34_moe_wxa16_pulled_add_f16"
            )?;
        }
    }
    let hybrid_slots = arena.alloc_aligned(128 * 1024)?;
    let hybrid_state = arena.alloc_aligned(4096)?;
    let hybrid_desc = ffi::Warp34MoeHybridDesc {
        d_req_slot: hybrid_slots as *mut c_void,
        x_f16: buf_a as *const c_void,
        global_topk_ids: zero_i32 as *const i32,
        topk_weights: reject_probs as *const f32,
        expert_to_slot: hybrid_state as *mut i32,
        slot_to_expert: (hybrid_state + 256) as *mut i32,
        slot_last_used: (hybrid_state + 512) as *mut u64,
        step_counter: (hybrid_state + 1024) as *mut u64,
        expert_freq: (hybrid_state + 1280) as *mut u8,
        remapped_topk_ids: (hybrid_state + 1536) as *mut i32,
        pulled_topk_ids: core::ptr::null_mut(),
        miss_local_experts: (hybrid_state + 1792) as *mut i32,
        miss_victim_slots: (hybrid_state + 2048) as *mut i32,
        num_misses_buf: (hybrid_state + 2304) as *mut i32,
        mapped_host_packs: core::ptr::null(),
        gate_up_packed_base: buf_b as *mut u32,
        gate_up_scales_base: (buf_b + 16384) as *mut c_void,
        down_packed_base: (buf_b + 32768) as *mut u32,
        down_scales_base: (buf_b + 49152) as *mut c_void,
        pull_stream: core::ptr::null_mut(),
        plan_event: core::ptr::null_mut(),
        pull_done_event: core::ptr::null_mut(),
        pack_stride_bytes: 28672,
        num_tokens: 1,
        top_k: 1,
        global_expert_offset: 0,
        local_num_experts: 16,
        k_static: 8,
        k_hot: 8,
        n_interm: 128,
        k_hidden: 128,
        group_size: 128,
        bits: 4,
        gpu_share_frac: 0.0,
        reserved: 0,
    };
    warmup_ffi!(
        ffi::warp34_moe_hybrid_submit_and_pull(&ex_ws, &hybrid_desc),
        "warmup warp34_moe_hybrid_submit_and_pull"
    )?;
    warmup_ffi!(
        ffi::warp34_moe_hybrid_wait_and_add(
            &ex_ws,
            hybrid_slots as *const c_void,
            (hybrid_slots + 65536) as *const c_void,
            buf_d as *mut c_void,
            (hybrid_state + 2304) as *const i32,
            1,
            128,
        ),
        "warmup warp34_moe_hybrid_wait_and_add"
    )?;
    Ok(())
}
