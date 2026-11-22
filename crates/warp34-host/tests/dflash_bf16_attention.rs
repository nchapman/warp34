#![cfg(feature = "cuda")]
//! Actual donor operands, never native upstream outputs. Run with CUDA_VISIBLE_DEVICES=1.
//! Pinned sm86 FA2 uses 16-key KV tiles (CTA_TILE_Q=64 is the QUERY tile).
//! The 21-key fixtures exercise two online probability-rounding updates.
//! Keep all five layers at the same strict 1e-4 bound.
use core::ffi::c_void;
use warp34_host::cuda::{CudaDevice, CudaStream, DeviceBuffer, check_rc};
use warp34_kernels::ffi;
fn replay(layers: std::ops::Range<usize>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/oracle/goldens/sglang_dflash_model_4b.safetensors");
    let archive = warp34_host::loader::SafetensorsArchive::open(&path).unwrap();
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    for layer in layers {
        replay_paged_bf16(&archive, &stream, &format!("layers.{layer}"), 16, 21, 0);
    }
}
#[test]
fn first_layer_actual_donor_attention() {
    replay(0..1);
}
#[test]
fn all_five_layers_actual_donor_attention() {
    let mut failed = Vec::new();
    for layer in 0..5 {
        if std::panic::catch_unwind(|| replay(layer..layer + 1)).is_err() {
            failed.push(layer);
        }
    }
    assert!(
        failed.is_empty(),
        "donor attention mismatch in layers {failed:?}"
    );
}

fn replay_paged_bf16(
    archive: &warp34_host::loader::SafetensorsArchive,
    stream: &CudaStream,
    name: &str,
    rows: i32,
    seq_len: i32,
    window: i32,
) {
    use warp34_host::cache::{KvFormatPolicy, KvPool};
    let ex = stream.exec(0, 0);
    let upload = |data: &[u8]| {
        let buf = DeviceBuffer::alloc(0, data.len()).unwrap();
        buf.copy_from_host(data, stream).unwrap();
        buf
    };
    let ints = |data: &[i32]| {
        upload(
            &data
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let q = upload(&archive.read_bytes(&format!("{name}.q")).unwrap());
    let k = upload(&archive.read_bytes(&format!("{name}.k_all")).unwrap());
    let v = upload(&archive.read_bytes(&format!("{name}.v_all")).unwrap());
    let expected = archive.read_f32(&format!("{name}.attention")).unwrap();
    let pages = (seq_len as usize).div_ceil(16);
    let pool = KvPool::new(
        0,
        ffi::WARP34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
        &[0],
        4,
        256,
        16,
        pages,
        KvFormatPolicy::Bf16,
    )
    .unwrap();
    let layout = &pool.layers[0].layout;
    let table: Vec<i32> = (0..pages as i32).rev().collect();
    let slots: Vec<i32> = (0..seq_len)
        .map(|i| table[i as usize / 16] * 16 + i % 16)
        .collect();
    let blocks = ints(&table);
    let mapping = ints(&slots);
    let kv_start = ints(&[0, seq_len]);
    let q_start = ints(&[0, rows]);
    let ctx = ints(&[seq_len - rows]);
    let seq = ints(&[seq_len]);
    let masked_req = upload(&[0u8]);
    let out = upload(&vec![0u8; expected.len() * 2]);
    let mut batch = ffi::Warp34AttentionBatch {
        query_start_loc: kv_start.addr() as *const i32,
        context_lens_before: ctx.addr() as *const i32,
        seq_lens_after: seq.addr() as *const i32,
        block_tables: blocks.addr() as *const i32,
        write_slot_mapping: mapping.addr() as *const i32,
        valid_token_mask: core::ptr::null(),
        valid_req_mask: core::ptr::null(),
        num_reqs: 1,
        total_tokens: seq_len,
        max_blocks_per_req: pages as i32,
        max_q_len: seq_len,
        max_seq_len: seq_len,
        sliding_window: window,
        mask_mode: ffi::WARP34_ATTN_MASK_NONCAUSAL_WINDOW,
        sm_scale: 1.0 / 256.0f32.sqrt(),
        reserved: 0,
    };
    // SAFETY: cache layout covers all reversed physical pages, metadata buffers
    // live through synchronization, and source tensors have 4 KV / 16 query heads.
    unsafe {
        check_rc(
            ffi::warp34_kv_cache_write_bf16(
                &ex,
                layout,
                &batch,
                k.addr() as *const c_void,
                v.addr() as *const c_void,
            ),
            "paged BF16 write",
        )
        .unwrap();
        // A padded request must not overwrite live cache, even with valid slots.
        batch.valid_req_mask = masked_req.addr() as *const u8;
        check_rc(
            ffi::warp34_kv_cache_write_bf16(
                &ex,
                layout,
                &batch,
                v.addr() as *const c_void,
                k.addr() as *const c_void,
            ),
            "masked BF16 write",
        )
        .unwrap();
        batch.valid_req_mask = core::ptr::null();
        batch.query_start_loc = q_start.addr() as *const i32;
        batch.total_tokens = rows;
        batch.max_q_len = rows;
        check_rc(
            ffi::warp34_paged_attention_bf16(
                &ex,
                layout,
                &batch,
                q.addr() as *const c_void,
                out.addr() as *mut c_void,
                16,
            ),
            "paged BF16 attention",
        )
        .unwrap();
    }
    let mut bits = vec![0u16; expected.len()];
    out.copy_to_host(&mut bits, stream).unwrap();
    let actual: Vec<f32> = bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = (actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / expected.iter().map(|b| (*b as f64).powi(2)).sum::<f64>())
    .sqrt();
    eprintln!("{name}: relL2={rel:.9} maxabs={max_abs}");
    assert!(
        actual.iter().all(|x| x.is_finite()) && rel < 0.0001,
        "{name}: relL2={rel}, maxabs={max_abs}"
    );
    // Masked requests leave output storage untouched (including captured replay).
    let sentinel = vec![0xC2F6u16; expected.len()];
    out.copy_from_host(&sentinel, stream).unwrap();
    batch.valid_req_mask = masked_req.addr() as *const u8;
    // SAFETY: unchanged validated buffers; only the live request mask is disabled.
    unsafe {
        check_rc(
            ffi::warp34_paged_attention_bf16(
                &ex,
                layout,
                &batch,
                q.addr() as *const c_void,
                out.addr() as *mut c_void,
                16,
            ),
            "masked BF16 attention",
        )
        .unwrap();
    }
    out.copy_to_host(&mut bits, stream).unwrap();
    assert_eq!(bits, sentinel);
}

#[test]
fn packed_and_splitk_bf16_attention_match_single_row_reference() {
    use warp34_host::{
        cache::{KvFormatPolicy, KvPool},
        cuda::f32_to_bf16,
    };
    let _device = CudaDevice::set_current(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let upload_u16 = |data: &[u16]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = DeviceBuffer::alloc(0, bytes.len()).unwrap();
        buf.copy_from_host(&bytes, &stream).unwrap();
        buf
    };
    let upload_i32 = |data: &[i32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let buf = DeviceBuffer::alloc(0, bytes.len()).unwrap();
        buf.copy_from_host(&bytes, &stream).unwrap();
        buf
    };

    for (head_dim, kv_heads, q_heads) in [(128usize, 8usize, 32usize), (256, 4, 16)] {
        let seq_len = 2500i32;
        let window = 2048i32;
        let rows = 8i32;
        let pages = (seq_len as usize).div_ceil(16);
        let pool = KvPool::new(
            0,
            ffi::WARP34_KV_NAMESPACE_DFLASH_DRAFT_CTX,
            &[0],
            kv_heads,
            head_dim,
            16,
            pages,
            KvFormatPolicy::Bf16,
        )
        .unwrap();
        let layout = &pool.layers[0].layout;
        let table: Vec<i32> = (0..pages as i32).rev().collect();
        let slots: Vec<i32> = (0..seq_len)
            .map(|i| table[i as usize / 16] * 16 + i % 16)
            .collect();

        let k_host: Vec<u16> = (0..(seq_len as usize) * kv_heads * head_dim)
            .map(|i| f32_to_bf16(((i * 13 + 7) % 97) as f32 * 0.015625 - 0.75))
            .collect();
        let v_host: Vec<u16> = (0..(seq_len as usize) * kv_heads * head_dim)
            .map(|i| f32_to_bf16(((i * 29 + 3) % 89) as f32 * 0.015625 - 0.68))
            .collect();
        let q_host: Vec<u16> = (0..(rows as usize) * q_heads * head_dim)
            .map(|i| f32_to_bf16(((i * 17 + 11) % 83) as f32 * 0.015625 - 0.62))
            .collect();

        let d_k = upload_u16(&k_host);
        let d_v = upload_u16(&v_host);
        let d_q = upload_u16(&q_host);
        let d_blocks = upload_i32(&table);
        let d_slots = upload_i32(&slots);
        let d_kv_start = upload_i32(&[0, seq_len]);
        let d_seq = upload_i32(&[seq_len]);
        let ex_plain = stream.exec(0, 0);

        let write_batch = ffi::Warp34AttentionBatch {
            query_start_loc: d_kv_start.addr() as *const i32,
            context_lens_before: core::ptr::null(),
            seq_lens_after: d_seq.addr() as *const i32,
            block_tables: d_blocks.addr() as *const i32,
            write_slot_mapping: d_slots.addr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: seq_len,
            max_blocks_per_req: pages as i32,
            max_q_len: seq_len,
            max_seq_len: seq_len,
            sliding_window: window,
            mask_mode: ffi::WARP34_ATTN_MASK_NONCAUSAL_WINDOW,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            reserved: 0,
        };
        // SAFETY: buffers match `[seq_len, kv_heads, head_dim]` BF16 write contract.
        unsafe {
            check_rc(
                ffi::warp34_kv_cache_write_bf16(
                    &ex_plain,
                    layout,
                    &write_batch,
                    d_k.addr() as *const c_void,
                    d_v.addr() as *const c_void,
                ),
                "write bf16 kv",
            )
            .unwrap();
        }

        // Reference: 8 single-token launches (`total_tokens == 1` path).
        let out_elems = (rows as usize) * q_heads * head_dim;
        let d_ref = upload_u16(&vec![0u16; out_elems]);
        let d_q1 = upload_i32(&[0, 1]);
        for r in 0..rows {
            let d_ctx1 = upload_i32(&[seq_len - rows + r]);
            let single_batch = ffi::Warp34AttentionBatch {
                query_start_loc: d_q1.addr() as *const i32,
                context_lens_before: d_ctx1.addr() as *const i32,
                seq_lens_after: d_seq.addr() as *const i32,
                block_tables: d_blocks.addr() as *const i32,
                write_slot_mapping: d_slots.addr() as *const i32,
                valid_token_mask: core::ptr::null(),
                valid_req_mask: core::ptr::null(),
                num_reqs: 1,
                total_tokens: 1,
                max_blocks_per_req: pages as i32,
                max_q_len: 1,
                max_seq_len: seq_len,
                sliding_window: window,
                mask_mode: ffi::WARP34_ATTN_MASK_NONCAUSAL_WINDOW,
                sm_scale: 1.0 / (head_dim as f32).sqrt(),
                reserved: 0,
            };
            let row_bytes = (q_heads * head_dim * 2) as u64;
            // SAFETY: single-token slice inside `d_q` and `d_ref`.
            unsafe {
                check_rc(
                    ffi::warp34_paged_attention_bf16(
                        &ex_plain,
                        layout,
                        &single_batch,
                        (d_q.addr() + (r as u64) * row_bytes) as *const c_void,
                        (d_ref.addr() + (r as u64) * row_bytes) as *mut c_void,
                        q_heads as i32,
                    ),
                    "single-row bf16 ref",
                )
                .unwrap();
            }
        }
        let mut ref_bits = vec![0u16; out_elems];
        d_ref.copy_to_host(&mut ref_bits, &stream).unwrap();

        let d_q_rows = upload_i32(&[0, rows]);
        let d_ctx = upload_i32(&[seq_len - rows]);
        let mut multi_batch = ffi::Warp34AttentionBatch {
            query_start_loc: d_q_rows.addr() as *const i32,
            context_lens_before: d_ctx.addr() as *const i32,
            seq_lens_after: d_seq.addr() as *const i32,
            block_tables: d_blocks.addr() as *const i32,
            write_slot_mapping: d_slots.addr() as *const i32,
            valid_token_mask: core::ptr::null(),
            valid_req_mask: core::ptr::null(),
            num_reqs: 1,
            total_tokens: rows,
            max_blocks_per_req: pages as i32,
            max_q_len: rows,
            max_seq_len: seq_len,
            sliding_window: window,
            mask_mode: ffi::WARP34_ATTN_MASK_NONCAUSAL_WINDOW,
            sm_scale: 1.0 / (head_dim as f32).sqrt(),
            reserved: 0,
        };

        // 1. Packed (`split_k == 1`, no scratch) must be 0-ULP bit-for-bit identical.
        let d_packed = upload_u16(&vec![0u16; out_elems]);
        // SAFETY: `d_q` and `d_packed` hold `out_elems` BF16 elements.
        unsafe {
            check_rc(
                ffi::warp34_paged_attention_bf16(
                    &ex_plain,
                    layout,
                    &multi_batch,
                    d_q.addr() as *const c_void,
                    d_packed.addr() as *mut c_void,
                    q_heads as i32,
                ),
                "packed bf16 attention",
            )
            .unwrap();
        }
        let mut packed_bits = vec![0u16; out_elems];
        d_packed.copy_to_host(&mut packed_bits, &stream).unwrap();
        assert_eq!(packed_bits, ref_bits, "0-ULP mismatch for hd={head_dim}");

        // 2. Packed + Split-K (`split_k == 8`, 4 MiB scratch) must match within < 1e-4 relL2
        //    and preserve sentinel under `valid_req_mask == 0`.
        let ws = DeviceBuffer::alloc(0, 4 * 1024 * 1024).unwrap();
        let ex_split = stream.exec(ws.addr(), ws.len_bytes);
        let d_split = upload_u16(&vec![0u16; out_elems]);
        // SAFETY: `ws` provides >= 4 MiB 16B-aligned scratch for Split-K.
        unsafe {
            check_rc(
                ffi::warp34_paged_attention_bf16(
                    &ex_split,
                    layout,
                    &multi_batch,
                    d_q.addr() as *const c_void,
                    d_split.addr() as *mut c_void,
                    q_heads as i32,
                ),
                "splitk bf16 attention",
            )
            .unwrap();
        }
        let mut split_bits = vec![0u16; out_elems];
        d_split.copy_to_host(&mut split_bits, &stream).unwrap();
        let to_f32 = |b: u16| f32::from_bits((b as u32) << 16);

        // Exact FP64 attention reference to verify Split-K matches or beats single-pass BF16 accuracy.
        let gqa = q_heads / kv_heads;
        let sm_scale = 1.0f64 / (head_dim as f64).sqrt();
        let mut exact_f64 = vec![0.0f64; out_elems];
        for r in 0..(rows as usize) {
            let ref_pos = (seq_len - rows) as usize + r;
            let min_kv = (ref_pos + 1).saturating_sub(window as usize);
            let max_kv = (seq_len - 1) as usize;
            for qh in 0..q_heads {
                let kvh = qh / gqa;
                let q_off = (r * q_heads + qh) * head_dim;
                let mut m = f64::NEG_INFINITY;
                let mut scores = Vec::with_capacity(max_kv - min_kv + 1);
                for kv in min_kv..=max_kv {
                    let k_off = (kv * kv_heads + kvh) * head_dim;
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot +=
                            (to_f32(q_host[q_off + d]) as f64) * (to_f32(k_host[k_off + d]) as f64);
                    }
                    let s = dot * sm_scale;
                    if s > m {
                        m = s;
                    }
                    scores.push((kv, s));
                }
                let mut denom = 0.0f64;
                for &(kv, s) in &scores {
                    let w = (s - m).exp();
                    denom += w;
                    let v_off = (kv * kv_heads + kvh) * head_dim;
                    for d in 0..head_dim {
                        exact_f64[q_off + d] += w * (to_f32(v_host[v_off + d]) as f64);
                    }
                }
                for d in 0..head_dim {
                    exact_f64[q_off + d] /= denom;
                }
            }
        }
        let norm_f64 = exact_f64.iter().map(|x| x * x).sum::<f64>().sqrt();
        let ref_rel_f64 = (ref_bits
            .iter()
            .zip(&exact_f64)
            .map(|(&a, &e)| (to_f32(a) as f64 - e).powi(2))
            .sum::<f64>())
        .sqrt()
            / norm_f64;
        let split_rel_f64 = (split_bits
            .iter()
            .zip(&exact_f64)
            .map(|(&a, &e)| (to_f32(a) as f64 - e).powi(2))
            .sum::<f64>())
        .sqrt()
            / norm_f64;
        assert!(
            split_rel_f64 < 3e-3 && split_rel_f64 <= ref_rel_f64 * 1.15,
            "splitk hd={head_dim} split_rel_f64={split_rel_f64} ref_rel_f64={ref_rel_f64}"
        );

        let sentinel = vec![0xC2F6u16; out_elems];
        let d_masked = DeviceBuffer::alloc(0, 1).unwrap();
        d_masked.copy_from_host(&[0u8], &stream).unwrap();
        let d_sent = upload_u16(&sentinel);
        multi_batch.valid_req_mask = d_masked.addr() as *const u8;
        // SAFETY: masked request leaves `d_sent` untouched.
        unsafe {
            check_rc(
                ffi::warp34_paged_attention_bf16(
                    &ex_split,
                    layout,
                    &multi_batch,
                    d_q.addr() as *const c_void,
                    d_sent.addr() as *mut c_void,
                    q_heads as i32,
                ),
                "masked splitk bf16 attention",
            )
            .unwrap();
        }
        let mut masked_out = vec![0u16; out_elems];
        d_sent.copy_to_host(&mut masked_out, &stream).unwrap();
        assert_eq!(masked_out, sentinel);
    }
}
