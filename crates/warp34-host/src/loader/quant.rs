use super::*;

/// Permutes a `[K / group_size, N]` FP16 scale matrix into `vLLM` Marlin scale layout.
///
/// When `group_size > 0 && group_size <= 128` (`group_blocks > 0`), applies the 64-element
/// 8x8 tile transpose expected by `Marlin<..., group_blocks > 0>` (`s_sh_rd = 8 * warp_col + lane / 4`),
/// even when `size_k == group_size` (e.g. `size_k = 128, group_size = 128, group_blocks = 8`).
/// When `group_size == 0` or `group_size > 128` (`group_blocks == -1`, channelwise W8A16),
/// applies the 32-element post-MMA column permutation (`scale_perm_single`).
pub fn marlin_permute_scales_k_by_n(
    scales_k_by_n: &[u16],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u16> {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    let num_groups = size_k / eff_gs;
    assert_eq!(scales_k_by_n.len(), num_groups * size_n);
    let mut out = vec![0u16; scales_k_by_n.len()];
    let is_grouped_marlin = group_size > 0 && group_size <= 128;
    if is_grouped_marlin {
        let mut scale_perm = [0usize; 64];
        for i in 0..8 {
            for j in 0..8 {
                scale_perm[i * 8 + j] = i + 8 * j;
            }
        }
        for (chunk_idx, src_chunk) in scales_k_by_n.as_chunks::<64>().0.iter().enumerate() {
            let dst_base = chunk_idx * 64;
            for (i, &p) in scale_perm.iter().enumerate() {
                out[dst_base + i] = src_chunk[p];
            }
        }
    } else {
        let mut scale_perm_single = [0usize; 32];
        const BASE: [usize; 8] = [0, 1, 8, 9, 16, 17, 24, 25];
        for i in 0..4 {
            for (j, &b) in BASE.iter().enumerate() {
                scale_perm_single[i * 8 + j] = 2 * i + b;
            }
        }
        for (chunk_idx, src_chunk) in scales_k_by_n.as_chunks::<32>().0.iter().enumerate() {
            let dst_base = chunk_idx * 32;
            for (i, &p) in scale_perm_single.iter().enumerate() {
                out[dst_base + i] = src_chunk[p];
            }
        }
    }
    out
}

/// Transposes an N-major `[N, K / group_size]` scale tensor into `[K / group_size, N]`
/// and applies `marlin_permute_scales_k_by_n`.
pub fn marlin_permute_scales_from_n_major(
    scales_n_by_kg: &[u16],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u16> {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    let num_groups = size_k / eff_gs;
    assert_eq!(scales_n_by_kg.len(), size_n * num_groups);
    let mut k_by_n = vec![0u16; num_groups * size_n];
    for n in 0..size_n {
        for g in 0..num_groups {
            k_by_n[g * size_n + n] = scales_n_by_kg[n * num_groups + g];
        }
    }
    marlin_permute_scales_k_by_n(&k_by_n, size_k, size_n, group_size)
}

/// Losslessly dilates N-major 2-bit `uint2b2` packed words (`16` crumbs/word, zero-point `2`)
/// into N-major 4-bit `uint4b8` packed words (`8` nibbles/word, zero-point `8`, `u4 = u2 + 6`).
pub fn repack_u2b2_to_u4b8(packed_u2: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(packed_u2.len() * 2);
    for &w in packed_u2 {
        for half in [w & 0xFFFF, w >> 16] {
            let mut x = half;
            x = (x | (x << 8)) & 0x00FF_00FF;
            x = (x | (x << 4)) & 0x0F0F_0F0F;
            x = (x | (x << 2)) & 0x3333_3333;
            x = x.wrapping_add(0x6666_6666);
            out.push(x);
        }
    }
    out
}

/// Losslessly dilates N-major half-integer symmetric 2-bit packed words (`code_offset == 0.5`,
/// `q = u2 - 1.5 in {-1.5, -0.5, +0.5, +1.5}`) into N-major 4-bit `uint4b8` packed words
/// (`u4 = 2 * u2 + 5 in {5, 7, 9, 11}`); caller must pair with [`halve_f16_scales`].
pub fn repack_u2b2_sym_to_u4b8(packed_u2: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(packed_u2.len() * 2);
    for &w in packed_u2 {
        for half in [w & 0xFFFF, w >> 16] {
            let mut x = half;
            x = (x | (x << 8)) & 0x00FF_00FF;
            x = (x | (x << 4)) & 0x0F0F_0F0F;
            x = (x | (x << 2)) & 0x3333_3333;
            x = (x << 1).wrapping_add(0x5555_5555);
            out.push(x);
        }
    }
    out
}

/// Halves FP16 scale values (`s * 0.5`) when dilating `code_offset == 0.5` 2-bit weights to `U4B8`.
pub fn halve_f16_scales(scales: &[u16]) -> Vec<u16> {
    scales
        .iter()
        .map(|&bits| f32_to_f16(f16_to_f32(bits) * 0.5))
        .collect()
}

/// Dequantizes an N-major `[N, K/16]` `U2B2` weight + `[N, K/group_size]` `FP16` scale tensor
/// into an N-major `[N, K]` `FP16` matrix (`code_offset == 0.0`).
pub fn dequant_u2b2_n_major_to_f16(
    packed_n_by_k16: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    dequant_u2b2_n_major_to_f16_with_offset(
        packed_n_by_k16,
        scales_n_by_kg,
        size_n,
        size_k,
        group_size,
        false,
    )
}

/// Dequantizes an N-major `[N, K/16]` 2-bit weight + `[N, K/group_size]` `FP16` scale tensor
/// into an N-major `[N, K]` `FP16` matrix with optional half-integer `code_offset == 0.5`.
pub fn dequant_u2b2_n_major_to_f16_with_offset(
    packed_n_by_k16: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
    code_offset_half: bool,
) -> Vec<u16> {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    let words_per_row = size_k / 16;
    let groups_per_row = size_k / eff_gs;
    assert_eq!(packed_n_by_k16.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    let zp = if code_offset_half { 1.5f32 } else { 2.0f32 };
    let dequant_rows = |row_start: usize, dst_rows: &mut [u16]| {
        for (rel_n, dst_row) in dst_rows.chunks_exact_mut(size_k).enumerate() {
            let n = row_start + rel_n;
            let row_words = &packed_n_by_k16[n * words_per_row..(n + 1) * words_per_row];
            let row_scales = &scales_n_by_kg[n * groups_per_row..(n + 1) * groups_per_row];
            if eff_gs.is_multiple_of(16) {
                let words_per_group = eff_gs / 16;
                for (g, &sc_bits) in row_scales.iter().enumerate() {
                    let scale = f16_to_f32(sc_bits);
                    let lut = if code_offset_half {
                        [
                            f32_to_f16(-1.5 * scale),
                            f32_to_f16(-0.5 * scale),
                            f32_to_f16(0.5 * scale),
                            f32_to_f16(1.5 * scale),
                        ]
                    } else {
                        [
                            f32_to_f16(-2.0 * scale),
                            f32_to_f16(-scale),
                            0u16,
                            f32_to_f16(scale),
                        ]
                    };
                    let g_words = &row_words[g * words_per_group..(g + 1) * words_per_group];
                    let g_dst = &mut dst_row[g * eff_gs..(g + 1) * eff_gs];
                    for (w_idx, &word) in g_words.iter().enumerate() {
                        let base = w_idx * 16;
                        for b in 0..16 {
                            g_dst[base + b] = lut[((word >> (b * 2)) & 0x3) as usize];
                        }
                    }
                }
            } else {
                for (w_idx, &word) in row_words.iter().enumerate() {
                    for b in 0..16 {
                        let k = w_idx * 16 + b;
                        let crumb = ((word >> (b * 2)) & 0x3) as f32 - zp;
                        let scale = f16_to_f32(row_scales[k / eff_gs]);
                        dst_row[k] = f32_to_f16(crumb * scale);
                    }
                }
            }
        }
    };
    if size_n < 1024 {
        dequant_rows(0, &mut out);
    } else {
        let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get().clamp(1, 16));
        let rows_per_chunk = size_n.div_ceil(num_threads).max(1);
        std::thread::scope(|s| {
            for (chunk_idx, dst_chunk) in out.chunks_mut(rows_per_chunk * size_k).enumerate() {
                let row_start = chunk_idx * rows_per_chunk;
                s.spawn(move || dequant_rows(row_start, dst_chunk));
            }
        });
    }
    out
}

/// Dequantizes an N-major `[N, K/8]` `U4B8` weight + `[N, K/group_size]` `FP16` scale tensor
/// into an N-major `[N, K]` `FP16` matrix (used for skinny projections with `N < 64` or `N % 64 != 0`).
pub fn dequant_u4b8_n_major_to_f16(
    packed_n_by_k8: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let words_per_row = size_k / 8;
    let groups_per_row = size_k / group_size;
    assert_eq!(packed_n_by_k8.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    let dequant_rows = |row_start: usize, dst_rows: &mut [u16]| {
        for (rel_n, dst_row) in dst_rows.chunks_exact_mut(size_k).enumerate() {
            let n = row_start + rel_n;
            let row_words = &packed_n_by_k8[n * words_per_row..(n + 1) * words_per_row];
            let row_scales = &scales_n_by_kg[n * groups_per_row..(n + 1) * groups_per_row];
            if group_size.is_multiple_of(8) {
                let words_per_group = group_size / 8;
                for (g, &sc_bits) in row_scales.iter().enumerate() {
                    let scale = f16_to_f32(sc_bits);
                    let mut lut = [0u16; 16];
                    for (u, slot) in lut.iter_mut().enumerate() {
                        *slot = f32_to_f16(((u as i32 - 8) as f32) * scale);
                    }
                    let g_words = &row_words[g * words_per_group..(g + 1) * words_per_group];
                    let g_dst = &mut dst_row[g * group_size..(g + 1) * group_size];
                    for (w_idx, &word) in g_words.iter().enumerate() {
                        let base = w_idx * 8;
                        for b in 0..8 {
                            g_dst[base + b] = lut[((word >> (b * 4)) & 0xF) as usize];
                        }
                    }
                }
            } else {
                for (w_idx, &word) in row_words.iter().enumerate() {
                    for b in 0..8 {
                        let k = w_idx * 8 + b;
                        let nib = ((word >> (b * 4)) & 0xF) as i32 - 8;
                        let scale = f16_to_f32(row_scales[k / group_size]);
                        dst_row[k] = f32_to_f16((nib as f32) * scale);
                    }
                }
            }
        }
    };
    if size_n < 1024 {
        dequant_rows(0, &mut out);
    } else {
        let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get().clamp(1, 16));
        let rows_per_chunk = size_n.div_ceil(num_threads).max(1);
        std::thread::scope(|s| {
            for (chunk_idx, dst_chunk) in out.chunks_mut(rows_per_chunk * size_k).enumerate() {
                let row_start = chunk_idx * rows_per_chunk;
                s.spawn(move || dequant_rows(row_start, dst_chunk));
            }
        });
    }
    out
}

/// Transposes a K-major `[K/8, N]` 4-bit GPTQ `qweight` tensor into N-major `[N, K/8]` `u32` words.
pub fn transpose_gptq_qweight_k8_n_to_n_major(
    qweight_k8_by_n: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    let k8 = size_k / 8;
    assert_eq!(qweight_k8_by_n.len(), k8 * size_n);
    let mut out = vec![0u32; size_n * k8];
    const TILE: usize = 32;
    let mut tile = [[0u32; TILE]; TILE];
    for kr_base in (0..k8).step_by(TILE) {
        let kr_len = (k8 - kr_base).min(TILE);
        for n_base in (0..size_n).step_by(TILE) {
            let n_len = (size_n - n_base).min(TILE);
            for (tk, tile_row) in tile.iter_mut().enumerate().take(kr_len) {
                let src = &qweight_k8_by_n[(kr_base + tk) * size_n + n_base..];
                for (tn, slot) in tile_row.iter_mut().enumerate().take(n_len) {
                    *slot = src[tn];
                }
            }
            for tn in 0..n_len {
                let dst = &mut out[(n_base + tn) * k8 + kr_base..];
                for (tk, &row) in tile.iter().enumerate().take(kr_len) {
                    dst[tk] = row[tn];
                }
            }
        }
    }
    out
}

/// Transposes a K-major `[K/4, N]` 8-bit GPTQ `qweight` tensor into N-major `[N, K/4]` `u32` words.
pub fn transpose_gptq_qweight_k4_n_to_n_major(
    qweight_k4_by_n: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    let k4 = size_k / 4;
    assert_eq!(qweight_k4_by_n.len(), k4 * size_n);
    let mut out = vec![0u32; size_n * k4];
    const TILE: usize = 32;
    let mut tile = [[0u32; TILE]; TILE];
    for kr_base in (0..k4).step_by(TILE) {
        let kr_len = (k4 - kr_base).min(TILE);
        for n_base in (0..size_n).step_by(TILE) {
            let n_len = (size_n - n_base).min(TILE);
            for (tk, tile_row) in tile.iter_mut().enumerate().take(kr_len) {
                let src = &qweight_k4_by_n[(kr_base + tk) * size_n + n_base..];
                for (tn, slot) in tile_row.iter_mut().enumerate().take(n_len) {
                    *slot = src[tn];
                }
            }
            for tn in 0..n_len {
                let dst = &mut out[(n_base + tn) * k4 + kr_base..];
                for (tk, &row) in tile.iter().enumerate().take(kr_len) {
                    dst[tk] = row[tn];
                }
            }
        }
    }
    out
}

/// Unpacks an AutoAWQ `gemm` `[K, N/8]` `qweight` tensor (interleaved `[0, 2, 4, 6, 1, 3, 5, 7]`
/// along `N`) into N-major `[N, K/8]` `u32` words (8 consecutive K-nibbles per `u32`).
pub fn unpack_awq_qweight_k_n8_to_n_major(
    qweight_k_by_n8: &[u32],
    size_k: usize,
    size_n: usize,
) -> Vec<u32> {
    const UNDO_AWQ: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let k8 = size_k / 8;
    let n8 = size_n / 8;
    assert_eq!(qweight_k_by_n8.len(), size_k * n8);
    let mut out = vec![0u32; size_n * k8];
    for k_word in 0..k8 {
        for b in 0..8 {
            let k = k_word * 8 + b;
            let src_row = &qweight_k_by_n8[k * n8..(k + 1) * n8];
            let shift_out = b * 4;
            for (n_pack, &awq_word) in src_row.iter().enumerate() {
                let n_base = n_pack * 8;
                for (c, &src_slot) in UNDO_AWQ.iter().enumerate() {
                    let nib = (awq_word >> (src_slot * 4)) & 0xF;
                    out[(n_base + c) * k8 + k_word] |= nib << shift_out;
                }
            }
        }
    }
    out
}

/// Unpacks an AutoAWQ `gemm` `[K / group_size, N / 8]` `qzeros` tensor (interleaved
/// `[0, 2, 4, 6, 1, 3, 5, 7]` along `N`) into N-major `[N / 8, K / group_size]` `u32` words
/// matching `compressed-tensors` `.weight_zero_point` layout (8 consecutive N-rows per `u32`).
pub fn unpack_awq_qzeros_kg_n8_to_n_major(
    qzeros_kg_by_n8: &[u32],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u32> {
    const UNDO_AWQ: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let num_groups = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(qzeros_kg_by_n8.len(), num_groups * n8);
    let mut out = vec![0u32; n8 * num_groups];
    for g in 0..num_groups {
        for n_pack in 0..n8 {
            let awq_word = qzeros_kg_by_n8[g * n8 + n_pack];
            let mut packed_col = 0u32;
            for (c, &src_slot) in UNDO_AWQ.iter().enumerate() {
                let nib = (awq_word >> (src_slot * 4)) & 0xF;
                packed_col |= nib << (c * 4);
            }
            out[n_pack * num_groups + g] = packed_col;
        }
    }
    out
}

/// Permutes an N-major `[N / 8, K / group_size]` packed 4-bit zero-point tensor into `vLLM`
/// Marlin `kU4` zero-point layout (`[K / group_size, N / 8]` `u32` words with `scale_perm` +
/// `[0, 2, 4, 6, 1, 3, 5, 7]` interleave).
pub fn marlin_permute_zero_points_from_n_major(
    qzeros_n8_by_kg: &[u32],
    size_k: usize,
    size_n: usize,
    group_size: usize,
) -> Vec<u32> {
    const INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let num_groups = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(size_n % 64, 0);
    assert_eq!(qzeros_n8_by_kg.len(), n8 * num_groups);

    let mut zp_g_by_n = vec![0u8; num_groups * size_n];
    for n_pack in 0..n8 {
        for g in 0..num_groups {
            let w = qzeros_n8_by_kg[n_pack * num_groups + g];
            let n_base = n_pack * 8;
            for b in 0..8 {
                zp_g_by_n[g * size_n + n_base + b] = ((w >> (b * 4)) & 0xF) as u8;
            }
        }
    }

    let mut permed = vec![0u8; zp_g_by_n.len()];
    let mut scale_perm = [0usize; 64];
    for i in 0..8 {
        for j in 0..8 {
            scale_perm[i * 8 + j] = i + 8 * j;
        }
    }
    for (chunk_idx, src_chunk) in zp_g_by_n.as_chunks::<64>().0.iter().enumerate() {
        let dst_base = chunk_idx * 64;
        for (i, &p) in scale_perm.iter().enumerate() {
            permed[dst_base + i] = src_chunk[p];
        }
    }

    let mut out = vec![0u32; num_groups * n8];
    for (word_idx, chunk8) in permed.as_chunks::<8>().0.iter().enumerate() {
        let mut packed = 0u32;
        for (i, &src_i) in INTERLEAVE.iter().enumerate() {
            packed |= ((chunk8[src_i] & 0xF) as u32) << (i * 4);
        }
        out[word_idx] = packed;
    }
    out
}

/// Dequantizes an N-major `[N, K/8]` asymmetric `U4` weight + `[N, K/group_size]` `FP16` scale
/// + `[N/8, K/group_size]` packed 4-bit zero-point tensor into an N-major `[N, K]` `FP16` matrix.
pub fn dequant_u4_zp_n_major_to_f16(
    packed_n_by_k8: &[u32],
    scales_n_by_kg: &[u16],
    qzeros_n8_by_kg: &[u32],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let words_per_row = size_k / 8;
    let groups_per_row = size_k / group_size;
    let n8 = size_n / 8;
    assert_eq!(size_n % 8, 0);
    assert_eq!(packed_n_by_k8.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    assert_eq!(qzeros_n8_by_kg.len(), n8 * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    for n in 0..size_n {
        let n_pack = n >> 3;
        let n_shift = (n & 7) * 4;
        for w_idx in 0..words_per_row {
            let word = packed_n_by_k8[n * words_per_row + w_idx];
            for b in 0..8 {
                let k = w_idx * 8 + b;
                let g = k / group_size;
                let nib = ((word >> (b * 4)) & 0xF) as i32;
                let zp = ((qzeros_n8_by_kg[n_pack * groups_per_row + g] >> n_shift) & 0xF) as i32;
                let scale = f16_to_f32(scales_n_by_kg[n * groups_per_row + g]);
                out[n * size_k + k] = f32_to_f16(((nib - zp) as f32) * scale);
            }
        }
    }
    out
}

/// Dequantizes an N-major `[N, K/4]` symmetric `U8B128` weight + `[N, K/group_size]` `FP16` scale
/// tensor into an N-major `[N, K]` `FP16` matrix.
pub fn dequant_u8b128_n_major_to_f16(
    packed_n_by_k4: &[u32],
    scales_n_by_kg: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> Vec<u16> {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    let words_per_row = size_k / 4;
    let groups_per_row = size_k / eff_gs;
    assert_eq!(packed_n_by_k4.len(), size_n * words_per_row);
    assert_eq!(scales_n_by_kg.len(), size_n * groups_per_row);
    let mut out = vec![0u16; size_n * size_k];
    let dequant_rows = |row_start: usize, dst_rows: &mut [u16]| {
        for (rel_n, dst_row) in dst_rows.chunks_exact_mut(size_k).enumerate() {
            let n = row_start + rel_n;
            let row_words = &packed_n_by_k4[n * words_per_row..(n + 1) * words_per_row];
            let row_scales = &scales_n_by_kg[n * groups_per_row..(n + 1) * groups_per_row];
            if eff_gs >= 256 && eff_gs.is_multiple_of(4) {
                let words_per_group = eff_gs / 4;
                for (g, &sc_bits) in row_scales.iter().enumerate() {
                    let scale = f16_to_f32(sc_bits);
                    let mut lut = [0u16; 256];
                    for (u, slot) in lut.iter_mut().enumerate() {
                        *slot = f32_to_f16(((u as i32 - 128) as f32) * scale);
                    }
                    let g_words = &row_words[g * words_per_group..(g + 1) * words_per_group];
                    let g_dst = &mut dst_row[g * eff_gs..(g + 1) * eff_gs];
                    for (w_idx, &word) in g_words.iter().enumerate() {
                        let base = w_idx * 4;
                        let bytes = word.to_le_bytes();
                        g_dst[base] = lut[bytes[0] as usize];
                        g_dst[base + 1] = lut[bytes[1] as usize];
                        g_dst[base + 2] = lut[bytes[2] as usize];
                        g_dst[base + 3] = lut[bytes[3] as usize];
                    }
                }
            } else if eff_gs.is_multiple_of(4) {
                let words_per_group = eff_gs / 4;
                for (g, &sc_bits) in row_scales.iter().enumerate() {
                    let scale = f16_to_f32(sc_bits);
                    let g_words = &row_words[g * words_per_group..(g + 1) * words_per_group];
                    let g_dst = &mut dst_row[g * eff_gs..(g + 1) * eff_gs];
                    for (w_idx, &word) in g_words.iter().enumerate() {
                        let base = w_idx * 4;
                        let bytes = word.to_le_bytes();
                        for b in 0..4 {
                            let q = bytes[b] as i32 - 128;
                            g_dst[base + b] = f32_to_f16((q as f32) * scale);
                        }
                    }
                }
            } else {
                for (w_idx, &word) in row_words.iter().enumerate() {
                    for b in 0..4 {
                        let k = w_idx * 4 + b;
                        let q = ((word >> (b * 8)) & 0xFF) as i32 - 128;
                        let scale = f16_to_f32(row_scales[k / eff_gs]);
                        dst_row[k] = f32_to_f16((q as f32) * scale);
                    }
                }
            }
        }
    };
    if size_n < 1024 {
        dequant_rows(0, &mut out);
    } else {
        let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get().clamp(1, 16));
        let rows_per_chunk = size_n.div_ceil(num_threads).max(1);
        std::thread::scope(|s| {
            for (chunk_idx, dst_chunk) in out.chunks_mut(rows_per_chunk * size_k).enumerate() {
                let row_start = chunk_idx * rows_per_chunk;
                s.spawn(move || dequant_rows(row_start, dst_chunk));
            }
        });
    }
    out
}

/// Quantizes an N-major `[N, K]` `FP16` matrix into symmetric `U8B128` (`[N, K/4]` `u32` words
/// with `+128` zero-point bias + `[N, K/group_size]` `FP16` per-group scales).
pub fn quantize_f16_n_major_to_u8b128(
    weight_f16: &[u16],
    size_n: usize,
    size_k: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<u16>) {
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    assert!(
        size_k >= 4
            && size_k.is_multiple_of(4)
            && size_k.is_multiple_of(eff_gs)
            && eff_gs.is_multiple_of(4)
    );
    assert_eq!(weight_f16.len(), size_n * size_k);
    let words_per_row = size_k / 4;
    let groups_per_row = size_k / eff_gs;
    let words_per_group = eff_gs / 4;
    let mut packed = vec![0u32; size_n * words_per_row];
    let mut scales = vec![0u16; size_n * groups_per_row];

    let quant_rows = |src_rows: &[u16], dst_words: &mut [u32], dst_scales: &mut [u16]| {
        for ((src_row, row_words), row_scales) in src_rows
            .chunks_exact(size_k)
            .zip(dst_words.chunks_exact_mut(words_per_row))
            .zip(dst_scales.chunks_exact_mut(groups_per_row))
        {
            for g in 0..groups_per_row {
                let g_src = &src_row[g * eff_gs..(g + 1) * eff_gs];
                let mut max_abs = 0.0f32;
                for &bits in g_src {
                    let v = f16_to_f32(bits).abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                let sc_bits = f32_to_f16(max_abs / 127.0);
                row_scales[g] = sc_bits;
                let sc = f16_to_f32(sc_bits);
                let inv = if sc > 0.0 && sc.is_finite() {
                    1.0 / sc
                } else {
                    0.0
                };
                let g_words = &mut row_words[g * words_per_group..(g + 1) * words_per_group];
                for (w_idx, slot) in g_words.iter_mut().enumerate() {
                    let base = w_idx * 4;
                    let mut bytes = [128u8; 4];
                    for b in 0..4 {
                        let w = f16_to_f32(g_src[base + b]);
                        let q = (w * inv).round_ties_even() as i32 + 128;
                        bytes[b] = q.clamp(0, 255) as u8;
                    }
                    *slot = u32::from_le_bytes(bytes);
                }
            }
        }
    };

    if size_n * size_k < 1_048_576 || size_n < 16 {
        quant_rows(weight_f16, &mut packed, &mut scales);
    } else {
        let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get().clamp(1, 16));
        let rows_per_chunk = size_n.div_ceil(num_threads).max(1);
        std::thread::scope(|s| {
            for ((src_chunk, words_chunk), scales_chunk) in weight_f16
                .chunks(rows_per_chunk * size_k)
                .zip(packed.chunks_mut(rows_per_chunk * words_per_row))
                .zip(scales.chunks_mut(rows_per_chunk * groups_per_row))
            {
                s.spawn(move || quant_rows(src_chunk, words_chunk, scales_chunk));
            }
        });
    }
    (packed, scales)
}

/// Validates that an optional GPTQ `.g_idx` tensor represents static sequential group assignment
/// (`g_idx[k] == k / group_size`, or all `0` for channelwise) rather than an unapplied `desc_act`
/// channel permutation.
pub fn validate_static_gptq_g_idx(
    g_idx_words: &[u32],
    size_k: usize,
    group_size: usize,
    tensor_name: &str,
) -> io::Result<()> {
    if g_idx_words.len() != size_k {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid g_idx length {} for '{tensor_name}' (expected K={size_k})",
                g_idx_words.len()
            ),
        ));
    }
    if g_idx_words.iter().all(|&x| x == 0) {
        return Ok(());
    }
    let eff_gs = if group_size == 0 { size_k } else { group_size };
    for (k, &raw_g) in g_idx_words.iter().enumerate() {
        let expected = (k / eff_gs) as u32;
        if raw_g != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor '{tensor_name}' contains non-sequential g_idx[{k}]={} (expected {expected}); activation-order (desc_act) channel permutation is not supported",
                    raw_g as i32
                ),
            ));
        }
    }
    Ok(())
}

/// Repacks an N-major `[N, K/128, 26]` `PTQ1_0` 5-trit tensor and `[N, K/128]` `FP16` scales
/// into the GPU-coalesced `[N/8, K/128, 7, 8]` `u32` layout consumed by `warp34_gemm_5trit_f16`.
///
/// Each `(n, g)` 28-byte block fuses 26 trit bytes (little-endian into `w[0..6]`, with bytes
/// `24..25` in the low 16 bits of `w[6]`) and the 2-byte `FP16` scale in the high 16 bits of `w[6]`.
pub fn repack_5trit_n_major_to_gpu(
    packed_5trit: &[u8],
    scales_f16: &[u16],
    size_n: usize,
    size_k: usize,
) -> Vec<u32> {
    assert!(size_n.is_multiple_of(8), "5-trit N must be a multiple of 8");
    assert!(
        size_k.is_multiple_of(128),
        "5-trit K must be a multiple of 128"
    );
    let groups = size_k / 128;
    assert_eq!(packed_5trit.len(), size_n * groups * 26);
    assert_eq!(scales_f16.len(), size_n * groups);
    if size_n == 0 || size_k == 0 {
        return Vec::new();
    }
    let n_blks = size_n / 8;
    let mut out = vec![0u32; n_blks * groups * 56];
    let repack_range = |blk_start: usize, dst: &mut [u32]| {
        for (rel_blk, blk_dst) in dst.chunks_exact_mut(groups * 56).enumerate() {
            let n_blk = blk_start + rel_blk;
            for g in 0..groups {
                let g_dst = &mut blk_dst[g * 56..(g + 1) * 56];
                for r in 0..8 {
                    let n = n_blk * 8 + r;
                    let src_off = (n * groups + g) * 26;
                    let b = &packed_5trit[src_off..src_off + 26];
                    for w in 0..6 {
                        let o = w * 4;
                        g_dst[w * 8 + r] = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                    }
                    let sc = scales_f16[n * groups + g] as u32;
                    g_dst[6 * 8 + r] = (sc << 16) | ((b[25] as u32) << 8) | (b[24] as u32);
                }
            }
        }
    };
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(n_blks);
    if workers <= 1 || n_blks < 16 {
        repack_range(0, &mut out);
    } else {
        let blks_per_chunk = n_blks.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(blks_per_chunk * groups * 56).enumerate() {
                let blk_start = ci * blks_per_chunk;
                s.spawn(move || repack_range(blk_start, chunk));
            }
        });
    }
    out
}

#[inline]
fn decode_5trit_group_into_f16(bytes26: &[u8], scale_f16: u16, dst128: &mut [u16]) {
    const POW3: [u32; 5] = [1, 3, 9, 27, 81];
    let s = f16_to_f32(scale_f16);
    let vals = [f32_to_f16(-s), 0u16, f32_to_f16(s)];
    // Segment 0: bytes 0..16 -> 16 x 5 = 80 trits @ stride 16
    for i in 0..16 {
        let q = bytes26[i] as u32;
        for (n, &p3) in POW3.iter().enumerate() {
            let u = (((q * p3) & 0xFF) * 3) >> 8;
            dst128[n * 16 + i] = vals[u as usize];
        }
    }
    // Segment 1: bytes 16..24 -> 8 x 5 = 40 trits @ stride 8
    for i in 0..8 {
        let q = bytes26[16 + i] as u32;
        for (n, &p3) in POW3.iter().enumerate() {
            let u = (((q * p3) & 0xFF) * 3) >> 8;
            dst128[80 + n * 8 + i] = vals[u as usize];
        }
    }
    // Segment 2: bytes 24..26 -> 2 x 4 = 8 trits @ stride 2
    for i in 0..2 {
        let q = bytes26[24 + i] as u32;
        for (n, &p3) in POW3[..4].iter().enumerate() {
            let u = (((q * p3) & 0xFF) * 3) >> 8;
            dst128[120 + n * 2 + i] = vals[u as usize];
        }
    }
}

/// Dequantizes an N-major `[N, K/128, 26]` `PTQ1_0` 5-trit tensor + `[N, K/128]` `FP16` scales
/// into an N-major `[N, K]` `FP16` matrix.
pub fn dequant_5trit_n_major_to_f16(
    packed_5trit: &[u8],
    scales_f16: &[u16],
    size_n: usize,
    size_k: usize,
) -> Vec<u16> {
    assert!(size_k.is_multiple_of(128));
    let groups = size_k / 128;
    assert_eq!(packed_5trit.len(), size_n * groups * 26);
    assert_eq!(scales_f16.len(), size_n * groups);
    if size_n == 0 || size_k == 0 {
        return Vec::new();
    }
    let mut out = vec![0u16; size_n * size_k];
    let dequant_rows = |row_start: usize, dst_rows: &mut [u16]| {
        for (rel_n, dst_row) in dst_rows.chunks_exact_mut(size_k).enumerate() {
            let n = row_start + rel_n;
            for g in 0..groups {
                let off = (n * groups + g) * 26;
                decode_5trit_group_into_f16(
                    &packed_5trit[off..off + 26],
                    scales_f16[n * groups + g],
                    &mut dst_row[g * 128..(g + 1) * 128],
                );
            }
        }
    };
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(size_n);
    if workers <= 1 || size_n < 128 {
        dequant_rows(0, &mut out);
    } else {
        let rows_per_chunk = size_n.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(rows_per_chunk * size_k).enumerate() {
                let row_start = ci * rows_per_chunk;
                s.spawn(move || dequant_rows(row_start, chunk));
            }
        });
    }
    out
}

/// Dequantizes a GPU-repacked `[N/8, K/128, 7, 8]` `PTQ1_0` 5-trit matrix back to N-major `[N, K]` `FP16`.
pub fn dequant_5trit_gpu_to_f16(packed_gpu: &[u32], size_n: usize, size_k: usize) -> Vec<u16> {
    assert!(size_n.is_multiple_of(8) && size_k.is_multiple_of(128));
    let groups = size_k / 128;
    let n_blks = size_n / 8;
    assert_eq!(packed_gpu.len(), n_blks * groups * 56);
    let mut out = vec![0u16; size_n * size_k];
    for n_blk in 0..n_blks {
        for g in 0..groups {
            let g_src = &packed_gpu[(n_blk * groups + g) * 56..][..56];
            for r in 0..8 {
                let n = n_blk * 8 + r;
                let mut b = [0u8; 26];
                for w in 0..6 {
                    let bytes = g_src[w * 8 + r].to_le_bytes();
                    b[w * 4..w * 4 + 4].copy_from_slice(&bytes);
                }
                let w6 = g_src[6 * 8 + r];
                b[24] = (w6 & 0xFF) as u8;
                b[25] = ((w6 >> 8) & 0xFF) as u8;
                let sc = (w6 >> 16) as u16;
                decode_5trit_group_into_f16(&b, sc, &mut out[n * size_k + g * 128..][..128]);
            }
        }
    }
    out
}
