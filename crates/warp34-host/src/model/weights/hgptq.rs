//! HGPTQ metadata stays attached to the weight through sharding and fusion.
//! Runtime rotation uses caller-owned scratch, never the shared source activation.

use super::*;
use crate::cuda::f32_to_f16;

#[cfg(test)]
#[path = "hgptq_tests.rs"]
mod tests;

impl RawLinearHost {
    pub(super) fn load_with_transform(
        archive: &SafetensorsArchive,
        prefix: &str,
        quant: &QuantScheme,
    ) -> Result<Self> {
        Self::load_with_transform_row_range(archive, prefix, quant, None).map(|(raw, _)| raw)
    }

    pub(super) fn load_with_transform_row_range(
        archive: &SafetensorsArchive,
        prefix: &str,
        quant: &QuantScheme,
        row_range: Option<(usize, usize)>,
    ) -> Result<(Self, usize)> {
        let name = ["weight_5trit", "weight_packed", "qweight", "weight"]
            .into_iter()
            .find_map(|suffix| {
                archive
                    .resolve_name(&format!("{prefix}.{suffix}"))
                    .map(|s| s.trim_end_matches(&format!(".{suffix}")).to_owned())
            })
            .unwrap_or_else(|| prefix.to_owned());
        let resolved = quant
            .for_linear(&name)
            .map_err(|e| loader_err_msg(e.to_string()))?;
        let block_and_signs = quant
            .input_hadamard_with_signs(&name)
            .map_err(|e| loader_err_msg(e.to_string()))?;
        let (raw, full_n) =
            Self::load_untransformed_row_range(archive, prefix, &resolved, row_range)?;
        let Some((block_size, pre_signs)) = block_and_signs else {
            return Ok((raw, full_n));
        };
        let (n, k, group_size) = match &raw {
            Self::U2B2NMajor {
                n, k, group_size, ..
            }
            | Self::U4B8NMajor {
                n, k, group_size, ..
            } => (*n, *k, *group_size),
            Self::Ternary5Trit { n, k, .. } => (*n, *k, 128),
            _ => {
                return Err(loader_err_msg(format!(
                    "HGPTQ requires symmetric packed W2/W4/5-Trit weights: {name}"
                )));
            }
        };
        if n == 0
            || k == 0
            || !k.is_multiple_of(block_size)
            || pre_signs.as_ref().is_some_and(|s| s.len() != k)
            || n > i32::MAX as usize - MARLIN_TILE_K_W4
            || k > i32::MAX as usize - MARLIN_TILE_K_W4
            || group_size != resolved.group_size_or(k)
        {
            return Err(loader_err_msg(format!(
                "HGPTQ geometry/group mismatch for {name}: N={n} K={k} group={group_size} block={block_size} pre_signs={:?}",
                pre_signs.as_ref().map(|s| s.len())
            )));
        }
        Ok((
            Self::Hadamard {
                inner: Box::new(raw),
                block_size,
                pre_signs,
            },
            full_n,
        ))
    }

    pub(super) fn concat_transformed_rows(parts: &[&Self]) -> Result<Self> {
        let Some(Self::Hadamard {
            block_size,
            pre_signs,
            ..
        }) = parts.first().copied()
        else {
            return Err(loader_err(
                "cannot fuse transformed and untransformed projections",
            ));
        };
        let inner: Result<Vec<&Self>> = parts
            .iter()
            .map(|part| match part {
                Self::Hadamard {
                    inner,
                    block_size: b,
                    pre_signs: ps,
                } if b == block_size && ps == pre_signs => Ok(inner.as_ref()),
                _ => Err(loader_err(
                    "cannot fuse projections with different input transforms",
                )),
            })
            .collect();
        Ok(Self::Hadamard {
            inner: Box::new(Self::concat_row_refs(&inner?)?),
            block_size: *block_size,
            pre_signs: pre_signs.clone(),
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn upload_hadamard_with(
        &self,
        block_size: usize,
        pre_signs: Option<&[i8]>,
        arena: &mut DeviceArena,
        stream: &CudaStream,
        staging: &mut UploadStaging<'_>,
    ) -> Result<LinearProj> {
        let (n, k) = self.dims();
        let padded;
        let storage = if matches!(self, Self::Ternary5Trit { .. })
            || (n.is_multiple_of(MARLIN_TILE_N) && k.is_multiple_of(MARLIN_TILE_K_W4))
        {
            self
        } else {
            padded = self.pad_hadamard_weight()?;
            &padded
        };
        let inner = storage.upload_with(arena, stream, staging)?;
        if !matches!(
            inner,
            LinearProj::MarlinW2A16 { .. }
                | LinearProj::MarlinW4A16 { .. }
                | LinearProj::Ternary5Trit { .. }
        ) {
            return Err(loader_err(
                "HGPTQ must remain native W2/W4/5-Trit after upload",
            ));
        }
        let pre_signs_ptr = match pre_signs {
            Some(signs) => arena.upload_slice(signs, stream)?,
            None => 0,
        };
        Ok(LinearProj::Hadamard {
            inner: Box::new(inner),
            size_n: n,
            size_k: k,
            block_size,
            pre_signs_ptr,
        })
    }

    // Marlin accepts N multiples of 64 with K multiples of 128. Padding is in
    // the rotated basis: nibble 8 is U4B8 zero (crumb 2 is U2B2 zero), while
    // code_offset=0.5 has no integer zero code so padded rows/groups use scale 0.
    #[cfg(any(feature = "cuda", test))]
    pub(super) fn pad_hadamard_weight(&self) -> Result<Self> {
        let (packed, scales, n, k, group_size, is_u2, code_offset_half) = match self {
            Self::U2B2NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
                code_offset_half,
            } => (packed, scales, *n, *k, *group_size, true, *code_offset_half),
            Self::U4B8NMajor {
                packed,
                scales,
                n,
                k,
                group_size,
            } => (packed, scales, *n, *k, *group_size, false, false),
            _ => return Err(loader_err("HGPTQ padding expects symmetric W2/W4")),
        };
        let pn = n.div_ceil(MARLIN_TILE_N) * MARLIN_TILE_N;
        let pk = k.div_ceil(MARLIN_TILE_K_W4) * MARLIN_TILE_K_W4;
        let channelwise = group_size == k;
        if !channelwise && !matches!(group_size, 32 | 64 | 128) {
            return Err(loader_err("unsupported HGPTQ group size"));
        }
        if code_offset_half && pk != k && (channelwise || !k.is_multiple_of(group_size)) {
            return Err(loader_err(
                "code_offset=0.5 HGPTQ K-padding requires K divisible by group_size (no zero code inside a live group)",
            ));
        }
        let groups = if channelwise { 1 } else { pk / group_size };
        let old_groups = k / group_size;
        let pack_factor = if is_u2 {
            U2_CRUMBS_PER_U32
        } else {
            U4_NIBBLES_PER_U32
        };
        let zero_word = if is_u2 {
            U2B2_PACKED_ZERO_WORD
        } else {
            U4B8_PACKED_ZERO_WORD
        };
        let words_pk = pk / pack_factor;
        let words_k = k / pack_factor;
        let mut padded = vec![zero_word; pn * words_pk];
        let pad_scale = if code_offset_half {
            0u16
        } else {
            f32_to_f16(1.0)
        };
        let mut padded_scales = vec![pad_scale; pn * groups];
        for row in 0..n {
            padded[row * words_pk..row * words_pk + words_k]
                .copy_from_slice(&packed[row * words_k..(row + 1) * words_k]);
            padded_scales[row * groups..row * groups + old_groups]
                .copy_from_slice(&scales[row * old_groups..(row + 1) * old_groups]);
        }
        let new_gs = if channelwise { pk } else { group_size };
        if is_u2 {
            Ok(Self::U2B2NMajor {
                packed: padded,
                scales: padded_scales,
                n: pn,
                k: pk,
                group_size: new_gs,
                code_offset_half,
            })
        } else {
            Ok(Self::U4B8NMajor {
                packed: padded,
                scales: padded_scales,
                n: pn,
                k: pk,
                group_size: new_gs,
            })
        }
    }

    pub(crate) fn unwrap_hadamard(self) -> Result<(Self, usize)> {
        match self {
            Self::Hadamard {
                inner,
                block_size,
                pre_signs,
            } => {
                if pre_signs.is_some() {
                    return Err(loader_err(
                        "signed Hadamard pre-signs are not supported on routed MoE expert projections",
                    ));
                }
                Ok((*inner, block_size))
            }
            other => Ok((other, 0)),
        }
    }
}

/// Applies `row -> H_{block}(row) * post_signs` in FP32 and converts back to FP16.
/// Used for both host-side HGPTQ unrotation and `warp_quant_embed_w2` post-dequantization.
pub(crate) fn hadamard_f16_rows_with_post_signs(
    dense: &mut [u16],
    block: usize,
    post_signs: Option<&[i8]>,
) {
    let inv_sqrt = 1.0f32 / (block as f32).sqrt();
    let mut values = vec![0.0f32; block];
    for (chunk_idx, chunk) in dense.chunks_exact_mut(block).enumerate() {
        for (v, &bits) in values.iter_mut().zip(chunk.iter()) {
            *v = f16_to_f32(bits);
        }
        let mut step = 1;
        while step < block {
            for base in (0..block).step_by(2 * step) {
                for i in 0..step {
                    let a = values[base + i];
                    let b = values[base + i + step];
                    values[base + i] = a + b;
                    values[base + i + step] = a - b;
                }
            }
            step *= 2;
        }
        if let Some(signs) = post_signs {
            let off = (chunk_idx * block) % signs.len();
            for ((bits, &v), &s) in chunk
                .iter_mut()
                .zip(&values)
                .zip(signs[off..off + block].iter())
            {
                *bits = f32_to_f16(v * inv_sqrt * (s as f32));
            }
        } else {
            for (bits, &v) in chunk.iter_mut().zip(&values) {
                *bits = f32_to_f16(v * inv_sqrt);
            }
        }
    }
}

/// Materializing a logical weight must undo its offline basis change too.
/// This path is for host diagnostics; GPU upload never dequantizes HGPTQ.
pub(super) fn unrotate_weights(
    inner: &RawLinearHost,
    block: usize,
    pre_signs: Option<&[i8]>,
) -> Vec<u16> {
    let mut dense = inner.to_f16_vec();
    hadamard_f16_rows_with_post_signs(&mut dense, block, pre_signs);
    dense
}

#[cfg(feature = "cuda")]
impl LinearProj {
    /// Extra workspace beyond existing GEMM workspace, for this projection.
    pub fn input_transform_workspace_bytes(&self, rows: usize) -> Option<usize> {
        let trit5_prefill_bytes = |proj: &Self| -> Option<usize> {
            if rows <= 16 {
                return Some(0);
            }
            if let Self::Ternary5Trit { size_n, size_k, .. } = proj {
                const MAX_STRIP_BYTES: usize = 16 * 1024 * 1024;
                let row_bytes = size_k.checked_mul(2)?;
                let max_rows = (MAX_STRIP_BYTES / row_bytes).max(8) & !7;
                let strip_rows = (*size_n).min(max_rows);
                align_up_scratch(strip_rows.checked_mul(row_bytes)?)
            } else {
                Some(0)
            }
        };
        match self {
            Self::Hadamard { inner, size_n, .. } => {
                let input = align_up_scratch(rows.checked_mul(inner.size_k())?.checked_mul(2)?)?;
                let output = if *size_n != inner.size_n() {
                    align_up_scratch(rows.checked_mul(inner.size_n())?.checked_mul(2)?)?
                } else {
                    0
                };
                input
                    .checked_add(output)?
                    .checked_add(trit5_prefill_bytes(inner)?)
            }
            Self::Ternary5Trit { .. } => trit5_prefill_bytes(self),
            _ => Some(0),
        }
    }

    /// Whether this projection is a `block_size` Hadamard projection with unpadded `K` and `N`,
    /// allowing a producer or shared-input kernel to emit `H_{block_size}`-rotated activations directly.
    #[inline]
    pub fn can_fuse_hadamard_from_producer(&self, target_block_size: usize) -> bool {
        target_block_size > 0
            && matches!(
                self,
                Self::Hadamard {
                    inner,
                    size_n,
                    size_k,
                    block_size,
                    pre_signs_ptr,
                } if *block_size == target_block_size
                    && *pre_signs_ptr == 0
                    && *size_k == inner.size_k()
                    && *size_n == inner.size_n()
                    && size_k.is_multiple_of(target_block_size)
            )
    }

    /// Whether this projection is a `B == 64` Hadamard projection with unpadded `K` and `N`,
    /// allowing its producer kernel to emit `H_64`-rotated activations directly.
    #[inline]
    pub fn can_fuse_h64_from_producer(&self) -> bool {
        self.can_fuse_hadamard_from_producer(HADAMARD_BLOCK_64)
    }

    /// Whether this projection is a `B == 512` Hadamard projection with unpadded `K` and `N`,
    /// allowing its producer kernel to emit `H_512`-rotated activations directly.
    #[inline]
    pub fn can_fuse_h512_from_producer(&self) -> bool {
        self.can_fuse_hadamard_from_producer(512)
    }

    /// Whether this projection is a `B == 1024` Hadamard projection with unpadded `K` and `N`,
    /// allowing its producer kernel to emit `H_1024`-rotated activations directly.
    #[inline]
    pub fn can_fuse_h1024_from_producer(&self) -> bool {
        self.can_fuse_hadamard_from_producer(1024)
    }

    /// Executes the underlying GEMM directly when `input` has already been rotated by a
    /// producer-fused Hadamard kernel (`H_64`, `H_128`, `H_512`, or `H_1024`), or falls back to `forward` otherwise.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn forward_prerotated(
        &self,
        ex: &Warp34Exec,
        cublas: &CublasHandle,
        marlin_ws_ptr: u64,
        marlin_ws_ints: usize,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        match self {
            Self::Hadamard {
                inner,
                size_n,
                size_k,
                block_size,
                pre_signs_ptr,
            } if *block_size > 0
                && *pre_signs_ptr == 0
                && *size_k == inner.size_k()
                && *size_n == inner.size_n()
                && size_k.is_multiple_of(*block_size) =>
            {
                let input_bytes = rows
                    .saturating_mul(*size_k)
                    .saturating_mul(2)
                    .next_multiple_of(256);
                let sub_ex = if input == ex.scratch_ptr && ex.scratch_ptr != 0 {
                    if ex.scratch_size >= input_bytes {
                        Warp34Exec {
                            stream: ex.stream,
                            scratch_ptr: ex.scratch_ptr + input_bytes as u64,
                            scratch_size: ex.scratch_size - input_bytes,
                            flags: ex.flags,
                        }
                    } else {
                        Warp34Exec {
                            stream: ex.stream,
                            scratch_ptr: 0,
                            scratch_size: 0,
                            flags: ex.flags,
                        }
                    }
                } else {
                    *ex
                };
                inner.forward(
                    &sub_ex,
                    cublas,
                    marlin_ws_ptr,
                    marlin_ws_ints,
                    input,
                    output,
                    rows,
                )
            }
            _ => self.forward(
                ex,
                cublas,
                marlin_ws_ptr,
                marlin_ws_ints,
                input,
                output,
                rows,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_hadamard(
        inner: &Self,
        n: usize,
        k: usize,
        block: usize,
        pre_signs_ptr: u64,
        ex: &Warp34Exec,
        cublas: &CublasHandle,
        locks: u64,
        lock_count: usize,
        input: u64,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        Self::forward_hadamard_with_bias(
            inner,
            n,
            k,
            block,
            pre_signs_ptr,
            ex,
            cublas,
            locks,
            lock_count,
            input,
            None,
            output,
            rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_hadamard_with_bias(
        inner: &Self,
        n: usize,
        k: usize,
        block: usize,
        pre_signs_ptr: u64,
        ex: &Warp34Exec,
        cublas: &CublasHandle,
        locks: u64,
        lock_count: usize,
        input: u64,
        bias: Option<u64>,
        output: u64,
        rows: usize,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let m = i32::try_from(rows).map_err(|_| loader_err("HGPTQ row count exceeds i32"))?;
        let input_bytes = rows
            .checked_mul(inner.size_k())
            .and_then(|v| v.checked_mul(2))
            .and_then(align_up_scratch)
            .ok_or_else(|| loader_err("HGPTQ input scratch overflow"))?;
        let output_bytes = if n != inner.size_n() {
            rows.checked_mul(inner.size_n())
                .and_then(|v| v.checked_mul(2))
                .and_then(align_up_scratch)
                .ok_or_else(|| loader_err("HGPTQ output scratch overflow"))?
        } else {
            0
        };
        let bytes = input_bytes
            .checked_add(output_bytes)
            .ok_or_else(|| loader_err("HGPTQ scratch overflow"))?;
        if ex.scratch_ptr == 0 || ex.scratch_size < bytes {
            return Err(loader_err_msg(format!(
                "HGPTQ scratch needs {bytes} bytes, available {}",
                ex.scratch_size
            )));
        }
        let tmp_out = if output_bytes == 0 {
            output
        } else {
            ex.scratch_ptr + input_bytes as u64
        };
        if pre_signs_ptr != 0 {
            check_rc(
                // SAFETY: caller supplies disjoint live activation/output/workspace buffers and
                // a live `block`-element `int8` sign vector; capacity is checked above.
                unsafe {
                    ffi::warp34_block_hadamard_signed_f16(
                        ex,
                        input as *const c_void,
                        pre_signs_ptr as *const i8,
                        ex.scratch_ptr as *mut c_void,
                        m,
                        k as i32,
                        inner.size_k() as i32,
                        block as i32,
                    )
                },
                "HGPTQ signed block Hadamard",
            )?;
        } else {
            check_rc(
                // SAFETY: caller supplies disjoint live activation/output/workspace buffers;
                // capacity is checked above. Loader validates block-aligned logical K.
                unsafe {
                    ffi::warp34_block_hadamard_f16(
                        ex,
                        input as *const c_void,
                        ex.scratch_ptr as *mut c_void,
                        m,
                        k as i32,
                        inner.size_k() as i32,
                        block as i32,
                    )
                },
                "HGPTQ block Hadamard",
            )?;
        }
        let inner_ex = Warp34Exec {
            stream: ex.stream,
            scratch_ptr: ex.scratch_ptr + bytes as u64,
            scratch_size: ex.scratch_size - bytes,
            flags: ex.flags,
        };
        inner.forward(
            &inner_ex,
            cublas,
            locks,
            lock_count,
            ex.scratch_ptr,
            tmp_out,
            rows,
        )?;
        if output_bytes != 0 || bias.is_some() {
            let bias_ptr = bias.map_or(std::ptr::null(), |b| b as *const c_void);
            check_rc(
                // SAFETY: `tmp_out` and `output` have checked extents (`tmp_out == output` only
                // when `n == inner.size_n()`), and `bias_ptr` is null or a live `n`-element FP16 vector.
                unsafe {
                    ffi::warp34_linear_epilogue_f16(
                        ex,
                        tmp_out as *const c_void,
                        bias_ptr,
                        output as *mut c_void,
                        m,
                        n as i32,
                        inner.size_n() as i32,
                    )
                },
                "HGPTQ linear epilogue",
            )?;
        }
        Ok(())
    }
}
