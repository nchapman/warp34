//! L2 Host-RAM virtual memory swap pool (`HostKvSwapPool`) for [`super::KvPool`].
//!
//! Extends L1 GPU VRAM KV pages (`0..num_pages`) with lazy-allocated, layer-major
//! `256`-page (`4,096`-token) pinned host RAM slabs addressed by tagged page handles
//! (`>= HOST_KV_PAGE_BASE`). Because both [`super::PageBitmap`] and [`HostKvSwapPool`]
//! allocate ascending contiguous page runs, layer-major slab layout coalesces multi-page
//! swap-out (`D2H`) and swap-in (`H2D`) batches into a single `cudaMemcpyAsync` DMA burst
//! per layer per slab.

#[cfg(feature = "cuda")]
use core::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda")]
use warp34_kernels::ffi;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaDevice, CudaError, CudaStream, MappedHostBuffer, Result, check_rc};

#[cfg(feature = "cuda")]
use super::{KvLayerBuffer, PageBitmap};

/// Page-ID offset distinguishing L2 Host-RAM swapped KV pages (`>= 1_000_000`)
/// from L1 GPU VRAM KV pages (`0..num_pages`).
pub const HOST_KV_PAGE_BASE: i32 = 1_000_000;

/// Number of KV pages per lazy-allocated L2 Host-RAM slab (`256` pages = `4,096` tokens).
pub const HOST_KV_SLAB_PAGES: usize = 256;

/// Default process-wide L2 Host-RAM KV swap budget (`16 GiB`) across all tensor-parallel ranks.
pub const DEFAULT_HOST_KV_SWAP_BUDGET_BYTES: usize = 16 * 1024 * 1024 * 1024;

/// Maximum automatically planned L2 Host-RAM KV swap pages (`65,536` pages = `1,048,576` tokens).
pub const MAX_AUTO_HOST_KV_SWAP_PAGES: usize = 65_536;

/// Returns `true` when `page_id` addresses an L2 Host-RAM swapped KV page (`>= HOST_KV_PAGE_BASE`).
#[inline]
pub const fn is_host_kv_page(page_id: i32) -> bool {
    page_id >= HOST_KV_PAGE_BASE
}

/// Encodes a 0-based `HostKvSwapPool` index as a tagged `page_id`.
#[inline]
pub const fn encode_host_kv_page(host_idx: usize) -> i32 {
    HOST_KV_PAGE_BASE + (host_idx as i32)
}

/// Decodes a tagged `page_id` back to its 0-based `HostKvSwapPool` index.
#[inline]
pub const fn decode_host_kv_page(page_id: i32) -> Option<usize> {
    if page_id >= HOST_KV_PAGE_BASE {
        Some((page_id - HOST_KV_PAGE_BASE) as usize)
    } else {
        None
    }
}

/// Plans per-rank L2 Host-RAM KV swap capacity (`<= 65,536` pages, `<= 16 GiB` across all TP ranks).
pub fn plan_auto_host_kv_swap_pages(
    page_bundle_bytes: usize,
    tp_size: usize,
    available_ram_bytes: Option<usize>,
) -> usize {
    if page_bundle_bytes == 0 {
        return 0;
    }
    if let Ok(raw) = std::env::var("WARP34_HOST_KV_SWAP_MB")
        && let Ok(mb) = raw.trim().parse::<usize>()
    {
        if mb == 0 {
            return 0;
        }
        let per_rank = (mb.saturating_mul(1024 * 1024)) / tp_size.max(1);
        let pages = (per_rank / page_bundle_bytes).min(MAX_AUTO_HOST_KV_SWAP_PAGES);
        return if pages >= HOST_KV_SLAB_PAGES {
            (pages / HOST_KV_SLAB_PAGES) * HOST_KV_SLAB_PAGES
        } else {
            pages
        };
    }
    let total_budget = available_ram_bytes.map_or(DEFAULT_HOST_KV_SWAP_BUDGET_BYTES / 4, |avail| {
        (avail / 4).min(DEFAULT_HOST_KV_SWAP_BUDGET_BYTES)
    });
    let per_rank_budget = total_budget / tp_size.max(1);
    let pages = (per_rank_budget / page_bundle_bytes).min(MAX_AUTO_HOST_KV_SWAP_PAGES);
    if pages >= HOST_KV_SLAB_PAGES {
        (pages / HOST_KV_SLAB_PAGES) * HOST_KV_SLAB_PAGES
    } else {
        pages
    }
}

/// Contiguous `(vram_page, host_page)` run within a single L2 host slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoalescedSwapRun {
    pub vram_start: usize,
    pub slab_idx: usize,
    pub slab_page_start: usize,
    pub run_len: usize,
}

/// Coalesces parallel `(vram_pages, host_pages)` lists into contiguous slab-local DMA runs.
pub fn coalesce_swap_runs(
    vram_pages: &[i32],
    host_pages: &[i32],
    slab_pages: usize,
) -> Vec<CoalescedSwapRun> {
    assert_eq!(vram_pages.len(), host_pages.len());
    let slab_pages = slab_pages.max(1);
    let mut runs: Vec<CoalescedSwapRun> = Vec::new();
    for (&vram_pid, &host_pid) in vram_pages.iter().zip(host_pages.iter()) {
        let vram_idx = vram_pid as usize;
        let host_idx = decode_host_kv_page(host_pid).expect("valid tagged host KV page ID");
        let slab_idx = host_idx / slab_pages;
        let slab_page = host_idx % slab_pages;
        if let Some(last) = runs.last_mut()
            && last.slab_idx == slab_idx
            && last.vram_start + last.run_len == vram_idx
            && last.slab_page_start + last.run_len == slab_page
        {
            last.run_len += 1;
            continue;
        }
        runs.push(CoalescedSwapRun {
            vram_start: vram_idx,
            slab_idx,
            slab_page_start: slab_page,
            run_len: 1,
        });
    }
    runs
}

#[cfg(feature = "cuda")]
static WARNED_KV_SWAP_PINNED_FALLBACK: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "cuda")]
enum HostKvSlab {
    Pinned(MappedHostBuffer),
    Pageable(Box<[u8]>),
}

#[cfg(feature = "cuda")]
impl HostKvSlab {
    fn alloc(device_id: i32, len_bytes: usize) -> Result<Self> {
        let dev = CudaDevice { device_id };
        dev.make_current()?;
        match MappedHostBuffer::alloc(len_bytes) {
            Ok(pinned) => Ok(Self::Pinned(pinned)),
            Err(_) => {
                if !WARNED_KV_SWAP_PINNED_FALLBACK.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "warning: cudaHostAllocMapped unavailable for HostKvSwapPool slab on GPU {device_id} ({len_bytes} B); falling back to pageable host RAM"
                    );
                }
                Ok(Self::Pageable(vec![0u8; len_bytes].into_boxed_slice()))
            }
        }
    }

    #[inline]
    fn host_ptr(&self) -> *const u8 {
        match self {
            Self::Pinned(buf) => buf.host_ptr as *const u8,
            Self::Pageable(buf) => buf.as_ptr(),
        }
    }

    #[inline]
    fn host_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Pinned(buf) => buf.host_ptr as *mut u8,
            Self::Pageable(buf) => buf.as_mut_ptr(),
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
struct HostKvLayerPlan {
    k_page_stride: usize,
    v_page_stride: usize,
    raw_page_stride: usize,
    comp_page_stride: usize,
    slab_k_offset: usize,
    slab_v_offset: usize,
    slab_raw_offset: usize,
    slab_comp_offset: usize,
}

/// Lazy slab-allocated L2 Host-RAM KV page swap store paired with a rank's [`super::KvPool`].
#[cfg(feature = "cuda")]
pub struct HostKvSwapPool {
    pub device_id: i32,
    pub max_pages: usize,
    pub slab_pages: usize,
    pub page_bundle_bytes: usize,
    pub slab_bytes: usize,
    pub swap_out_pages: u64,
    pub swap_in_pages: u64,
    pub swap_out_batches: u64,
    pub swap_in_batches: u64,
    layer_plans: Vec<HostKvLayerPlan>,
    slabs: Vec<Option<HostKvSlab>>,
    free_pages: PageBitmap,
}

#[cfg(feature = "cuda")]
impl HostKvSwapPool {
    pub fn new(
        device_id: i32,
        layers: &[KvLayerBuffer],
        total_vram_pages: usize,
        max_pages: usize,
    ) -> Self {
        let slab_pages = max_pages.clamp(1, HOST_KV_SLAB_PAGES);
        let num_slabs = max_pages.div_ceil(slab_pages);
        let vram_pages = total_vram_pages.max(1);
        let mut layer_plans = Vec::with_capacity(layers.len());
        let mut slab_offset = 0usize;
        let mut page_bundle_bytes = 0usize;
        for layer in layers {
            let k_page_stride = layer.layout.k_page_stride_bytes as usize;
            let v_page_stride = layer.layout.v_page_stride_bytes as usize;
            let raw_page_stride = layer
                .qsa_raw_k_buf
                .as_ref()
                .map_or(0, |b| b.len_bytes / vram_pages);
            let comp_page_stride = layer
                .qsa_comp_k_buf
                .as_ref()
                .map_or(0, |b| b.len_bytes / vram_pages);
            let slab_k_offset = slab_offset;
            slab_offset += slab_pages * k_page_stride;
            let slab_v_offset = slab_offset;
            slab_offset += slab_pages * v_page_stride;
            let slab_raw_offset = slab_offset;
            slab_offset += slab_pages * raw_page_stride;
            let slab_comp_offset = slab_offset;
            slab_offset += slab_pages * comp_page_stride;
            page_bundle_bytes += k_page_stride + v_page_stride + raw_page_stride + comp_page_stride;
            layer_plans.push(HostKvLayerPlan {
                k_page_stride,
                v_page_stride,
                raw_page_stride,
                comp_page_stride,
                slab_k_offset,
                slab_v_offset,
                slab_raw_offset,
                slab_comp_offset,
            });
        }
        let mut slabs = Vec::with_capacity(num_slabs);
        slabs.resize_with(num_slabs, || None);
        Self {
            device_id,
            max_pages,
            slab_pages,
            page_bundle_bytes,
            slab_bytes: slab_offset,
            swap_out_pages: 0,
            swap_in_pages: 0,
            swap_out_batches: 0,
            swap_in_batches: 0,
            layer_plans,
            slabs,
            free_pages: PageBitmap::new(max_pages),
        }
    }

    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_pages.free_count()
    }

    #[inline]
    pub fn used_count(&self) -> usize {
        self.max_pages.saturating_sub(self.free_pages.free_count())
    }

    #[inline]
    pub fn allocated_slabs(&self) -> usize {
        self.slabs.iter().filter(|s| s.is_some()).count()
    }

    pub fn alloc_host_page(&mut self) -> Option<i32> {
        self.free_pages
            .alloc()
            .map(|idx| encode_host_kv_page(idx as usize))
    }

    pub fn free_host_page(&mut self, host_page_id: i32) {
        if let Some(idx) = decode_host_kv_page(host_page_id) {
            self.free_pages.free(idx as i32);
        }
    }

    pub fn reset_all_pages(&mut self) {
        self.free_pages.reset();
    }

    fn ensure_slab(&mut self, slab_idx: usize) -> Result<&mut HostKvSlab> {
        if slab_idx >= self.slabs.len() {
            return Err(CudaError::with_detail(
                -2,
                "HostKvSwapPool slab index out of range",
                format!("slab_idx={slab_idx} >= {}", self.slabs.len()),
            ));
        }
        if self.slabs[slab_idx].is_none() {
            self.slabs[slab_idx] = Some(HostKvSlab::alloc(self.device_id, self.slab_bytes)?);
        }
        Ok(self.slabs[slab_idx].as_mut().expect("slab initialized"))
    }

    /// Copies `vram_pages` (`0..num_pages`) from GPU VRAM into freshly allocated L2 Host-RAM
    /// pages on `stream`, returning the allocated tagged `host_page_id`s (`>= HOST_KV_PAGE_BASE`).
    pub fn swap_out_pages(
        &mut self,
        layers: &[KvLayerBuffer],
        total_vram_pages: usize,
        vram_pages: &[i32],
        stream: &CudaStream,
    ) -> Result<Vec<i32>> {
        if vram_pages.is_empty() {
            return Ok(Vec::new());
        }
        if self.free_count() < vram_pages.len() {
            return Err(CudaError::with_detail(
                -2,
                "HostKvSwapPool exhausted",
                format!(
                    "need {} host KV pages, only {} free",
                    vram_pages.len(),
                    self.free_count()
                ),
            ));
        }
        let mut host_pages = Vec::with_capacity(vram_pages.len());
        for &vram_pid in vram_pages {
            if vram_pid < 0 || (vram_pid as usize) >= total_vram_pages {
                for allocated in host_pages {
                    self.free_host_page(allocated);
                }
                return Err(CudaError::with_detail(
                    -2,
                    "invalid VRAM page ID in swap_out_pages",
                    format!("vram_pid={vram_pid} not in 0..{total_vram_pages}"),
                ));
            }
            let host_pid = self.alloc_host_page().expect("checked free_count");
            host_pages.push(host_pid);
        }

        let runs = coalesce_swap_runs(vram_pages, &host_pages, self.slab_pages);
        for run in &runs {
            if let Err(e) = self.ensure_slab(run.slab_idx) {
                for allocated in host_pages {
                    self.free_host_page(allocated);
                }
                return Err(e);
            }
        }

        CudaDevice {
            device_id: self.device_id,
        }
        .make_current()?;

        for (layer, plan) in layers.iter().zip(self.layer_plans.iter()) {
            for run in &runs {
                let slab_ptr = self.slabs[run.slab_idx]
                    .as_mut()
                    .expect("slab allocated")
                    .host_mut_ptr();
                if plan.k_page_stride > 0 {
                    let bytes = run.run_len * plan.k_page_stride;
                    let src = layer.k_buf.addr() + (run.vram_start * plan.k_page_stride) as u64;
                    let off = plan.slab_k_offset + run.slab_page_start * plan.k_page_stride;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `src` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_d2h_async(
                                slab_ptr.add(off) as *mut c_void,
                                src as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_out K D2H",
                    )?;
                }
                if plan.v_page_stride > 0 {
                    let bytes = run.run_len * plan.v_page_stride;
                    let src = layer.v_buf.addr() + (run.vram_start * plan.v_page_stride) as u64;
                    let off = plan.slab_v_offset + run.slab_page_start * plan.v_page_stride;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `src` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_d2h_async(
                                slab_ptr.add(off) as *mut c_void,
                                src as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_out V D2H",
                    )?;
                }
                if plan.raw_page_stride > 0
                    && let Some(raw_buf) = &layer.qsa_raw_k_buf
                {
                    let bytes = run.run_len * plan.raw_page_stride;
                    let src = raw_buf.addr() + (run.vram_start * plan.raw_page_stride) as u64;
                    let off = plan.slab_raw_offset + run.slab_page_start * plan.raw_page_stride;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `src` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_d2h_async(
                                slab_ptr.add(off) as *mut c_void,
                                src as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_out qsa_raw_k D2H",
                    )?;
                }
                if plan.comp_page_stride > 0
                    && let Some(comp_buf) = &layer.qsa_comp_k_buf
                {
                    let bytes = run.run_len * plan.comp_page_stride;
                    let src = comp_buf.addr() + (run.vram_start * plan.comp_page_stride) as u64;
                    let off = plan.slab_comp_offset + run.slab_page_start * plan.comp_page_stride;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `src` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_d2h_async(
                                slab_ptr.add(off) as *mut c_void,
                                src as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_out qsa_comp_k D2H",
                    )?;
                }
            }
        }

        self.swap_out_pages += vram_pages.len() as u64;
        self.swap_out_batches += 1;
        Ok(host_pages)
    }

    /// Copies `host_pages` (`>= HOST_KV_PAGE_BASE`) from L2 Host RAM back into `vram_pages`
    /// (`0..num_pages`) on `stream` and releases `host_pages` back to the free bitmap.
    pub fn swap_in_pages(
        &mut self,
        layers: &[KvLayerBuffer],
        total_vram_pages: usize,
        host_pages: &[i32],
        vram_pages: &[i32],
        free_host_after_copy: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        if host_pages.len() != vram_pages.len() {
            return Err(CudaError::with_detail(
                -2,
                "HostKvSwapPool swap_in_pages length mismatch",
                format!(
                    "host_pages.len()={} != vram_pages.len()={}",
                    host_pages.len(),
                    vram_pages.len()
                ),
            ));
        }
        if host_pages.is_empty() {
            return Ok(());
        }
        for (&host_pid, &vram_pid) in host_pages.iter().zip(vram_pages.iter()) {
            let Some(host_idx) = decode_host_kv_page(host_pid) else {
                return Err(CudaError::with_detail(
                    -2,
                    "invalid host page ID in swap_in_pages",
                    format!("host_pid={host_pid} < {HOST_KV_PAGE_BASE}"),
                ));
            };
            if host_idx >= self.max_pages || vram_pid < 0 || (vram_pid as usize) >= total_vram_pages
            {
                return Err(CudaError::with_detail(
                    -2,
                    "out-of-range page ID in swap_in_pages",
                    format!(
                        "host_idx={host_idx}/{}, vram_pid={vram_pid}/{total_vram_pages}",
                        self.max_pages
                    ),
                ));
            }
            let slab_idx = host_idx / self.slab_pages;
            if self.slabs.get(slab_idx).and_then(|s| s.as_ref()).is_none() {
                return Err(CudaError::with_detail(
                    -2,
                    "unallocated host slab in swap_in_pages",
                    format!("slab_idx={slab_idx} for host_pid={host_pid}"),
                ));
            }
        }

        let runs = coalesce_swap_runs(vram_pages, host_pages, self.slab_pages);
        CudaDevice {
            device_id: self.device_id,
        }
        .make_current()?;

        for (layer, plan) in layers.iter().zip(self.layer_plans.iter()) {
            for run in &runs {
                let slab_ptr = self.slabs[run.slab_idx]
                    .as_ref()
                    .expect("slab verified")
                    .host_ptr();
                if plan.k_page_stride > 0 {
                    let bytes = run.run_len * plan.k_page_stride;
                    let off = plan.slab_k_offset + run.slab_page_start * plan.k_page_stride;
                    let dst = layer.k_buf.addr() + (run.vram_start * plan.k_page_stride) as u64;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `dst` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_h2d_async(
                                dst as *mut c_void,
                                slab_ptr.add(off) as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_in K H2D",
                    )?;
                }
                if plan.v_page_stride > 0 {
                    let bytes = run.run_len * plan.v_page_stride;
                    let off = plan.slab_v_offset + run.slab_page_start * plan.v_page_stride;
                    let dst = layer.v_buf.addr() + (run.vram_start * plan.v_page_stride) as u64;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `dst` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_h2d_async(
                                dst as *mut c_void,
                                slab_ptr.add(off) as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_in V H2D",
                    )?;
                }
                if plan.raw_page_stride > 0
                    && let Some(raw_buf) = &layer.qsa_raw_k_buf
                {
                    let bytes = run.run_len * plan.raw_page_stride;
                    let off = plan.slab_raw_offset + run.slab_page_start * plan.raw_page_stride;
                    let dst = raw_buf.addr() + (run.vram_start * plan.raw_page_stride) as u64;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `dst` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_h2d_async(
                                dst as *mut c_void,
                                slab_ptr.add(off) as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_in qsa_raw_k H2D",
                    )?;
                }
                if plan.comp_page_stride > 0
                    && let Some(comp_buf) = &layer.qsa_comp_k_buf
                {
                    let bytes = run.run_len * plan.comp_page_stride;
                    let off = plan.slab_comp_offset + run.slab_page_start * plan.comp_page_stride;
                    let dst = comp_buf.addr() + (run.vram_start * plan.comp_page_stride) as u64;
                    check_rc(
                        // SAFETY: `slab_ptr` holds `slab_bytes` host memory; `dst` and `off`
                        // are bounds-checked within `total_vram_pages` and `slab_pages`.
                        unsafe {
                            ffi::warp34_memcpy_h2d_async(
                                dst as *mut c_void,
                                slab_ptr.add(off) as *const c_void,
                                bytes,
                                stream.handle,
                            )
                        },
                        "HostKvSwapPool swap_in qsa_comp_k H2D",
                    )?;
                }
            }
        }

        if free_host_after_copy {
            for &host_pid in host_pages {
                self.free_host_page(host_pid);
            }
        }
        self.swap_in_pages += host_pages.len() as u64;
        self.swap_in_batches += 1;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for HostKvSwapPool {
    fn drop(&mut self) {
        if self.slabs.iter().any(Option::is_some) {
            // SAFETY: binds calling thread to `self.device_id` and synchronizes the device
            // before dropping pinned/pageable host slabs.
            unsafe {
                let _ = ffi::warp34_set_device(self.device_id);
                let _ = ffi::warp34_device_synchronize();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_kv_page_tagging_and_run_coalescing() {
        assert!(!is_host_kv_page(0));
        assert!(!is_host_kv_page(16_383));
        assert_eq!(decode_host_kv_page(42), None);

        let h0 = encode_host_kv_page(0);
        let h255 = encode_host_kv_page(255);
        let h256 = encode_host_kv_page(256);
        assert!(is_host_kv_page(h0));
        assert_eq!(decode_host_kv_page(h0), Some(0));
        assert_eq!(decode_host_kv_page(h255), Some(255));
        assert_eq!(decode_host_kv_page(h256), Some(256));

        // 4 contiguous pages across a 256-page slab boundary (254, 255 in slab 0; 256, 257 in slab 1)
        // must split into 2 slab-local coalesced DMA runs of length 2 each.
        let vram = [10, 11, 12, 13];
        let host = [
            encode_host_kv_page(254),
            encode_host_kv_page(255),
            encode_host_kv_page(256),
            encode_host_kv_page(257),
        ];
        let runs = coalesce_swap_runs(&vram, &host, 256);
        assert_eq!(
            runs,
            vec![
                CoalescedSwapRun {
                    vram_start: 10,
                    slab_idx: 0,
                    slab_page_start: 254,
                    run_len: 2,
                },
                CoalescedSwapRun {
                    vram_start: 12,
                    slab_idx: 1,
                    slab_page_start: 0,
                    run_len: 2,
                },
            ]
        );
    }

    #[test]
    fn host_kv_swap_budget_planner_scales_across_tp_ranks() {
        let page_bytes_27b_tp2 = 171_008usize;
        let avail_64g = Some(64 * 1024 * 1024 * 1024usize);
        let pages_tp1 = plan_auto_host_kv_swap_pages(page_bytes_27b_tp2 * 2, 1, avail_64g);
        let pages_tp2 = plan_auto_host_kv_swap_pages(page_bytes_27b_tp2, 2, avail_64g);
        assert_eq!(pages_tp1, pages_tp2);
        assert_eq!(pages_tp2 % HOST_KV_SLAB_PAGES, 0);
        assert!(pages_tp2 >= 40_000, "got {pages_tp2}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn kv_pool_host_swap_roundtrips_multi_layer_pages_bit_exact() {
        use crate::cache::{KvFormatPolicy, KvPool};
        let stream = CudaStream::new(0).expect("stream");
        let mut pool = KvPool::new_target_full_attn(
            0,
            &[3, 7, 11, 15, 19],
            2,
            256,
            16,
            8,
            KvFormatPolicy::Tq4,
        )
        .expect("KvPool");
        pool.enable_host_swap(16);
        assert_eq!(pool.host_free_count(), 16);
        assert_eq!(pool.host_swap.as_ref().unwrap().allocated_slabs(), 0);

        let p0 = pool.alloc_page().unwrap();
        let p1 = pool.alloc_page().unwrap();
        let p2 = pool.alloc_page().unwrap();
        assert_eq!([p0, p1, p2], [0, 1, 2]);

        let mut expected_k = Vec::new();
        let mut expected_v = Vec::new();
        for (li, layer) in pool.layers.iter().enumerate() {
            let ks = layer.layout.k_page_stride_bytes as usize;
            let vs = layer.layout.v_page_stride_bytes as usize;
            let mut kh = vec![0u8; 8 * ks];
            let mut vh = vec![0u8; 8 * vs];
            for p in 0..3 {
                for b in 0..ks {
                    kh[p * ks + b] = ((li * 37 + p * 73 + b * 11 + 5) & 0xff) as u8;
                }
                for b in 0..vs {
                    vh[p * vs + b] = ((li * 53 + p * 97 + b * 17 + 9) & 0xff) as u8;
                }
            }
            layer.k_buf.copy_from_host(&kh, &stream).unwrap();
            layer.v_buf.copy_from_host(&vh, &stream).unwrap();
            expected_k.push(kh[..3 * ks].to_vec());
            expected_v.push(vh[..3 * vs].to_vec());
        }

        // Swap out pages [0, 1, 2] to L2 Host RAM.
        let host_pids = pool.swap_out_pages_to_host(&[0, 1, 2], &stream).unwrap();
        assert_eq!(host_pids.len(), 3);
        assert!(host_pids.iter().all(|&hp| is_host_kv_page(hp)));
        assert_eq!(pool.free_count(), 8);
        assert_eq!(pool.host_free_count(), 13);
        assert_eq!(pool.host_swap.as_ref().unwrap().allocated_slabs(), 1);

        // Zero all L1 VRAM pages so swap-in must restore from L2 Host RAM.
        for layer in &pool.layers {
            layer.k_buf.zero_async(&stream).unwrap();
            layer.v_buf.zero_async(&stream).unwrap();
        }

        // Allocate fresh L1 VRAM pages [4, 5, 6] and swap the 3 host pages back into them.
        for _ in 0..4 {
            let _ = pool.alloc_page().unwrap();
        }
        let dst_vram = [
            pool.alloc_page().unwrap(),
            pool.alloc_page().unwrap(),
            pool.alloc_page().unwrap(),
        ];
        assert_eq!(dst_vram, [4, 5, 6]);
        pool.swap_in_pages_from_host(&host_pids, &dst_vram, true, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(pool.host_free_count(), 16);

        for (li, layer) in pool.layers.iter().enumerate() {
            let ks = layer.layout.k_page_stride_bytes as usize;
            let vs = layer.layout.v_page_stride_bytes as usize;
            let mut kh = vec![0u8; 8 * ks];
            let mut vh = vec![0u8; 8 * vs];
            layer.k_buf.copy_to_host(&mut kh, &stream).unwrap();
            layer.v_buf.copy_to_host(&mut vh, &stream).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(&kh[4 * ks..7 * ks], expected_k[li].as_slice());
            assert_eq!(&vh[4 * vs..7 * vs], expected_v[li].as_slice());
        }
    }
}
