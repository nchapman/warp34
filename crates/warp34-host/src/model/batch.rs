//! Pool coordination, phased graph replay, pipelining, and batched stepping.

use super::*;

/// A Result error before launch is recoverable; a launched peer may already be
/// waiting in a collective. This handles Result failures, not unwinding panics.
#[cfg(feature = "cuda")]
fn finish_graph_replay(
    result: std::result::Result<(), crate::coordinated::ReplayError<CudaError>>,
) -> Result<()> {
    result.map_err(|error| {
        if error.partial_launch {
            // CUDA offers no safe cancellation of the peer's collective. Do not
            // unwind/free its buffers or retry; let the supervisor restart.
            eprintln!(
                "[warp34] fatal: asymmetric TP graph launch failure: {}",
                error.source
            );
            std::process::abort();
        }
        error.source
    })
}

#[cfg(feature = "cuda")]
impl Qwen3_5Engine {
    pub fn alloc_kv_page(&mut self) -> Option<i32> {
        let mut radix = std::mem::replace(&mut self.radix, MambaRadixCache::new(16, 64));
        let out = self.alloc_kv_page_from(&mut radix);
        self.radix = radix;
        out
    }

    /// Frees at least `needed_vram` L1 GPU VRAM pages by first non-destructively demoting
    /// unlocked `MambaRadixCache` KV pages to L2 Host RAM (`HostKvSwapPool`), evicting the
    /// coldest L2-backed leaves if `HostKvSwapPool` is full, and falling back to destructive
    /// L1 leaf eviction (`evict_full_pages`) only when L2 Host swap is disabled or exhausted.
    fn reclaim_vram_kv_pages_from(&mut self, radix: &mut MambaRadixCache, needed_vram: usize) {
        let avail = self.ranks[0].kv_pool.free_count();
        if avail >= needed_vram {
            return;
        }
        let deficit = needed_vram - avail;
        let max_host_pages = self
            .ranks
            .iter()
            .map(|r| r.kv_pool.host_swap.as_ref().map_or(0, |h| h.max_pages))
            .min()
            .unwrap_or(0);
        if max_host_pages > 0 {
            let desired_swap = deficit.max(64).min(max_host_pages);
            let mut targets = radix.collect_swappable_vram_pages(desired_swap);
            let mut host_free = self
                .ranks
                .iter()
                .map(|r| r.kv_pool.host_free_count())
                .min()
                .unwrap_or(0);
            if !targets.is_empty() && host_free < targets.len() {
                let ev_host = radix.evict_host_pages(targets.len() - host_free);
                if !ev_host.freed_kv_pages.is_empty() || !ev_host.freed_mamba_slots.is_empty() {
                    self.free_kv_pages(&ev_host.freed_kv_pages);
                    self.free_state_slots(&ev_host.freed_mamba_slots);
                }
                host_free = self
                    .ranks
                    .iter()
                    .map(|r| r.kv_pool.host_free_count())
                    .min()
                    .unwrap_or(0);
                targets = radix.collect_swappable_vram_pages(desired_swap.min(host_free));
            }
            if !targets.is_empty() && host_free >= targets.len() {
                let vram_pages: Vec<i32> = targets.iter().map(|&(_, _, pid)| pid).collect();
                let mut per_rank_host: Vec<Vec<i32>> = Vec::with_capacity(self.ranks.len());
                let mut ok = true;
                for r in &mut self.ranks {
                    let res = r
                        .kv_pool
                        .host_swap
                        .as_mut()
                        .expect("max_host_pages > 0")
                        .swap_out_pages(
                            &r.kv_pool.layers,
                            r.kv_pool.num_pages,
                            &vram_pages,
                            &r.stream,
                        );
                    match res {
                        Ok(hp) => per_rank_host.push(hp),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && per_rank_host.windows(2).all(|w| w[0] == w[1]) {
                    for r in &mut self.ranks {
                        for &vp in &vram_pages {
                            r.kv_pool.free_page(vp);
                        }
                    }
                    radix.apply_swapped_out_pages(&targets, &per_rank_host[0]);
                } else {
                    // Roll back allocated L2 host slots on any rank that succeeded; L1 VRAM
                    // pages were only read (not freed yet), so TP=2 PageBitmaps stay in lockstep.
                    for (r, hp) in self.ranks.iter_mut().zip(per_rank_host) {
                        for h_pid in hp {
                            r.kv_pool.free_page(h_pid);
                        }
                    }
                }
            }
        }
        let avail_after_swap = self.ranks[0].kv_pool.free_count();
        if avail_after_swap < needed_vram {
            let ev = radix.evict_full_pages(needed_vram - avail_after_swap);
            if !ev.freed_kv_pages.is_empty() || !ev.freed_mamba_slots.is_empty() {
                self.free_kv_pages(&ev.freed_kv_pages);
                self.free_state_slots(&ev.freed_mamba_slots);
            }
        }
    }

    pub fn alloc_kv_page_from(&mut self, radix: &mut MambaRadixCache) -> Option<i32> {
        self.reclaim_vram_kv_pages_from(radix, 1);
        if let Some(p0) = self.ranks[0].kv_pool.alloc_page() {
            for r in &mut self.ranks[1..] {
                let p_peer = r.kv_pool.alloc_page();
                debug_assert_eq!(Some(p0), p_peer);
            }
            return Some(p0);
        }
        None
    }

    /// Promotes any L2 Host-RAM KV pages in `dst` back to L1 GPU VRAM and allocates up to
    /// `target_pages` total L1 VRAM pages, non-destructively demoting unlocked `radix` pages
    /// to L2 Host RAM before falling back to leaf eviction.
    pub fn reserve_kv_pages_from(
        &mut self,
        radix: &mut MambaRadixCache,
        dst: &mut Vec<i32>,
        target_pages: usize,
    ) -> bool {
        for pid in dst.iter_mut() {
            if crate::cache::is_host_kv_page(*pid)
                && let Some(already_vram) = radix.get_promoted_vram_page(*pid)
            {
                *pid = already_vram;
            }
        }
        let host_positions: Vec<(usize, i32)> = dst
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, pid)| crate::cache::is_host_kv_page(pid))
            .collect();
        let missing = target_pages.saturating_sub(dst.len());
        let total_vram_needed = host_positions.len() + missing;
        if total_vram_needed == 0 {
            return true;
        }
        self.reclaim_vram_kv_pages_from(radix, total_vram_needed);
        if self.ranks[0].kv_pool.free_count() < total_vram_needed {
            return false;
        }
        if !host_positions.is_empty() {
            let mut promoted_vram = Vec::with_capacity(host_positions.len());
            for _ in 0..host_positions.len() {
                let p0 = self.ranks[0]
                    .kv_pool
                    .alloc_page()
                    .expect("free_count verified before host page promotion");
                for r in &mut self.ranks[1..] {
                    let p_peer = r.kv_pool.alloc_page();
                    debug_assert_eq!(Some(p0), p_peer);
                }
                promoted_vram.push(p0);
            }
            let host_pages: Vec<i32> = host_positions.iter().map(|&(_, h)| h).collect();
            let mut swap_in_ok = true;
            for r in &mut self.ranks {
                if r.kv_pool
                    .swap_in_pages_from_host(&host_pages, &promoted_vram, false, &r.stream)
                    .is_err()
                {
                    swap_in_ok = false;
                    break;
                }
            }
            if !swap_in_ok {
                for r in &mut self.ranks {
                    for &vp in &promoted_vram {
                        r.kv_pool.free_page(vp);
                    }
                }
                return false;
            }
            for r in &mut self.ranks {
                for &hp in &host_pages {
                    r.kv_pool.free_page(hp);
                }
            }
            for (&(idx, _), &vram_pid) in host_positions.iter().zip(promoted_vram.iter()) {
                dst[idx] = vram_pid;
            }
            radix.record_host_page_promotions(&host_pages, &promoted_vram);
        }
        dst.reserve(missing);
        for _ in 0..missing {
            let p0 = self.ranks[0]
                .kv_pool
                .alloc_page()
                .expect("free_count verified before page allocation");
            for r in &mut self.ranks[1..] {
                let p_peer = r.kv_pool.alloc_page();
                debug_assert_eq!(Some(p0), p_peer);
            }
            dst.push(p0);
        }
        true
    }

    pub fn free_kv_pages(&mut self, pages: &[i32]) {
        for r in &mut self.ranks {
            for &p in pages {
                if p >= 0 {
                    r.kv_pool.free_page(p);
                }
            }
        }
    }

    pub fn alloc_state_slot(&mut self) -> Option<i32> {
        let mut radix = std::mem::replace(&mut self.radix, MambaRadixCache::new(16, 64));
        let out = self.alloc_state_slot_from(&mut radix);
        self.radix = radix;
        out
    }

    #[inline]
    pub fn has_checkpoint_pool(&self) -> bool {
        self.ranks
            .first()
            .and_then(|r| r.checkpoint_pool.as_ref())
            .is_some_and(|p| p.total_slots > 0)
    }

    fn try_alloc_lockstep_checkpoint_slot(&mut self) -> Option<i32> {
        let c0 = self
            .ranks
            .first_mut()?
            .checkpoint_pool
            .as_mut()?
            .alloc_slot()?;
        let mut peer_slots = Vec::with_capacity(self.ranks.len().saturating_sub(1));
        for r in &mut self.ranks[1..] {
            match r.checkpoint_pool.as_mut().and_then(|c| c.alloc_slot()) {
                Some(cp) if cp == c0 => peer_slots.push(cp),
                other => {
                    if let Some(cp) = other
                        && let Some(ckpt) = &mut r.checkpoint_pool
                    {
                        ckpt.free_slot(cp);
                    }
                    if let Some(ckpt0) = &mut self.ranks[0].checkpoint_pool {
                        ckpt0.free_slot(c0);
                    }
                    for (idx, cp) in peer_slots.into_iter().enumerate() {
                        if let Some(ckpt) = &mut self.ranks[idx + 1].checkpoint_pool {
                            ckpt.free_slot(cp);
                        }
                    }
                    return None;
                }
            }
        }
        Some(c0)
    }

    pub fn alloc_checkpoint_slot_from(&mut self, radix: &mut MambaRadixCache) -> Option<i32> {
        if !self.has_checkpoint_pool() {
            return None;
        }
        if let Some(c0) = self.try_alloc_lockstep_checkpoint_slot() {
            return Some(c0);
        }
        let ev = radix.evict_mamba(1);
        if !ev.freed_kv_pages.is_empty() || !ev.freed_mamba_slots.is_empty() {
            self.free_kv_pages(&ev.freed_kv_pages);
            self.free_state_slots(&ev.freed_mamba_slots);
            if let Some(c0) = self.try_alloc_lockstep_checkpoint_slot() {
                return Some(c0);
            }
        }
        None
    }

    pub fn alloc_state_slot_from(&mut self, radix: &mut MambaRadixCache) -> Option<i32> {
        if let Some(s0) = self.ranks[0].state_pool.alloc_slot() {
            for r in &mut self.ranks[1..] {
                let s_peer = r.state_pool.alloc_slot();
                debug_assert_eq!(Some(s0), s_peer);
            }
            return Some(s0);
        }
        if self.has_checkpoint_pool() {
            // Checkpoints in `radix` live in `checkpoint_pool`, not `state_pool`.
            return None;
        }
        let ev = radix.evict_mamba(1);
        if !ev.freed_kv_pages.is_empty() || !ev.freed_mamba_slots.is_empty() {
            self.free_kv_pages(&ev.freed_kv_pages);
            self.free_state_slots(&ev.freed_mamba_slots);
            if let Some(s0) = self.ranks[0].state_pool.alloc_slot() {
                for r in &mut self.ranks[1..] {
                    let s_peer = r.state_pool.alloc_slot();
                    debug_assert_eq!(Some(s0), s_peer);
                }
                return Some(s0);
            }
        }
        None
    }

    /// Allocates a GDN `StateBundle` slot from the free pool or by reclaiming a superseded
    /// internal Mamba snapshot / unverified decode tail whose parent prompt frontier remains
    /// intact in `radix`, without ever deleting a waiting session's prompt frontier.
    pub fn alloc_state_slot_preserving_leaves(
        &mut self,
        radix: &mut MambaRadixCache,
    ) -> Option<i32> {
        if let Some(s0) = self.ranks[0].state_pool.alloc_slot() {
            for r in &mut self.ranks[1..] {
                let s_peer = r.state_pool.alloc_slot();
                debug_assert_eq!(Some(s0), s_peer);
            }
            return Some(s0);
        }
        if !self.has_checkpoint_pool() && radix.has_evictable_internal_mamba() {
            return self.alloc_state_slot_from(radix);
        }
        None
    }

    pub fn free_state_slots(&mut self, slots: &[i32]) {
        for r in &mut self.ranks {
            for &s in slots {
                if crate::cache::is_checkpoint_pool_slot(s) {
                    if let Some(ckpt) = &mut r.checkpoint_pool {
                        ckpt.free_slot(s);
                    }
                } else if s >= 0 {
                    r.state_pool.free_slot(s);
                }
            }
        }
    }

    pub fn pack_checkpoint_from_active(
        &mut self,
        src_active_slot: i32,
        dst_ckpt_slot: i32,
    ) -> Result<()> {
        for r in &mut self.ranks {
            let ckpt = r
                .checkpoint_pool
                .as_mut()
                .ok_or_else(|| loader_err("GdnCheckpointPool is not enabled on rank"))?;
            ckpt.pack_from_active(&r.state_pool, src_active_slot, dst_ckpt_slot, &r.stream)?;
        }
        Ok(())
    }

    pub fn unpack_checkpoint_to_active(
        &mut self,
        src_ckpt_slot: i32,
        dst_active_slot: i32,
    ) -> Result<()> {
        for r in &mut self.ranks {
            let ckpt = r
                .checkpoint_pool
                .as_mut()
                .ok_or_else(|| loader_err("GdnCheckpointPool is not enabled on rank"))?;
            ckpt.unpack_to_active(&mut r.state_pool, src_ckpt_slot, dst_active_slot, &r.stream)?;
        }
        Ok(())
    }

    #[inline]
    pub fn materialize_pending_checkpoint(
        &mut self,
        req_state: &mut crate::radix::ReqRadixState,
    ) -> Result<()> {
        if let Some(ckpt_slot) = req_state.pending_unpack_ckpt_slot.take() {
            self.unpack_checkpoint_to_active(ckpt_slot, req_state.active_state_slot)?;
        }
        Ok(())
    }

    pub fn apply_radix_donation(
        &mut self,
        donation: &crate::radix::RadixDonationOutcome,
    ) -> Result<()> {
        let pack_res = if let Some((src_active, dst_ckpt)) = donation.pending_checkpoint_pack {
            let res = self.pack_checkpoint_from_active(src_active, dst_ckpt);
            if res.is_err() {
                self.free_state_slots(&[dst_ckpt]);
            }
            res
        } else {
            Ok(())
        };
        self.free_kv_pages(&donation.freed_kv_pages);
        self.free_state_slots(&donation.freed_mamba_slots);
        pack_res
    }

    pub fn apply_radix_donation_with_radix(
        &mut self,
        radix: &mut MambaRadixCache,
        donation: &crate::radix::RadixDonationOutcome,
    ) -> Result<()> {
        let res = self.apply_radix_donation(donation);
        if res.is_err()
            && let Some((_, dst_ckpt)) = donation.pending_checkpoint_pack
        {
            let inv = radix.invalidate_mamba_slot(dst_ckpt);
            self.free_kv_pages(&inv.freed_kv_pages);
            self.free_state_slots(&inv.freed_mamba_slots);
        }
        res
    }

    /// Publishes a newly committed radix checkpoint (`node_id`) into the shared cross-replica
    /// L2 Host-RAM prefix pool if not already present. Best-effort: skips silently on empty
    /// pool, short prefixes, non-VRAM pages, or missing checkpoint pool on hybrid GDN models.
    pub fn try_publish_radix_node_to_shared_pool(
        &self,
        radix: &MambaRadixCache,
        node_id: crate::radix::RadixNodeId,
        lora_id: i32,
        raw_radix_tokens: &[i32],
    ) {
        self.try_publish_radix_node_to_shared_pool_with_features(
            radix,
            node_id,
            lora_id,
            raw_radix_tokens,
            false,
            || None,
        );
    }

    /// Publishes a newly committed radix checkpoint (`node_id`) along with optional speculative
    /// proposer features (`MTP` / `DFlash2`) into the shared cross-replica L2 Host-RAM prefix pool.
    /// `make_features` is invoked lazily only after fast-path deduplication, stride throttling,
    /// and base-size checks pass.
    pub fn try_publish_radix_node_to_shared_pool_with_features<F>(
        &self,
        radix: &MambaRadixCache,
        node_id: crate::radix::RadixNodeId,
        lora_id: i32,
        raw_radix_tokens: &[i32],
        has_features_hint: bool,
        make_features: F,
    ) where
        F: FnOnce() -> Option<Vec<u8>>,
    {
        let Some(pool) = &self.dp_prefix_pool else {
            return;
        };
        if raw_radix_tokens.len() < crate::dp_prefix_pool::DP_L2_MIN_PREFIX_TOKENS {
            return;
        }
        let is_hybrid_gdn = self.cfg.num_hidden_layers > self.cfg.num_full_attention_layers();
        if is_hybrid_gdn && !self.has_checkpoint_pool() {
            return;
        }
        let Some((num_tokens, kv_pages, mamba_slot)) = radix.node_kv_pages_and_mamba(node_id)
        else {
            return;
        };
        if num_tokens < crate::dp_prefix_pool::DP_L2_MIN_PREFIX_TOKENS
            || num_tokens > raw_radix_tokens.len()
        {
            return;
        }
        let prefix_tokens = &raw_radix_tokens[..num_tokens];
        if pool.contains_and_touch_with_features(lora_id, prefix_tokens, has_features_hint) {
            return;
        }
        // Throttle redundant synchronous D2H publishes on every short turn of a long pinned
        // agent trajectory: scale the refresh stride proportionally past 4K (4K at 8K–16K,
        // 8K at 32K, 14K at 56K) so deep sessions halve D2H churn while keeping spillover fresh.
        let refresh_stride = (num_tokens / 4).max(4096);
        if num_tokens > 4096
            && pool.has_recent_prefix_within(lora_id, prefix_tokens, refresh_stride)
        {
            return;
        }
        // Guard against allocating oversized host buffers that `pool.insert` would immediately
        // reject (> max_bytes / 2).
        let est_kv_bytes: usize = self
            .ranks
            .iter()
            .map(|r| {
                r.kv_pool
                    .layers
                    .iter()
                    .map(|l| {
                        let k_s = l.k_buf.len_bytes / r.kv_pool.num_pages;
                        let v_s = l.v_buf.len_bytes / r.kv_pool.num_pages;
                        let raw_s = l
                            .qsa_raw_k_buf
                            .as_ref()
                            .map_or(0, |b| b.len_bytes / r.kv_pool.num_pages);
                        let comp_s = l
                            .qsa_comp_k_buf
                            .as_ref()
                            .map_or(0, |b| b.len_bytes / r.kv_pool.num_pages);
                        (k_s + v_s + raw_s + comp_s) * kv_pages.len()
                    })
                    .sum::<usize>()
            })
            .sum();
        let est_ckpt_bytes: usize = if is_hybrid_gdn {
            self.ranks
                .iter()
                .filter_map(|r| r.checkpoint_pool.as_ref().map(|c| c.layout.bundle_bytes))
                .sum()
        } else {
            0
        };
        let est_base = est_kv_bytes
            .saturating_add(est_ckpt_bytes)
            .saturating_add(std::mem::size_of_val(prefix_tokens));
        if est_base == 0 || est_base > pool.max_bytes() / 2 {
            return;
        }
        let proposer_features = make_features();
        let est_feat_bytes = proposer_features.as_ref().map_or(0, Vec::len);
        let est_total = est_base.saturating_add(est_feat_bytes);
        if est_total > pool.max_bytes() / 2 {
            return;
        }
        let uses_ple = self.cfg.uses_ple();
        let has_ckpt_ranks = is_hybrid_gdn || uses_ple;
        let kv_pages_ref = &kv_pages;
        let exported = crate::coordinated::run_tp_closures(
            self.tp_workers.as_deref(),
            self.ranks.iter().map(|r| {
                move || {
                    let kv_bytes = r
                        .kv_pool
                        .export_pages_to_host_bytes(kv_pages_ref, &r.stream)
                        .ok()?;
                    let ckpt_entry = if is_hybrid_gdn {
                        let ckpt = r.checkpoint_pool.as_ref()?;
                        let (frame, ple_ngram_context) =
                            ckpt.export_slot_to_host_bytes(mamba_slot, &r.stream).ok()?;
                        Some(crate::dp_prefix_pool::SharedGdnCheckpoint {
                            frame,
                            ple_ngram_context,
                        })
                    } else if uses_ple {
                        let idx = usize::try_from(mamba_slot).ok()?;
                        let &ple_ngram_context = r.state_pool.ple_ngram_context.get(idx)?;
                        Some(crate::dp_prefix_pool::SharedGdnCheckpoint {
                            frame: Vec::new(),
                            ple_ngram_context,
                        })
                    } else {
                        None
                    };
                    Some((kv_bytes, ckpt_entry))
                }
            }),
        );
        let mut kv_ranks = Vec::with_capacity(self.ranks.len());
        let mut ckpt_ranks = Vec::with_capacity(if has_ckpt_ranks { self.ranks.len() } else { 0 });
        for item in exported {
            let Some((kv_bytes, ckpt_opt)) = item else {
                return;
            };
            kv_ranks.push(kv_bytes);
            if let Some(c) = ckpt_opt {
                ckpt_ranks.push(c);
            }
        }
        let entry = crate::dp_prefix_pool::SharedPrefixEntry::new(
            lora_id,
            prefix_tokens.to_vec(),
            kv_pages.len(),
            kv_ranks,
            ckpt_ranks,
            proposer_features,
            self.dp_rank,
        );
        let _ = pool.insert(entry);
    }

    /// Attempts to hydrate a longer matching prefix for a waiting request from the shared
    /// cross-replica L2 Host-RAM prefix pool into this replica's L1 VRAM + `MambaRadixCache`.
    /// Returns the number of tokens hydrated (`0` on local hit, miss, or insufficient VRAM).
    pub fn try_hydrate_waiting_from_shared_pool(
        &mut self,
        radix: &mut MambaRadixCache,
        lora_id: i32,
        raw_radix_tokens: &[i32],
    ) -> Option<std::sync::Arc<crate::dp_prefix_pool::SharedPrefixEntry>> {
        let pool = self.dp_prefix_pool.clone()?;
        if pool.is_empty()
            || raw_radix_tokens.len() < crate::dp_prefix_pool::DP_L2_MIN_PREFIX_TOKENS
        {
            return None;
        }
        let is_hybrid_gdn = self.cfg.num_hidden_layers > self.cfg.num_full_attention_layers();
        if is_hybrid_gdn && !self.has_checkpoint_pool() {
            return None;
        }
        let prev_lora = radix.active_lora_id;
        radix.active_lora_id = lora_id;
        let local_matched = radix.peek_matched_tokens(raw_radix_tokens);
        radix.active_lora_id = prev_lora;

        // Require a meaningful net gain (>= 1024 tokens) before hydrating over a warm local
        // prefix so we never reclaim/copy a full 48K-token entry just to gain +16 tokens.
        let min_tokens = if local_matched > 0 {
            local_matched.saturating_add(1024)
        } else {
            0
        };
        let uses_ple = self.cfg.uses_ple();
        let has_ckpt_ranks = is_hybrid_gdn || uses_ple;
        let entry = pool.lookup_longest_match(lora_id, raw_radix_tokens, min_tokens)?;
        if entry.kv_ranks.len() != self.ranks.len()
            || entry.ckpt_ranks.len() != if has_ckpt_ranks { self.ranks.len() } else { 0 }
        {
            return None;
        }
        let need_pages = entry.num_pages;
        // Ensure enough free VRAM KV pages exist across every TP rank before allocating so we
        // never starve active sequences or desynchronize TP peer PageBitmaps.
        self.reclaim_vram_kv_pages_from(radix, need_pages.saturating_add(1));
        if self
            .ranks
            .iter()
            .any(|r| r.kv_pool.free_count() < need_pages)
        {
            return None;
        }
        let ckpt_slot = if is_hybrid_gdn {
            self.alloc_checkpoint_slot_from(radix)?
        } else {
            self.alloc_state_slot_from(radix)?
        };
        let mut vram_pages = Vec::with_capacity(need_pages);
        for _ in 0..need_pages {
            let mut allocated_this_step: Vec<(usize, i32)> = Vec::with_capacity(self.ranks.len());
            let Some(p0) = self.ranks[0].kv_pool.alloc_page() else {
                self.free_kv_pages(&vram_pages);
                self.free_state_slots(&[ckpt_slot]);
                return None;
            };
            allocated_this_step.push((0, p0));
            let mut peer_ok = true;
            for (rank_idx, r) in self.ranks.iter_mut().enumerate().skip(1) {
                match r.kv_pool.alloc_page() {
                    Some(p_peer) if p_peer == p0 => allocated_this_step.push((rank_idx, p_peer)),
                    Some(p_peer) => {
                        allocated_this_step.push((rank_idx, p_peer));
                        peer_ok = false;
                        break;
                    }
                    None => {
                        peer_ok = false;
                        break;
                    }
                }
            }
            if !peer_ok {
                for (rank_idx, pid) in allocated_this_step {
                    self.ranks[rank_idx].kv_pool.free_page(pid);
                }
                self.free_kv_pages(&vram_pages);
                self.free_state_slots(&[ckpt_slot]);
                return None;
            }
            vram_pages.push(p0);
        }

        let vram_pages_ref = &vram_pages;
        let entry_ref = entry.as_ref();
        let pool_workers = self.tp_workers.clone();
        let import_ok = crate::coordinated::run_tp_closures(
            pool_workers.as_deref(),
            self.ranks.iter_mut().enumerate().map(|(rank_idx, r)| {
                move || -> bool {
                    if r.kv_pool
                        .import_pages_from_host_bytes(
                            vram_pages_ref,
                            &entry_ref.kv_ranks[rank_idx],
                            &r.stream,
                        )
                        .is_err()
                    {
                        return false;
                    }
                    if is_hybrid_gdn {
                        let Some(ckpt) = r.checkpoint_pool.as_mut() else {
                            return false;
                        };
                        let c = &entry_ref.ckpt_ranks[rank_idx];
                        if ckpt
                            .import_host_bytes_to_slot(
                                ckpt_slot,
                                &c.frame,
                                c.ple_ngram_context,
                                &r.stream,
                            )
                            .is_err()
                        {
                            return false;
                        }
                    } else if uses_ple {
                        let Ok(idx) = usize::try_from(ckpt_slot) else {
                            return false;
                        };
                        let Some(dst) = r.state_pool.ple_ngram_context.get_mut(idx) else {
                            return false;
                        };
                        *dst = entry_ref.ckpt_ranks[rank_idx].ple_ngram_context;
                    }
                    true
                }
            }),
        )
        .into_iter()
        .all(|ok| ok);
        if !import_ok {
            self.free_kv_pages(&vram_pages);
            self.free_state_slots(&[ckpt_slot]);
            return None;
        }

        let prev_lora = radix.active_lora_id;
        radix.active_lora_id = lora_id;
        let ins = radix.insert(&entry.token_ids, &vram_pages, &[], ckpt_slot, 0);
        radix.active_lora_id = prev_lora;
        self.free_kv_pages(&ins.freed_duplicate_kv_pages);
        if ins.mamba_already_existed {
            self.free_state_slots(&[ckpt_slot]);
        }
        pool.record_hydration(entry.token_ids.len());
        Some(entry)
    }

    pub fn donate_req_checkpoint(
        &mut self,
        radix: &mut MambaRadixCache,
        req_state: &mut crate::radix::ReqRadixState,
        finish_req: bool,
    ) -> Result<crate::radix::RadixDonationOutcome> {
        if self.has_checkpoint_pool() {
            let can_recycle_ckpt =
                radix.can_recycle_locked_checkpoint_mamba(req_state.last_locked_node);
            let recycle_in_place = !finish_req && req_state.can_recycle_prefill_checkpoint(radix);
            let ckpt = if req_state.has_donatable_checkpoint() && !recycle_in_place {
                self.alloc_checkpoint_slot_from(radix)
            } else {
                None
            };
            let outcome =
                if ckpt.is_none() && !can_recycle_ckpt && req_state.has_donatable_checkpoint() {
                    if finish_req {
                        // All checkpoint-pool slots are locked by other requests; release active
                        // FP32 state back to `state_pool` rather than leaking it into `radix`.
                        req_state.abort_and_release(radix)
                    } else {
                        crate::radix::RadixDonationOutcome::default()
                    }
                } else {
                    req_state.donate_committed_checkpoint(radix, ckpt, finish_req)
                };
            self.apply_radix_donation_with_radix(radix, &outcome)?;
            Ok(outcome)
        } else {
            let can_recycle = radix.can_recycle_locked_mamba(req_state.last_locked_node);
            let rep = if finish_req || !req_state.has_donatable_checkpoint() {
                None
            } else {
                self.alloc_state_slot_preserving_leaves(radix).or_else(|| {
                    (!can_recycle)
                        .then(|| self.alloc_state_slot_from(radix))
                        .flatten()
                })
            };
            let outcome = if finish_req || rep.is_some() || can_recycle {
                req_state.donate_committed_checkpoint(radix, rep, finish_req)
            } else {
                crate::radix::RadixDonationOutcome::default()
            };
            self.apply_radix_donation_with_radix(radix, &outcome)?;
            Ok(outcome)
        }
    }

    #[inline]
    pub(crate) fn batch_lora_mode(&self, num_reqs: usize) -> GraphLoraMode {
        if num_reqs == 0 || self.lora_registry.is_empty() {
            return GraphLoraMode::BaseOnly;
        }
        let first = self.slice_lora_id(0);
        if (1..num_reqs).all(|i| self.slice_lora_id(i) == first) {
            GraphLoraMode::from_single(first)
        } else {
            GraphLoraMode::Mixed
        }
    }

    fn step_spec_verify_graph(
        &mut self,
        s: &EngineBatchSlice<'_>,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let q_len = s.tokens.len();
        let long_ctx = !self.cfg.uses_qsa() && s.ctx_len_before + q_len > 1024;
        let sample_mode = if skip_host_greedy {
            DecodeGraphSampleMode::LogitsOnly
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };
        let lora_id = self.slice_lora_id(0);
        let lora_mode = self.batch_lora_mode(1);
        let host_meta = SpecVerifyGraphIo::pack_meta_with_lora(s, lora_id)?;
        use crate::coordinated::{ReplayPhase, replay_ranks};
        let comm = self.tp2_comm.as_ref().cloned();
        if let Some(comm) = &comm {
            let stage_bytes = q_len
                .checked_mul(self.cfg.hidden_size)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| loader_err("tp2 staging payload overflow"))?;
            if stage_bytes > comm.max_stage_bytes_per_epoch {
                return Err(loader_err(
                    "tp2 batch staging bytes exceed max_stage_bytes_per_epoch",
                ));
            }
        }
        // Capture records collectives without executing them. Both cached and
        // first-use graphs finish every rank's uploads/capture before any launch.
        let result = replay_ranks(self.ranks.len(), |phase, rank| {
            let r = &mut self.ranks[rank];
            r.device.make_current()?;
            match phase {
                ReplayPhase::Upload => {
                    if r.ple_runtime.is_some() {
                        let mut ckpt_dst = Vec::with_capacity(s.checkpoints.len());
                        let mut ckpt_off = Vec::with_capacity(s.checkpoints.len());
                        for &(trig_off, dst_slot) in s.checkpoints {
                            ckpt_off.push(trig_off);
                            ckpt_dst.push(dst_slot);
                        }
                        r.stage_ple_f8_for_batch(
                            s.tokens,
                            &[0, q_len as i32],
                            &[s.state_in_slot],
                            &[s.state_out_slot],
                            &[u8::from(s.has_initial_state)],
                            s.spec_slots,
                            s.spec_slots.len(),
                            &ckpt_dst,
                            &ckpt_off,
                            s.checkpoints.len(),
                        )?;
                    }
                    r.spec_verify_io
                        .meta_i32
                        .copy_from_host(&host_meta, &r.stream)?;
                    r.spec_verify_io
                        .block_tables
                        .copy_from_host(s.block_table, &r.stream)
                }
                ReplayPhase::Capture => r.ensure_spec_verify_graph(
                    q_len,
                    long_ctx,
                    sample_mode,
                    lora_mode,
                    comm.as_deref(),
                ),
                ReplayPhase::Launch => {
                    r.bind_graph_lora_mode(lora_mode, r.spec_verify_io.token_lora_ids_ptr());
                    let (captured_logits_ptr, graph) = r
                        .find_spec_verify_graph(q_len, long_ctx, sample_mode, lora_mode)
                        .ok_or_else(|| loader_err("missing captured spec_verify graph"))?;
                    graph.launch(&r.stream)?;
                    r.mark_tiered_moe_device_lru_dirty();
                    r.speculative_logits_rows = if r.has_spec_logits_capacity(q_len) {
                        q_len
                    } else {
                        0
                    };
                    r.last_logits_ptr = captured_logits_ptr;
                    r.last_logits_rows = q_len;
                    Ok(())
                }
            }
        });
        finish_graph_replay(result)?;
        if skip_host_greedy {
            return Ok(vec![Vec::new()]);
        }
        let mut out_ids = vec![0i32; q_len];
        self.ranks[0].device.make_current()?;
        self.ranks[0]
            .spec_verify_io
            .out_greedy_ids
            .copy_to_host(&mut out_ids, &self.ranks[0].stream)?;
        for r in &self.ranks[1..] {
            r.device.make_current()?;
            r.stream.synchronize()?;
        }
        Ok(vec![out_ids])
    }

    fn step_batched_spec_verify_graph(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        total_tokens: usize,
        max_q_len: usize,
        long_ctx: bool,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let num_reqs = slices.len();
        let sample_mode = if skip_host_greedy {
            DecodeGraphSampleMode::LogitsOnly
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };
        let lora_mode = self.batch_lora_mode(num_reqs);
        let slice_lids =
            (!self.slice_lora_ids.is_empty()).then_some(self.slice_lora_ids.as_slice());
        let host_meta = BatchedSpecVerifyGraphIo::pack_meta_with_lora(
            slices,
            max_q_len,
            slice_lids,
            self.active_lora_id,
        )?;
        use crate::coordinated::{ReplayPhase, replay_ranks};
        let comm = self.tp2_comm.as_ref().cloned();
        if let Some(comm) = &comm {
            let stage_bytes = total_tokens
                .checked_mul(self.cfg.hidden_size)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| loader_err("tp2 staging payload overflow"))?;
            if stage_bytes > comm.max_stage_bytes_per_epoch {
                return Err(loader_err(
                    "tp2 batch staging bytes exceed max_stage_bytes_per_epoch",
                ));
            }
        }
        let meta_bytes = core::mem::size_of_val(&host_meta);
        let bt_off = (meta_bytes + 63) & !63;
        let result = replay_ranks(self.ranks.len(), |phase, rank| {
            let r = &mut self.ranks[rank];
            r.device.make_current()?;
            match phase {
                ReplayPhase::Upload => {
                    if r.ple_runtime.is_some() {
                        let mut flat_tokens = Vec::with_capacity(total_tokens);
                        let mut q_start = Vec::with_capacity(num_reqs + 1);
                        q_start.push(0i32);
                        let mut st_in = Vec::with_capacity(num_reqs);
                        let mut st_out = Vec::with_capacity(num_reqs);
                        let mut has_init = Vec::with_capacity(num_reqs);
                        let mut spec_slots = vec![-1i32; num_reqs * max_q_len];
                        let mut ckpt_dst = vec![-1i32; num_reqs * MAX_SPEC_VERIFY_CHECKPOINTS];
                        let mut ckpt_off = vec![-1i32; num_reqs * MAX_SPEC_VERIFY_CHECKPOINTS];
                        for (idx, s) in slices.iter().enumerate() {
                            flat_tokens.extend_from_slice(s.tokens);
                            q_start.push(flat_tokens.len() as i32);
                            st_in.push(s.state_in_slot);
                            st_out.push(s.state_out_slot);
                            has_init.push(u8::from(s.has_initial_state));
                            spec_slots[idx * max_q_len..idx * max_q_len + s.spec_slots.len()]
                                .copy_from_slice(s.spec_slots);
                            for (e, &(trig_off, dst_slot)) in s.checkpoints.iter().enumerate() {
                                ckpt_off[idx * MAX_SPEC_VERIFY_CHECKPOINTS + e] = trig_off;
                                ckpt_dst[idx * MAX_SPEC_VERIFY_CHECKPOINTS + e] = dst_slot;
                            }
                        }
                        r.stage_ple_f8_for_batch(
                            &flat_tokens,
                            &q_start,
                            &st_in,
                            &st_out,
                            &has_init,
                            &spec_slots,
                            max_q_len,
                            &ckpt_dst,
                            &ckpt_off,
                            MAX_SPEC_VERIFY_CHECKPOINTS,
                        )?;
                    }
                    if r.batched_decode_io
                        .pinned_in_flight
                        .swap(false, std::sync::atomic::Ordering::AcqRel)
                    {
                        r.stream.synchronize()?;
                    }
                    let base = r.batched_decode_io.pinned_staging.host_ptr.cast::<u8>();
                    // SAFETY: `meta_bytes` (~1.2 KiB) fits inside `PINNED_META_BYTES` (160 KiB)
                    // and any prior slot-0 pinned transfer was synchronized above.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            host_meta.as_ptr().cast::<u8>(),
                            base,
                            meta_bytes,
                        );
                        r.batched_decode_io
                            .pinned_in_flight
                            .store(true, std::sync::atomic::Ordering::Release);
                        check_rc(
                            ffi::warp34_memcpy_h2d_async(
                                r.batched_spec_verify_io.meta_i32.addr() as *mut c_void,
                                base.cast(),
                                meta_bytes,
                                r.stream.handle,
                            ),
                            "upload pinned BatchedSpecVerifyGraphIo meta",
                        )?;
                    }
                    r.batched_decode_io
                        .upload_block_tables_slot(0, bt_off, slices, &r.stream)
                }
                ReplayPhase::Capture => r.ensure_batched_spec_verify_graph(
                    num_reqs,
                    total_tokens,
                    max_q_len,
                    long_ctx,
                    sample_mode,
                    lora_mode,
                    comm.as_deref(),
                ),
                ReplayPhase::Launch => {
                    r.bind_graph_lora_mode(
                        lora_mode,
                        r.batched_spec_verify_io.token_lora_ids_ptr(),
                    );
                    let (captured_logits_ptr, graph) = r
                        .find_batched_spec_verify_graph(
                            num_reqs,
                            total_tokens,
                            max_q_len,
                            long_ctx,
                            sample_mode,
                            lora_mode,
                        )
                        .ok_or_else(|| loader_err("missing captured batched_spec_verify graph"))?;
                    graph.launch(&r.stream)?;
                    r.mark_tiered_moe_device_lru_dirty();
                    r.speculative_logits_rows = if r.has_spec_logits_capacity(total_tokens) {
                        total_tokens
                    } else {
                        0
                    };
                    r.last_logits_ptr = captured_logits_ptr;
                    r.last_logits_rows = total_tokens;
                    Ok(())
                }
            }
        });
        finish_graph_replay(result)?;
        if skip_host_greedy {
            return Ok(vec![Vec::new(); num_reqs]);
        }
        let mut flat_ids = vec![0i32; total_tokens];
        self.ranks[0].device.make_current()?;
        self.ranks[0]
            .batched_spec_verify_io
            .out_greedy_ids
            .copy_to_host(&mut flat_ids, &self.ranks[0].stream)?;
        self.ranks[0]
            .batched_decode_io
            .pinned_in_flight
            .store(false, std::sync::atomic::Ordering::Release);
        for r in &self.ranks[1..] {
            r.device.make_current()?;
            r.stream.synchronize()?;
            r.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
        let mut per_req = Vec::with_capacity(num_reqs);
        let mut cur = 0usize;
        for s in slices {
            let k = s.tokens.len();
            per_req.push(flat_ids[cur..cur + k].to_vec());
            cur += k;
        }
        Ok(per_req)
    }

    pub fn captured_decode_graph_keys(&self) -> Vec<(usize, bool)> {
        let mut keys: Vec<(usize, bool)> = self.ranks[0]
            .batched_decode_graphs
            .iter()
            .map(|(k, _, _)| (k.0, k.1))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    pub fn captured_spec_verify_graph_shapes(&self) -> Vec<(usize, bool)> {
        let mut shapes: Vec<(usize, bool)> = self.ranks[0]
            .spec_verify_graphs
            .iter()
            .map(|(k, _, _)| (k.0, k.1))
            .collect();
        shapes.sort_unstable();
        shapes.dedup();
        shapes
    }

    pub fn captured_batched_spec_verify_graph_shapes(&self) -> Vec<(usize, usize, usize, bool)> {
        let mut shapes: Vec<(usize, usize, usize, bool)> = self.ranks[0]
            .batched_spec_verify_graphs
            .iter()
            .map(|(k, _, _)| (k.0, k.1, k.2, k.3))
            .collect();
        shapes.sort_unstable();
        shapes.dedup();
        shapes
    }

    pub fn captured_batched_mtp_propose_graph_shapes(&self) -> Vec<(usize, usize, bool)> {
        self.speculative
            .as_ref()
            .map_or_else(Vec::new, |p| p.captured_batched_mtp_propose_graph_shapes())
    }

    fn step_batched_decode_graph(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        bucket_b: usize,
        long_ctx: bool,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let n = slices.len();
        let lora_mode = self.batch_lora_mode(n);
        let slice_lids =
            (!self.slice_lora_ids.is_empty()).then_some(self.slice_lora_ids.as_slice());
        let (host_meta, host_masks) = BatchedDecodeGraphIo::pack_meta_with_lora(
            slices,
            rope_pos_deltas,
            slice_lids,
            self.active_lora_id,
        )?;
        use crate::coordinated::{ReplayPhase, replay_ranks};
        let comm = self.tp2_comm.as_ref().cloned();
        let stage_bytes = bucket_b
            .checked_mul(self.cfg.hidden_size)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| loader_err("tp2 staging payload overflow"))?;
        if comm
            .as_ref()
            .is_some_and(|c| stage_bytes > c.max_stage_bytes_per_epoch)
        {
            return Err(loader_err(
                "tp2 batch staging bytes exceed max_stage_bytes_per_epoch",
            ));
        }
        let sample_mode = if skip_host_greedy {
            DecodeGraphSampleMode::LogitsOnly
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };
        // Capture records nodes; it never executes collectives. Complete all uploads
        // and captures before launching either rank, including on first-use buckets.
        let result = replay_ranks(self.ranks.len(), |phase, rank| {
            let r = &mut self.ranks[rank];
            r.device.make_current()?;
            match phase {
                ReplayPhase::Upload => {
                    r.stage_ple_f8_for_decode_slices(slices)?;
                    r.batched_decode_io
                        .upload_inputs(&host_meta, &host_masks, slices, &r.stream)
                }
                ReplayPhase::Capture => r.ensure_batched_decode_graph(
                    bucket_b,
                    long_ctx,
                    sample_mode,
                    lora_mode,
                    comm.as_deref(),
                ),
                ReplayPhase::Launch => {
                    r.bind_graph_lora_mode(lora_mode, r.batched_decode_io.token_lora_ids_ptr());
                    let (logits_ptr, graph) = r
                        .find_batched_decode_graph(bucket_b, long_ctx, sample_mode, lora_mode)
                        .unwrap();
                    graph.launch(&r.stream)?;
                    r.mark_tiered_moe_device_lru_dirty();
                    r.speculative_logits_rows = if r.has_spec_logits_capacity(bucket_b) {
                        bucket_b
                    } else {
                        0
                    };
                    r.last_logits_ptr = logits_ptr;
                    r.last_logits_rows = bucket_b;
                    r.last_mtp_hidden_ptr = 0;
                    r.last_mtp_hidden_rows = 0;
                    Ok(())
                }
            }
        });
        finish_graph_replay(result)?;
        if skip_host_greedy {
            return Ok(Vec::new());
        }
        let mut out_ids = vec![0i32; n];
        self.ranks[0].device.make_current()?;
        self.ranks[0]
            .batched_decode_io
            .out_greedy_ids
            .copy_to_host(&mut out_ids, &self.ranks[0].stream)?;
        self.ranks[0]
            .batched_decode_io
            .pinned_in_flight
            .store(false, std::sync::atomic::Ordering::Release);
        for r in &self.ranks[1..] {
            r.device.make_current()?;
            r.stream.synchronize()?;
            r.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
        Ok(out_ids.into_iter().map(|t| vec![t]).collect())
    }

    pub fn can_pipeline_decode_slices(&self, slices: &[EngineBatchSlice<'_>]) -> bool {
        if self.tp_size != 1 || !self.use_cuda_graph || self.cfg.uses_ple() || slices.is_empty() {
            return false;
        }
        let Some(bucket_b) = decode_graph_bucket(slices.len()) else {
            return false;
        };
        let lora_mode = self.batch_lora_mode(slices.len());
        let r0 = &self.ranks[0];
        if r0.current_pos_plane_stride != 0
            || r0.pending_visual_splice.is_some()
            || !r0.supports_cuda_graph_for_tokens_with_lora(bucket_b, lora_mode)
        {
            return false;
        }
        let max_pages = r0.kv_pool.num_pages;
        let max_blocks = r0.batched_decode_io.max_blocks_per_req;
        slices.iter().all(|s| {
            let seq_after = s.ctx_len_before + 1;
            s.tokens.len() == 1
                && s.spec_slots.is_empty()
                && s.checkpoints.len() <= 1
                && s.block_table.len() <= max_blocks
                && seq_after.div_ceil(16) <= s.block_table.len()
                && s.block_table
                    .iter()
                    .all(|&p| p >= 0 && (p as usize) < max_pages)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_pipelined_decode_step(
        &mut self,
        slot: usize,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        tokens_already_on_device: bool,
        scatter_next_tokens_on_device: bool,
        fast_greedy_inputs: Option<(
            &[ffi::Warp34SamplerRowCfg],
            &[ffi::Warp34SamplerSparseEntry],
        )>,
    ) -> Result<InFlightDecodeTicket> {
        if self.tp_size != 1 {
            return Err(loader_err("pipelined decode overlap requires tp_size == 1"));
        }
        if self.cfg.uses_ple() {
            return Err(loader_err(
                "pipelined decode overlap is not supported with PLE staging",
            ));
        }
        if let Some(deltas) = rope_pos_deltas
            && deltas.len() != slices.len()
        {
            return Err(loader_err("rope_pos_deltas length mismatch"));
        }
        let bucket_b = decode_graph_bucket(slices.len())
            .ok_or_else(|| loader_err("invalid batch size for pipelined decode graph"))?;
        let max_seq_after = slices
            .iter()
            .map(|s| s.ctx_len_before + 1)
            .max()
            .unwrap_or(1);
        let long_ctx = max_seq_after > 1024;
        let sample_mode = if fast_greedy_inputs.is_some() {
            DecodeGraphSampleMode::FastGreedy
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };

        let slot = slot & 1;
        let n = slices.len();
        let lora_mode = self.batch_lora_mode(n);
        let slice_lids =
            (!self.slice_lora_ids.is_empty()).then_some(self.slice_lora_ids.as_slice());
        let (host_meta, host_masks) = BatchedDecodeGraphIo::pack_meta_with_lora(
            slices,
            rope_pos_deltas,
            slice_lids,
            self.active_lora_id,
        )?;
        let r0 = &mut self.ranks[0];
        r0.device.make_current()?;
        if tokens_already_on_device
            && r0
                .find_batched_decode_graph(bucket_b, long_ctx, sample_mode, lora_mode)
                .is_none()
        {
            return Err(loader_err(
                "pipelined lookahead step missing pre-captured decode graph",
            ));
        }
        r0.ensure_batched_decode_graph(bucket_b, long_ctx, sample_mode, lora_mode, None)?;
        r0.stage_ple_f8_for_decode_slices(slices)?;
        r0.batched_decode_io.upload_inputs_slot(
            slot,
            &host_meta,
            &host_masks,
            slices,
            tokens_already_on_device,
            true,
            &r0.stream,
        )?;
        if let Some((cfgs, sparse)) = fast_greedy_inputs {
            r0.batched_decode_io
                .upload_fast_greedy_slot(slot, cfgs, sparse, bucket_b, &r0.stream)?;
        }
        r0.bind_graph_lora_mode(lora_mode, r0.batched_decode_io.token_lora_ids_ptr());
        let (logits_ptr, graph) = r0
            .find_batched_decode_graph(bucket_b, long_ctx, sample_mode, lora_mode)
            .unwrap();
        graph.launch(&r0.stream)?;
        r0.mark_tiered_moe_device_lru_dirty();
        r0.speculative_logits_rows = if r0.has_spec_logits_capacity(bucket_b) {
            bucket_b
        } else {
            0
        };
        r0.last_logits_ptr = logits_ptr;
        r0.last_logits_rows = bucket_b;
        r0.last_mtp_hidden_ptr = 0;
        r0.last_mtp_hidden_rows = 0;

        if scatter_next_tokens_on_device {
            let ex0 = r0.stream.exec(0, 0);
            let (row_outs_ptr, greedy_ids_ptr) = if fast_greedy_inputs.is_some() {
                (
                    r0.batched_decode_io.fast_greedy_outs.addr() as *const ffi::Warp34SamplerRowOut,
                    core::ptr::null(),
                )
            } else {
                (
                    core::ptr::null(),
                    r0.batched_decode_io.out_greedy_ids.addr() as *const i32,
                )
            };
            check_rc(
                // SAFETY: n is within the decode bucket. Exactly one source is
                // nonnull, selected to match the captured sampler mode; both source
                // and token-ID storage are rank-owned static buffers. This launch
                // follows graph output production on the same stream.
                unsafe {
                    ffi::warp34_sampler_scatter_ids_to_tokens(
                        &ex0,
                        row_outs_ptr,
                        greedy_ids_ptr,
                        r0.batched_decode_io.token_ids_ptr() as *mut i32,
                        n as i32,
                    )
                },
                "warp34_sampler_scatter_ids_to_tokens",
            )?;
        }

        // SAFETY: slot is 0/1, and new() allocates two fixed-size slots with
        // room for MAX_BATCHED_DECODE_REQS outputs at this aligned offset.
        // The pipeline caller must finish a slot's ticket before reusing it.
        let pout = unsafe {
            r0.batched_decode_io
                .pinned_staging
                .host_ptr
                .cast::<u8>()
                .add(
                    slot * BatchedDecodeGraphIo::PINNED_SLOT_BYTES
                        + BatchedDecodeGraphIo::PINNED_PIPELINE_OUT_OFFSET,
                )
        };
        let (dev_out_ptr, d2h_bytes) = if fast_greedy_inputs.is_some() {
            (
                r0.batched_decode_io.fast_greedy_outs.addr(),
                n * core::mem::size_of::<ffi::Warp34SamplerRowOut>(),
            )
        } else {
            (r0.batched_decode_io.out_greedy_ids.addr(), n * 4)
        };
        check_rc(
            // SAFETY: n <= MAX_BATCHED_DECODE_REQS bounds source and pinned output
            // extents for either ABI. The rank retains the pinned allocation;
            // the recorded completion event must be waited before host reads/reuse.
            unsafe {
                ffi::warp34_memcpy_d2h_async(
                    pout.cast(),
                    dev_out_ptr as *const c_void,
                    d2h_bytes,
                    r0.stream.handle,
                )
            },
            "pipelined decode D2H output",
        )?;
        let mtp_row_bytes = self
            .speculative
            .as_ref()
            .filter(|p| p.is_mtp())
            .map_or(0, |p| p.row_bytes());
        let has_hidden = if mtp_row_bytes > 0
            && let Some(hidden_buf) = r0.last_final_hidden_f16.as_ref()
            && n.saturating_mul(mtp_row_bytes) <= hidden_buf.len_bytes
        {
            let hidden_addr = hidden_buf.addr();
            let dst = r0
                .batched_decode_io
                .ensure_pipeline_hidden_slot_ptr(slot, mtp_row_bytes)?;
            check_rc(
                // SAFETY: n <= MAX_BATCHED_DECODE_REQS and n * mtp_row_bytes <= hidden_buf.len_bytes.
                // Ordered on r0.stream strictly after graph.launch(t) and before pipeline_done_events[slot].
                unsafe {
                    ffi::warp34_memcpy_d2h_async(
                        dst.cast(),
                        hidden_addr as *const c_void,
                        n * mtp_row_bytes,
                        r0.stream.handle,
                    )
                },
                "pipelined decode D2H MTP hidden",
            )?;
            true
        } else {
            false
        };
        r0.batched_decode_io.pipeline_done_events[slot].record(&r0.stream)?;

        Ok(InFlightDecodeTicket {
            slot,
            num_reqs: n,
            fast_greedy: fast_greedy_inputs.is_some(),
            has_hidden,
        })
    }

    pub fn wait_pipelined_decode_step(
        &mut self,
        ticket: InFlightDecodeTicket,
    ) -> Result<PipelinedDecodeOutput> {
        let r0 = &mut self.ranks[0];
        r0.device.make_current()?;
        r0.batched_decode_io.pipeline_done_events[ticket.slot].synchronize()?;
        if ticket.slot == 0 {
            r0.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            r0.batched_decode_io
                .pinned_sampler_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
        // SAFETY: a ticket returned by enqueue identifies one of the two slots;
        // the event wait above completes its D2H. Ticket fields are public, so
        // provenance and not reusing the slot early remain caller obligations.
        let pout = unsafe {
            r0.batched_decode_io
                .pinned_staging
                .host_ptr
                .cast::<u8>()
                .add(
                    ticket.slot * BatchedDecodeGraphIo::PINNED_SLOT_BYTES
                        + BatchedDecodeGraphIo::PINNED_PIPELINE_OUT_OFFSET,
                )
        };
        if ticket.fast_greedy {
            let mut outs = vec![ffi::Warp34SamplerRowOut::default(); ticket.num_reqs];
            // SAFETY: for an unmodified completed ticket, this slot contains
            // num_reqs initialized RowOut records. The offset is 64-byte aligned;
            // outs is a distinct allocation with the same record count.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pout.cast::<ffi::Warp34SamplerRowOut>(),
                    outs.as_mut_ptr(),
                    ticket.num_reqs,
                );
            }
            Ok(PipelinedDecodeOutput::FastGreedy(outs))
        } else {
            let mut ids = vec![0i32; ticket.num_reqs];
            // SAFETY: an unmodified plain-greedy ticket denotes num_reqs i32
            // values whose D2H completed above. Slot offsets are i32-aligned and
            // ids is a separate allocation sized to exactly that count.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pout.cast::<i32>(),
                    ids.as_mut_ptr(),
                    ticket.num_reqs,
                );
            }
            Ok(PipelinedDecodeOutput::PlainGreedy(ids))
        }
    }

    pub fn abort_pipelined_decode(&mut self) {
        for r in &self.ranks {
            let _ = r.device.make_current();
            let _ = r.stream.synchronize();
            let _ = r.aux_stream.synchronize();
            r.batched_decode_io
                .pinned_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            r.batched_decode_io
                .pinned_sampler_in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    /// Executes a multi-request / speculative / checkpoint-capturing batch step across `TP=1` or `TP=2`.
    /// Returns one `Vec<i32>` of greedy token IDs per slice in `slices` (either `[last_tok]` or all
    /// `q_len` query tokens when `slice.emit_all_query_tokens == true`).
    pub fn step_batch_slices(&mut self, slices: &[EngineBatchSlice<'_>]) -> Result<Vec<Vec<i32>>> {
        self.step_batch_slices_with_pos_deltas(slices, None)
    }

    pub fn step_batch_slices_with_pos_deltas(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
    ) -> Result<Vec<Vec<i32>>> {
        self.step_batch_slices_for_sampler(slices, rope_pos_deltas, false)
    }

    /// Internal batch-step entrypoint that optionally skips host greedy argmax D2H (`Ok(Vec::new())`)
    /// when `skip_host_greedy == true` and the caller immediately samples device logits via `run_gpu_sampler_rows`.
    pub(crate) fn step_batch_slices_for_sampler(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        self.step_batch_slices_for_sampler_filtered(slices, rope_pos_deltas, None, skip_host_greedy)
    }

    /// Batch-step entrypoint with an optional per-slice `emit_mask`. When `emit_mask[r] == false`
    /// (e.g. non-final chunked-prefill slices whose logits are discarded), skips selecting/projecting
    /// `lm_head` rows for slice `r`.
    pub(crate) fn step_batch_slices_for_sampler_filtered(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        emit_mask: Option<&[bool]>,
        skip_host_greedy: bool,
    ) -> Result<Vec<Vec<i32>>> {
        self.step_batch_slices_for_sampler_filtered_bidir(
            slices,
            rope_pos_deltas,
            emit_mask,
            skip_host_greedy,
            false,
        )
    }

    fn try_step_spec_verify_graph_fastpath(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        all_emit: bool,
        noncausal_override: bool,
        skip_host_greedy: bool,
    ) -> Option<Result<Vec<Vec<i32>>>> {
        if rope_pos_deltas.is_some() || !all_emit || noncausal_override || !self.use_cuda_graph {
            return None;
        }
        if slices.len() == 1 {
            return self.try_single_spec_verify_graph(&slices[0], skip_host_greedy);
        }
        if (2..=MAX_BATCHED_SPEC_REQS).contains(&slices.len()) {
            return self.try_batched_spec_verify_graph(slices, skip_host_greedy);
        }
        None
    }

    fn try_single_spec_verify_graph(
        &mut self,
        s: &EngineBatchSlice<'_>,
        skip_host_greedy: bool,
    ) -> Option<Result<Vec<Vec<i32>>>> {
        let q_len = s.tokens.len();
        if q_len == 0 {
            return Some(Err(loader_err("empty slice in step_batch_slices")));
        }
        let max_pages = self.ranks[0].kv_pool.num_pages;
        let seq_after = s.ctx_len_before + q_len;
        if seq_after.div_ceil(16) > s.block_table.len()
            || s.block_table
                .iter()
                .any(|&p| p < 0 || (p as usize) >= max_pages)
        {
            return Some(Err(loader_err(
                "KV cache page capacity exceeded in step_batch_slices",
            )));
        }
        let long_ctx = !self.cfg.uses_qsa() && seq_after > 1024;
        let sample_mode = if skip_host_greedy {
            DecodeGraphSampleMode::LogitsOnly
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };
        let lora_mode = self.batch_lora_mode(1);
        let spec_key = (q_len, long_ctx, sample_mode, lora_mode);
        let can_use_spec_graph = (1..=MAX_SPEC_VERIFY_TOKENS).contains(&q_len)
            && s.has_initial_state
            && s.emit_all_query_tokens
            && s.spec_slots.len() == q_len
            && s.checkpoints.len() <= MAX_SPEC_VERIFY_CHECKPOINTS
            && s.block_table.len() <= self.ranks[0].spec_verify_io.max_blocks_per_req
            && self
                .ranks
                .iter()
                .all(|r| r.supports_cuda_graph_for_tokens_with_lora(q_len, lora_mode))
            && self.ranks.iter().all(|r| {
                let expected_m = r
                    .speculative_logits
                    .as_ref()
                    .map(|b| b.len_bytes / (r.local_vocab_size * 2));
                r.current_pos_plane_stride == 0
                    && r.pending_visual_splice.is_none()
                    && !r.failed_spec_verify_keys.contains(&spec_key)
                    && expected_m.is_none_or(|cap| q_len <= cap)
            });
        if !can_use_spec_graph {
            return None;
        }
        let needs_capture = self.ranks.iter().any(|r| {
            r.find_spec_verify_graph(q_len, long_ctx, sample_mode, lora_mode)
                .is_none()
        });
        match self.step_spec_verify_graph(s, skip_host_greedy) {
            Ok(out) => Some(Ok(out)),
            Err(e) if needs_capture => {
                for r in &mut self.ranks {
                    r.spec_verify_graphs.retain(|(k, _, _)| *k != spec_key);
                    if !r.failed_spec_verify_keys.contains(&spec_key) {
                        r.failed_spec_verify_keys.push(spec_key);
                    }
                }
                eprintln!(
                    "[warp34] warning: spec_verify CUDA Graph capture failed for shape {spec_key:?} ({e}); degrading shape to eager verification"
                );
                None
            }
            Err(e) => Some(Err(e)),
        }
    }

    fn try_batched_spec_verify_graph(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        skip_host_greedy: bool,
    ) -> Option<Result<Vec<Vec<i32>>>> {
        let max_pages = self.ranks[0].kv_pool.num_pages;
        let mut valid_inputs = true;
        let mut total_q = 0usize;
        let mut max_q = 0usize;
        let mut max_seq_after = 0usize;
        for s in slices {
            let q_len = s.tokens.len();
            if q_len == 0 {
                return Some(Err(loader_err("empty slice in step_batch_slices")));
            }
            let seq_after = s.ctx_len_before + q_len;
            if seq_after.div_ceil(16) > s.block_table.len()
                || s.block_table
                    .iter()
                    .any(|&p| p < 0 || (p as usize) >= max_pages)
            {
                return Some(Err(loader_err(
                    "KV cache page capacity exceeded in step_batch_slices",
                )));
            }
            total_q += q_len;
            max_q = max_q.max(q_len);
            max_seq_after = max_seq_after.max(seq_after);
            if !s.has_initial_state
                || !s.emit_all_query_tokens
                || s.spec_slots.len() != q_len
                || s.checkpoints.len() > MAX_SPEC_VERIFY_CHECKPOINTS
                || s.block_table.len() > self.ranks[0].batched_decode_io.max_blocks_per_req
            {
                valid_inputs = false;
            }
        }
        let long_ctx = !self.cfg.uses_qsa() && max_seq_after > 1024;
        let sample_mode = if skip_host_greedy {
            DecodeGraphSampleMode::LogitsOnly
        } else {
            DecodeGraphSampleMode::PlainGreedy
        };
        let lora_mode = self.batch_lora_mode(slices.len());
        let bspec_key = (
            slices.len(),
            total_q,
            max_q,
            long_ctx,
            sample_mode,
            lora_mode,
        );
        let can_use_batched_spec = valid_inputs
            && (1..=MAX_SPEC_VERIFY_TOKENS).contains(&max_q)
            && total_q <= MAX_BATCHED_SPEC_TOKENS
            && self
                .ranks
                .iter()
                .all(|r| r.supports_cuda_graph_for_tokens_with_lora(total_q, lora_mode))
            && self.ranks.iter().all(|r| {
                let taps_fit = (r.tap_layer_ids.is_empty() && r.last_final_hidden_f16.is_none())
                    || total_q <= r.max_tap_rows;
                r.current_pos_plane_stride == 0
                    && r.pending_visual_splice.is_none()
                    && !r.failed_batched_spec_verify_keys.contains(&bspec_key)
                    && taps_fit
            });
        if !can_use_batched_spec {
            return None;
        }
        let needs_capture = self.ranks.iter().any(|r| {
            r.find_batched_spec_verify_graph(
                slices.len(),
                total_q,
                max_q,
                long_ctx,
                sample_mode,
                lora_mode,
            )
            .is_none()
        });
        match self.step_batched_spec_verify_graph(
            slices,
            total_q,
            max_q,
            long_ctx,
            skip_host_greedy,
        ) {
            Ok(out) => Some(Ok(out)),
            Err(e) if needs_capture => {
                for r in &mut self.ranks {
                    r.batched_spec_verify_graphs
                        .retain(|(k, _, _)| *k != bspec_key);
                    if !r.failed_batched_spec_verify_keys.contains(&bspec_key) {
                        r.failed_batched_spec_verify_keys.push(bspec_key);
                    }
                }
                eprintln!(
                    "[warp34] warning: batched_spec_verify CUDA Graph capture failed for shape {bspec_key:?} ({e}); degrading shape to eager verification"
                );
                None
            }
            Err(e) => Some(Err(e)),
        }
    }

    pub(crate) fn step_batch_slices_for_sampler_filtered_bidir(
        &mut self,
        slices: &[EngineBatchSlice<'_>],
        rope_pos_deltas: Option<&[i32]>,
        emit_mask: Option<&[bool]>,
        skip_host_greedy: bool,
        noncausal_override: bool,
    ) -> Result<Vec<Vec<i32>>> {
        if slices.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(deltas) = rope_pos_deltas
            && deltas.len() != slices.len()
        {
            return Err(loader_err("rope_pos_deltas length mismatch"));
        }
        if let Some(mask) = emit_mask
            && mask.len() != slices.len()
        {
            return Err(loader_err("emit_mask length mismatch"));
        }
        let all_emit = emit_mask.is_none_or(|m| m.iter().all(|&e| e));
        let max_pages = self.ranks[0].kv_pool.num_pages;
        let lora_mode = self.batch_lora_mode(slices.len());
        if !noncausal_override
            && all_emit
            && let Some(bucket_b) = decode_graph_bucket(slices.len())
            && self.use_cuda_graph
            && self
                .ranks
                .iter()
                .all(|r| r.supports_cuda_graph_for_tokens_with_lora(bucket_b, lora_mode))
            && slices.iter().all(|s| {
                s.tokens.len() == 1
                    && s.spec_slots.is_empty()
                    && s.checkpoints.len() <= 1
                    && s.block_table.len() <= self.ranks[0].batched_decode_io.max_blocks_per_req
            })
            && self
                .ranks
                .iter()
                .all(|r| r.current_pos_plane_stride == 0 && r.pending_visual_splice.is_none())
        {
            let mut max_seq_after = 0usize;
            for s in slices {
                let seq_after = s.ctx_len_before + 1;
                if seq_after.div_ceil(16) > s.block_table.len()
                    || s.block_table
                        .iter()
                        .any(|&p| p < 0 || (p as usize) >= max_pages)
                {
                    return Err(loader_err(
                        "KV cache page capacity exceeded in step_batch_slices",
                    ));
                }
                max_seq_after = max_seq_after.max(seq_after);
            }
            let long_ctx = max_seq_after > 1024;
            // Failed preparation/capture has not run a forward pass. Surface the error
            // without disabling unrelated graph families or replaying mutable state.
            return self.step_batched_decode_graph(
                slices,
                rope_pos_deltas,
                bucket_b,
                long_ctx,
                skip_host_greedy,
            );
        }
        if let Some(res) = self.try_step_spec_verify_graph_fastpath(
            slices,
            rope_pos_deltas,
            all_emit,
            noncausal_override,
            skip_host_greedy,
        ) {
            return res;
        }
        let num_reqs = slices.len();
        let mut flat_tokens = Vec::new();
        let mut flat_positions = Vec::new();
        let mut flat_write_slots = Vec::new();
        let mut q_start_loc = Vec::with_capacity(num_reqs + 1);
        q_start_loc.push(0i32);
        let mut ctx_lens_before = Vec::with_capacity(num_reqs);
        let mut seq_lens_after = Vec::with_capacity(num_reqs);
        let mut state_in_indices = Vec::with_capacity(num_reqs);
        let mut state_out_indices = Vec::with_capacity(num_reqs);
        let mut has_initial_state = Vec::with_capacity(num_reqs);
        let mut selected_rows = Vec::new();
        let mut row_counts_per_req = Vec::with_capacity(num_reqs);

        let max_blocks_per_req = slices
            .iter()
            .map(|s| s.block_table.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let max_verify_tokens = slices.iter().map(|s| s.spec_slots.len()).max().unwrap_or(0);
        let max_checkpoints_per_req = slices
            .iter()
            .map(|s| s.checkpoints.len())
            .max()
            .unwrap_or(0);
        let mut max_q_len = 0usize;
        let mut max_seq_len = 0usize;
        let max_pages = self.ranks[0].kv_pool.num_pages;

        let mut block_tables = vec![0i32; num_reqs * max_blocks_per_req];
        let mut spec_state_indices = if max_verify_tokens > 0 {
            vec![-1i32; num_reqs * max_verify_tokens]
        } else {
            Vec::new()
        };
        let mut checkpoint_dst_indices = if max_checkpoints_per_req > 0 {
            vec![-1i32; num_reqs * max_checkpoints_per_req]
        } else {
            Vec::new()
        };
        let mut checkpoint_token_offsets = if max_checkpoints_per_req > 0 {
            vec![-1i32; num_reqs * max_checkpoints_per_req]
        } else {
            Vec::new()
        };

        for (r, s) in slices.iter().enumerate() {
            let q_len = s.tokens.len();
            if q_len == 0 {
                return Err(loader_err("empty slice in step_batch_slices"));
            }
            let seq_after = s.ctx_len_before + q_len;
            if seq_after.div_ceil(16) > s.block_table.len()
                || s.block_table
                    .iter()
                    .any(|&p| p < 0 || (p as usize) >= max_pages)
            {
                return Err(loader_err(
                    "KV cache page capacity exceeded in step_batch_slices",
                ));
            }
            let base_tok = flat_tokens.len();
            flat_tokens.extend_from_slice(s.tokens);
            let delta = rope_pos_deltas.and_then(|d| d.get(r).copied()).unwrap_or(0);
            for step in 0..q_len {
                let pos = s.ctx_len_before + step;
                flat_positions.push((pos as i32) + delta);
                let page_id = s.block_table[pos / 16];
                let page_off = (pos % 16) as i32;
                flat_write_slots.push(page_id * 16 + page_off);
            }
            let emit_slice = emit_mask.is_none_or(|m| m[r]);
            if !emit_slice {
                row_counts_per_req.push(0);
            } else if s.emit_all_query_tokens {
                for step in 0..q_len {
                    selected_rows.push(base_tok + step);
                }
                row_counts_per_req.push(q_len);
            } else {
                selected_rows.push(base_tok + q_len - 1);
                row_counts_per_req.push(1);
            }
            q_start_loc.push(flat_tokens.len() as i32);
            ctx_lens_before.push(s.ctx_len_before as i32);
            seq_lens_after.push(seq_after as i32);
            state_in_indices.push(s.state_in_slot);
            state_out_indices.push(s.state_out_slot);
            has_initial_state.push(u8::from(s.has_initial_state));
            max_q_len = max_q_len.max(q_len);
            max_seq_len = max_seq_len.max(seq_after);

            block_tables[r * max_blocks_per_req..r * max_blocks_per_req + s.block_table.len()]
                .copy_from_slice(s.block_table);
            if max_verify_tokens > 0 {
                for (k, &slot) in s.spec_slots.iter().enumerate() {
                    spec_state_indices[r * max_verify_tokens + k] = slot;
                }
            }
            if max_checkpoints_per_req > 0 {
                for (e, &(trig_off, dst_slot)) in s.checkpoints.iter().enumerate() {
                    checkpoint_token_offsets[r * max_checkpoints_per_req + e] = trig_off;
                    checkpoint_dst_indices[r * max_checkpoints_per_req + e] = dst_slot;
                }
            }
        }

        let total_tokens = flat_tokens.len();
        let first_lora_id = self.slice_lora_id(0);
        let all_same_lora = (1..slices.len()).all(|r| self.slice_lora_id(r) == first_lora_id);
        let flat_lora_ids: Option<Vec<i32>> = (!all_same_lora && !self.lora_registry.is_empty())
            .then(|| {
                let mut out = Vec::with_capacity(total_tokens);
                for (r, s) in slices.iter().enumerate() {
                    let lid = self.slice_lora_id(r);
                    out.resize(out.len() + s.tokens.len(), lid);
                }
                out
            });
        let flat_lora_ref = flat_lora_ids.as_deref();
        let flat_ids = if self.tp_size == 1 {
            let r0 = &mut self.ranks[0];
            let (tok_ptr, pos_ptr, mut attn_batch, gdn_batch) = r0.upload_batch_meta(
                &flat_tokens,
                &flat_positions,
                &q_start_loc,
                &ctx_lens_before,
                &seq_lens_after,
                &block_tables,
                &flat_write_slots,
                &state_in_indices,
                &state_out_indices,
                &has_initial_state,
                &spec_state_indices,
                &checkpoint_dst_indices,
                &checkpoint_token_offsets,
                num_reqs,
                total_tokens,
                max_blocks_per_req,
                max_q_len,
                max_seq_len,
                max_verify_tokens,
                max_checkpoints_per_req,
            )?;
            r0.bind_lora_for_batch(Some(first_lora_id), flat_lora_ref)?;
            if noncausal_override {
                attn_batch.mask_mode = warp34_kernels::ffi::WARP34_ATTN_MASK_SYMMETRIC_WINDOW;
            }
            r0.forward_selected_rows_maybe_skip_greedy(
                tok_ptr,
                pos_ptr,
                total_tokens,
                &attn_batch,
                &gdn_batch,
                None,
                &selected_rows,
                skip_host_greedy,
            )?
        } else {
            let comm = self.tp2_comm.as_ref().unwrap().clone();
            let pool = self.tp_workers.as_ref().unwrap().clone();
            let ft_ref = &flat_tokens;
            let fp_ref = &flat_positions;
            let qs_ref = &q_start_loc;
            let cb_ref = &ctx_lens_before;
            let sa_ref = &seq_lens_after;
            let bt_ref = &block_tables;
            let ws_ref = &flat_write_slots;
            let si_ref = &state_in_indices;
            let so_ref = &state_out_indices;
            let hi_ref = &has_initial_state;
            let sp_ref = &spec_state_indices;
            let cd_ref = &checkpoint_dst_indices;
            let co_ref = &checkpoint_token_offsets;
            let sr_ref = &selected_rows;

            let stage_bytes = total_tokens
                .checked_mul(self.cfg.hidden_size)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| loader_err("tp2 staging payload overflow"))?;
            if stage_bytes > comm.max_stage_bytes_per_epoch {
                return Err(loader_err(
                    "tp2 batch staging bytes exceed max_stage_bytes_per_epoch",
                ));
            }

            let comm_ref = &comm;
            let (res0, peer_results) = pool.run_split(
                self.ranks.as_mut_slice(),
                |r0| -> Result<Vec<i32>> {
                    let (tok_ptr, pos_ptr, mut attn_batch, gdn_batch) = r0.upload_batch_meta(
                        ft_ref,
                        fp_ref,
                        qs_ref,
                        cb_ref,
                        sa_ref,
                        bt_ref,
                        ws_ref,
                        si_ref,
                        so_ref,
                        hi_ref,
                        sp_ref,
                        cd_ref,
                        co_ref,
                        num_reqs,
                        total_tokens,
                        max_blocks_per_req,
                        max_q_len,
                        max_seq_len,
                        max_verify_tokens,
                        max_checkpoints_per_req,
                    )?;
                    r0.bind_lora_for_batch(Some(first_lora_id), flat_lora_ref)?;
                    if noncausal_override {
                        attn_batch.mask_mode =
                            warp34_kernels::ffi::WARP34_ATTN_MASK_SYMMETRIC_WINDOW;
                    }
                    r0.forward_selected_rows_maybe_skip_greedy(
                        tok_ptr,
                        pos_ptr,
                        total_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(comm_ref),
                        sr_ref,
                        skip_host_greedy,
                    )
                },
                |_, r_peer| -> Result<()> {
                    let (tok_ptr, pos_ptr, mut attn_batch, gdn_batch) = r_peer.upload_batch_meta(
                        ft_ref,
                        fp_ref,
                        qs_ref,
                        cb_ref,
                        sa_ref,
                        bt_ref,
                        ws_ref,
                        si_ref,
                        so_ref,
                        hi_ref,
                        sp_ref,
                        cd_ref,
                        co_ref,
                        num_reqs,
                        total_tokens,
                        max_blocks_per_req,
                        max_q_len,
                        max_seq_len,
                        max_verify_tokens,
                        max_checkpoints_per_req,
                    )?;
                    r_peer.bind_lora_for_batch(Some(first_lora_id), flat_lora_ref)?;
                    if noncausal_override {
                        attn_batch.mask_mode =
                            warp34_kernels::ffi::WARP34_ATTN_MASK_SYMMETRIC_WINDOW;
                    }
                    let _ = r_peer.forward_selected_rows_maybe_skip_greedy(
                        tok_ptr,
                        pos_ptr,
                        total_tokens,
                        &attn_batch,
                        &gdn_batch,
                        Some(comm_ref),
                        sr_ref,
                        skip_host_greedy,
                    )?;
                    Ok(())
                },
            );
            for pr in peer_results {
                pr?;
            }
            res0?
        };

        if skip_host_greedy {
            return Ok(Vec::new());
        }
        let mut out_per_req = Vec::with_capacity(num_reqs);
        let mut cursor = 0usize;
        for count in row_counts_per_req {
            out_per_req.push(flat_ids[cursor..cursor + count].to_vec());
            cursor += count;
        }
        Ok(out_per_req)
    }
}
