//! Native proposer orchestration and transactional greedy generation.
//! Warm target radix hits are used only when the matching committed proposer
//! features are available. A bounded, single-branch host memo rematerializes
//! draft KV without rerunning target layers; it is not a second target cache.
use crate::{
    cache::StateTx,
    cuda::{DeviceBuffer, Result, check_rc, f16_to_f32},
    dflash::{DFLASH2_MAX_BATCH_ROWS, DFlash2Rank},
    loader::resolve_hf_model_dir,
    model::{EngineBatchSlice, Qwen3_5Engine, RadixBenchMetrics, loader_err},
    mtp::{MAX_MTP_BATCH_SLOTS, Qwen3_5MtpRank},
    parser::{SamplerV1, SamplingParams, TokenBitmaskConstraint},
    radix::{PlannedCheckpoint, ReqRadixState},
    speculative::RejectionSampler,
};
use std::time::Instant;
use warp34_kernels::ffi;

mod scheduled;

fn generation_stopped(sampler: &SamplerV1, anchor: i32, stop_eos: bool, model_eos: &[i32]) -> bool {
    // A closed grammar cannot be sampled again, even below min_tokens or when
    // ordinary EOS stopping is disabled.
    sampler
        .constraint
        .as_ref()
        .is_some_and(|c| c.is_terminal_eos())
        || (stop_eos
            && sampler.output_history.len() >= sampler.params.min_tokens
            && sampler.params.is_stop_or_eos(anchor, model_eos))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeculativeKind {
    Mtp,
    DFlash2,
}
#[derive(Clone, Debug)]
pub struct SpeculativeConfig {
    pub kind: SpeculativeKind,
    pub draft_model: Option<String>,
    pub mtp_k: usize,
}
enum Ranks {
    Mtp(Vec<Qwen3_5MtpRank>),
    DFlash(Vec<DFlash2Rank>),
}
struct PrefixMemo {
    tokens: Vec<i32>,
    features: Vec<u8>,
}
struct MtpBatchSlot {
    tail: Vec<DeviceBuffer>,
    captured_verify_hidden: Vec<DeviceBuffer>,
    resident_tokens: Vec<i32>,
    resident_valid_from: usize,
    processed: usize,
    current: Option<PrefixMemo>,
    last_used_tick: u64,
}
pub struct NativeProposer {
    ranks: Ranks,
    capacity: usize,
    total_capacity: usize,
    max_rows: usize,
    mtp_prefill_max_rows: usize,
    memo_budget: usize,
    row_bytes: usize,
    tap_row_bytes: usize,
    pub(crate) k: usize,
    max_k: usize,
    tail: Vec<DeviceBuffer>,
    shifted: Vec<DeviceBuffer>,
    memos: Vec<PrefixMemo>,
    current: Option<PrefixMemo>,
    plain_memos: std::collections::HashMap<u64, PrefixMemo>,
    memo_pinned_staging: Option<crate::cuda::MappedHostBuffer>,
    pending_plain_tap_rows: usize,
    resident_tokens: Vec<i32>,
    resident_valid_from: usize,
    processed: usize,
    mtp_batch_active_slot: usize,
    active_slot_last_used_tick: u64,
    batch_slot_tick_counter: u64,
    mtp_batch_slots: Vec<MtpBatchSlot>,
    pub last_accepted: Vec<usize>,
    /// Draft rows actually included in the last successful scheduled verify call
    /// (summed across requests for batched MTP), after budget/grammar/stop trimming.
    pub(crate) last_verified_draft_tokens: usize,
    pub last_restored_tokens: usize,
}
const MEMO_BUDGET: usize = 4 * 1024 * 1024 * 1024;
const MIN_DP_MEMO_BUDGET: usize = 512 * 1024 * 1024;
const MAX_MEMO_ENTRIES: usize = 64;
const MEMO_GROWTH_ROWS: usize = 512;
const SERVING_PROPOSER_MAX_TAP_ROWS: usize = 2048;
#[cfg(any(test, feature = "cuda"))]
#[inline]
pub(crate) fn plan_memo_budget(dp_size: usize) -> usize {
    (MEMO_BUDGET / dp_size.max(1)).max(MIN_DP_MEMO_BUDGET)
}

#[cfg(any(test, feature = "cuda"))]
#[inline]
pub(crate) fn memo_fits_budget_limit(
    _kind: SpeculativeKind,
    cur_tokens: usize,
    add_tokens: usize,
    row_bytes: usize,
    budget: usize,
) -> bool {
    let total_tokens = cur_tokens.saturating_add(add_tokens);
    total_tokens.saturating_mul(row_bytes) <= budget
}

#[cfg(test)]
#[inline]
pub(crate) fn memo_fits_budget(
    kind: SpeculativeKind,
    cur_tokens: usize,
    add_tokens: usize,
    row_bytes: usize,
) -> bool {
    memo_fits_budget_limit(kind, cur_tokens, add_tokens, row_bytes, MEMO_BUDGET)
}

fn grow_memo_features(buf: &mut Vec<u8>, new_len: usize, row_bytes: usize) {
    if new_len > buf.capacity() {
        let chunk = MEMO_GROWTH_ROWS.saturating_mul(row_bytes.max(1));
        let target_cap = new_len.div_ceil(chunk).saturating_mul(chunk);
        buf.reserve_exact(target_cap.saturating_sub(buf.len()));
    }
    buf.resize(new_len, 0);
}

pub(crate) fn dflash_effective_valid_from(
    res_len: usize,
    valid_from: usize,
    ring_pages: Option<usize>,
) -> usize {
    if let Some(phys_pages) = ring_pages
        && phys_pages > 0
    {
        let l_pages = res_len.div_ceil(16);
        let ring_floor = l_pages.saturating_sub(phys_pages) * 16;
        valid_from.max(ring_floor)
    } else {
        valid_from
    }
}

pub(crate) fn dflash_slot_has_prefix_for_prompt(
    res: &[i32],
    valid_from: usize,
    tokens: &[i32],
    prompt_len: usize,
    ring_pages: Option<usize>,
    window: usize,
) -> bool {
    if !res.starts_with(tokens) {
        return false;
    }
    if let Some(phys_pages) = ring_pages {
        let w = if window > 0 { window } else { 2048 };
        let effective_len = prompt_len.max(tokens.len());
        let win_start_page = effective_len.saturating_sub(w) / 16;
        let win_start_tok = win_start_page * 16;
        if tokens.len() <= win_start_tok {
            return true;
        }
        if valid_from > win_start_tok {
            return false;
        }
        let l_pages = res.len().div_ceil(16);
        if l_pages.saturating_sub(win_start_page) > phys_pages {
            return false;
        }
    } else if valid_from > 0 {
        return false;
    }
    true
}

fn push_mtp_memo_with_budget(memos: &mut Vec<PrefixMemo>, memo: PrefixMemo, budget: usize) {
    if memo.tokens.is_empty() {
        return;
    }
    memos.retain(|existing| {
        !(memo.tokens.starts_with(&existing.tokens)
            && memo.features.len() >= existing.features.len())
    });
    if let Some(pos) = memos.iter().position(|existing| {
        existing.tokens.starts_with(&memo.tokens) && existing.features.len() >= memo.features.len()
    }) {
        let existing = memos.remove(pos);
        memos.push(existing);
        return;
    }
    let mut total_bytes: usize = memos.iter().map(|m| m.features.capacity()).sum();
    let incoming = memo.features.capacity();
    while !memos.is_empty()
        && (memos.len() >= MAX_MEMO_ENTRIES || total_bytes.saturating_add(incoming) > budget)
    {
        let removed = memos.remove(0);
        total_bytes = total_bytes.saturating_sub(removed.features.capacity());
    }
    if incoming <= budget {
        memos.push(memo);
    }
}

#[cfg(test)]
fn push_mtp_memo(memos: &mut Vec<PrefixMemo>, memo: PrefixMemo) {
    push_mtp_memo_with_budget(memos, memo, MEMO_BUDGET);
}
const MTP_STAGE_ROWS: usize = 256;
const TIERED_MOE_MTP_MAX_TAP_ROWS: usize = 6144;
const TIERED_MOE_DFLASH2_MAX_TAP_ROWS: usize = 2048;
pub(crate) fn plan_proposer_max_rows(
    uses_tiered_moe: bool,
    kind: SpeculativeKind,
    tp_size: usize,
    num_pages: usize,
) -> usize {
    let capacity = num_pages.saturating_mul(16);
    if uses_tiered_moe {
        match kind {
            SpeculativeKind::Mtp => capacity.min(TIERED_MOE_MTP_MAX_TAP_ROWS),
            SpeculativeKind::DFlash2 => capacity.min(TIERED_MOE_DFLASH2_MAX_TAP_ROWS),
        }
    } else if num_pages >= 4096 && (num_pages < 14336 || tp_size > 1) {
        capacity.min(SERVING_PROPOSER_MAX_TAP_ROWS)
    } else if num_pages >= 14336 {
        capacity.min(crate::scheduler::MEDIUM_SCRATCH_MAX_BATCHED_TOKENS)
    } else {
        capacity.min(DFLASH2_MAX_BATCH_ROWS)
    }
}

impl NativeProposer {
    pub fn set_mtp_k(&mut self, k: usize) -> Result<()> {
        if !matches!(self.ranks, Ranks::Mtp(_)) || !(1..=self.max_k).contains(&k) {
            return Err(loader_err(
                "set_mtp_k requires MTP proposer and k in 1..=max_k",
            ));
        }
        self.k = k;
        Ok(())
    }

    pub fn load(engine: &mut Qwen3_5Engine, config: &SpeculativeConfig) -> Result<Self> {
        for t in &mut engine.ranks {
            t.device.make_current()?;
            t.stream.synchronize()?;
            t.decode_graph.clear();
            t.spec_verify_graphs.clear();
            t.failed_spec_verify_keys.clear();
            t.batched_spec_verify_graphs.clear();
            t.failed_batched_spec_verify_keys.clear();
            t.batched_decode_graphs.clear();
        }
        if config.kind == SpeculativeKind::Mtp
            && (!matches!(config.mtp_k, 1..=3) || config.draft_model.is_some())
        {
            return Err(loader_err(
                "native MTP requires k=1..3 and uses the target checkpoint",
            ));
        }
        let num_pages = engine.ranks[0].kv_pool.num_pages;
        let capacity = num_pages * 16;
        let mut max_rows = plan_proposer_max_rows(
            engine.cfg.uses_tiered_moe(),
            config.kind,
            engine.tp_size,
            num_pages,
        );
        if let Ok(prefill_cap) = engine.ranks[0].ordinary_prefill_token_capacity(1) {
            let aligned_cap = (prefill_cap / 16) * 16;
            if aligned_cap >= DFLASH2_MAX_BATCH_ROWS {
                max_rows = max_rows.min(aligned_cap);
            }
        }
        if max_rows < 16 {
            return Err(loader_err("insufficient speculative context capacity"));
        }
        let h = engine.cfg.hidden_size;
        let mtp_row_bytes = engine.cfg.hyper_hidden_size() * 2;
        let use_graph = engine.use_cuda_graph;
        let (ranks, taps, k, row_bytes, tap_row_bytes) = match config.kind {
            SpeculativeKind::Mtp => {
                let model_dir = &engine.model_dir;
                let ranks = if engine.ranks.len() <= 1 {
                    engine
                        .ranks
                        .iter()
                        .map(|t| {
                            let mut r = Qwen3_5MtpRank::load(t, model_dir, capacity)?;
                            r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                            Ok(r)
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    std::thread::scope(|s| -> Result<Vec<Qwen3_5MtpRank>> {
                        let handles: Vec<_> = engine
                            .ranks
                            .iter()
                            .map(|t| {
                                s.spawn(move || {
                                    let mut r = Qwen3_5MtpRank::load(t, model_dir, capacity)?;
                                    r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                                    Ok(r)
                                })
                            })
                            .collect();
                        let mut out = Vec::with_capacity(handles.len());
                        for h in handles {
                            out.push(
                                h.join()
                                    .map_err(|_| loader_err("MTP rank loader thread panicked"))??,
                            );
                        }
                        Ok(out)
                    })?
                };
                (
                    Ranks::Mtp(ranks),
                    vec![],
                    config.mtp_k,
                    mtp_row_bytes,
                    mtp_row_bytes,
                )
            }
            SpeculativeKind::DFlash2 => {
                let name = config
                    .draft_model
                    .as_ref()
                    .ok_or_else(|| loader_err("DFlash2 requires --draft-model"))?;
                let dir = resolve_hf_model_dir(name)
                    .map_err(|_| loader_err("resolve DFlash2 checkpoint"))?;
                let dflash_max_rows = max_rows.min(DFLASH2_MAX_BATCH_ROWS);
                let ranks = if engine.ranks.len() <= 1 {
                    engine
                        .ranks
                        .iter()
                        .map(|t| {
                            let mut r = DFlash2Rank::load(t, &dir, capacity, dflash_max_rows)?;
                            r.use_cuda_graph = use_graph;
                            Ok(r)
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    let dir_ref = &dir;
                    std::thread::scope(|s| -> Result<Vec<DFlash2Rank>> {
                        let handles: Vec<_> = engine
                            .ranks
                            .iter()
                            .map(|t| {
                                s.spawn(move || {
                                    let mut r =
                                        DFlash2Rank::load(t, dir_ref, capacity, dflash_max_rows)?;
                                    r.use_cuda_graph = use_graph;
                                    Ok(r)
                                })
                            })
                            .collect();
                        let mut out = Vec::with_capacity(handles.len());
                        for h in handles {
                            out.push(h.join().map_err(|_| {
                                loader_err("DFlash2 rank loader thread panicked")
                            })??);
                        }
                        Ok(out)
                    })?
                };
                let taps = ranks[0].cfg.target_layers.clone();
                let k = ranks[0].cfg.block_size - 1;
                // Host memos store projected `ctx = rms_norm(fc(tap_aux))` (`1 × H × 2B`, 5× smaller
                // than raw `tap_aux_bf16` and 0-ULP exact); `tap_row_bytes` tracks raw GPU tap stride.
                let row_bytes = h * 2;
                let tap_row_bytes = h * taps.len() * 2;
                (Ranks::DFlash(ranks), taps, k, row_bytes, tap_row_bytes)
            }
        };
        let tail = engine
            .ranks
            .iter()
            .map(|t| DeviceBuffer::alloc(t.device.device_id, mtp_row_bytes))
            .collect::<Result<Vec<_>>>()?;
        let shifted = engine
            .ranks
            .iter()
            .map(|t| {
                DeviceBuffer::alloc(t.device.device_id, max_rows.min(MTP_STAGE_ROWS) * row_bytes)
            })
            .collect::<Result<Vec<_>>>()?;
        let logits = engine
            .ranks
            .iter()
            .map(|t| DeviceBuffer::alloc(t.device.device_id, (k + 1) * t.local_vocab_size * 2))
            .collect::<Result<Vec<_>>>()?;
        let try_configure_taps = |engine: &mut Qwen3_5Engine, rows: usize| -> Result<()> {
            for rank in 0..engine.ranks.len() {
                if let Err(e) = engine.ranks[rank].configure_speculative_taps(
                    &taps,
                    config.kind == SpeculativeKind::Mtp,
                    rows,
                ) {
                    for t in &mut engine.ranks {
                        let _ = t.configure_speculative_taps(&[], false, 0);
                    }
                    return Err(e);
                }
            }
            Ok(())
        };
        if let Err(e) = try_configure_taps(engine, max_rows) {
            let fallback_rows = capacity.min(DFLASH2_MAX_BATCH_ROWS);
            if max_rows > fallback_rows {
                max_rows = fallback_rows;
                try_configure_taps(engine, max_rows)?;
            } else {
                return Err(e);
            }
        }
        for (t, buffer) in engine.ranks.iter_mut().zip(logits) {
            t.speculative_logits = Some(buffer);
        }
        let slot_count = match config.kind {
            SpeculativeKind::Mtp => MAX_MTP_BATCH_SLOTS,
            SpeculativeKind::DFlash2 => crate::dflash::MAX_DFLASH_BATCH_SLOTS,
        };
        let mtp_batch_slots = (0..slot_count)
            .map(|_| {
                let slot_tail = if config.kind == SpeculativeKind::Mtp {
                    engine
                        .ranks
                        .iter()
                        .map(|t| DeviceBuffer::alloc(t.device.device_id, mtp_row_bytes))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    vec![]
                };
                let captured_verify_hidden = engine
                    .ranks
                    .iter()
                    .map(|t| DeviceBuffer::alloc(t.device.device_id, (k + 1) * tap_row_bytes))
                    .collect::<Result<Vec<_>>>()?;
                Ok(MtpBatchSlot {
                    tail: slot_tail,
                    captured_verify_hidden,
                    resident_tokens: vec![],
                    resident_valid_from: 0,
                    processed: 0,
                    current: None,
                    last_used_tick: 0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let pinned_stage_bytes = max_rows
            .min(MTP_STAGE_ROWS)
            .saturating_mul(row_bytes)
            .max(256 * 1024);
        let memo_pinned_staging = crate::cuda::MappedHostBuffer::alloc(pinned_stage_bytes).ok();
        let mtp_prefill_max_rows = if config.kind == SpeculativeKind::Mtp
            && !engine.cfg.uses_tiered_moe()
            && !(num_pages >= 14336 && engine.tp_size == 1)
        {
            engine.ranks[0]
                .ordinary_prefill_token_capacity(1)
                .map_or(max_rows, |c| {
                    ((c / 64) * 64).clamp(max_rows, SERVING_PROPOSER_MAX_TAP_ROWS)
                })
        } else {
            max_rows
        };
        let memo_budget = plan_memo_budget(engine.dp_size);
        Ok(Self {
            ranks,
            capacity,
            total_capacity: capacity,
            max_rows,
            mtp_prefill_max_rows,
            memo_budget,
            row_bytes,
            tap_row_bytes,
            k,
            max_k: k,
            tail,
            shifted,
            memos: vec![],
            current: None,
            plain_memos: std::collections::HashMap::new(),
            memo_pinned_staging,
            pending_plain_tap_rows: 0,
            resident_tokens: vec![],
            resident_valid_from: 0,
            processed: 0,
            mtp_batch_active_slot: 0,
            active_slot_last_used_tick: 0,
            batch_slot_tick_counter: 0,
            mtp_batch_slots,
            last_accepted: vec![],
            last_verified_draft_tokens: 0,
            last_restored_tokens: 0,
        })
    }
    pub(crate) fn clear_all_caches(&mut self) -> Result<()> {
        let _ = self.set_mtp_batch_mode(false);
        if let Ranks::Mtp(rs) = &mut self.ranks {
            for r in rs {
                r.reset_batch_slot_count();
            }
        }
        self.abort_scheduled();
        self.memos.clear();
        self.current = None;
        self.plain_memos.clear();
        self.reset()
    }
    pub(crate) fn reset(&mut self) -> Result<()> {
        self.resident_tokens.clear();
        self.resident_valid_from = 0;
        self.processed = 0;
        self.last_accepted.clear();
        self.last_restored_tokens = 0;
        match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs {
                    r.truncate(0)?;
                }
            }
            Ranks::DFlash(rs) => {
                for r in rs {
                    r.truncate(0)?;
                }
            }
        }
        Ok(())
    }
    fn commit_current_memo(&mut self) {
        let Some(memo) = self.current.take() else {
            return;
        };
        if memo.tokens.is_empty() {
            return;
        }
        push_mtp_memo_with_budget(&mut self.memos, memo, self.memo_budget);
    }
    pub(crate) fn can_restore_for_prompt(&self, tokens: &[i32], prompt_len: usize) -> bool {
        if tokens.is_empty() {
            return true;
        }
        match &self.ranks {
            Ranks::DFlash(rs) => {
                let slot_count = rs.first().map_or(0, |r| r.batch_slot_count());
                let ring_pages = rs.first().and_then(|r| r.ring_phys_pages_per_slot());
                let window = rs.first().map_or(0, |r| r.sliding_window());
                if window > 0
                    && ring_pages.is_some()
                    && prompt_len.saturating_sub(tokens.len()) >= window
                {
                    return tokens.len() <= self.capacity;
                }
                let need_bytes = tokens.len().saturating_mul(self.row_bytes);
                dflash_slot_has_prefix_for_prompt(
                    &self.resident_tokens,
                    self.resident_valid_from,
                    tokens,
                    prompt_len,
                    ring_pages,
                    window,
                ) || (rs.first().is_some_and(|r| r.is_batch_mode())
                    && self.mtp_batch_slots.iter().take(slot_count).any(|s| {
                        dflash_slot_has_prefix_for_prompt(
                            &s.resident_tokens,
                            s.resident_valid_from,
                            tokens,
                            prompt_len,
                            ring_pages,
                            window,
                        )
                    }))
                    || (tokens.len() <= self.capacity
                        && self
                            .memos
                            .iter()
                            .chain(self.current.iter())
                            .chain(self.plain_memos.values())
                            .chain(
                                self.mtp_batch_slots
                                    .iter()
                                    .filter_map(|s| s.current.as_ref()),
                            )
                            .any(|m| {
                                m.tokens.starts_with(tokens) && m.features.len() >= need_bytes
                            }))
            }
            Ranks::Mtp(_) => {
                let need_bytes = tokens.len().saturating_mul(self.row_bytes);
                self.memos
                    .iter()
                    .chain(self.current.iter())
                    .chain(self.plain_memos.values())
                    .chain(
                        self.mtp_batch_slots
                            .iter()
                            .filter_map(|s| s.current.as_ref()),
                    )
                    .any(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_bytes)
            }
        }
    }

    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub(crate) fn tap_row_bytes(&self) -> usize {
        self.tap_row_bytes
    }

    pub(crate) fn is_dflash(&self) -> bool {
        matches!(self.ranks, Ranks::DFlash(_))
    }

    /// Projects `rows` raw `[rows, 5*H]` BF16 target taps on rank 0 into `[rows, H]` BF16
    /// `ctx = rms_norm(fc(tap_aux))` on `t0.stream` without mutating DFlash2 KV state.
    pub(crate) fn project_plain_dflash_taps_rank0(
        &mut self,
        t0: &crate::model::Qwen3_5ModelRank,
        tap_bf16: u64,
        rows: usize,
    ) -> Result<u64> {
        let Ranks::DFlash(rs) = &mut self.ranks else {
            return Err(loader_err("expected DFlash2 ranks for tap projection"));
        };
        rs[0].project_taps_to_ctx_no_sync(t0, tap_bf16, rows)
    }

    /// Appends committed `step_plain` DFlash2 slices by projecting `tap_aux_bf16` → `ctx`
    /// in `<= DFLASH2_MAX_BATCH_ROWS` chunks on rank 0 and streaming projected `ctx` bytes D2H.
    pub(crate) fn append_plain_dflash_taps_chunked_d2h(
        &mut self,
        t0: &crate::model::Qwen3_5ModelRank,
        tap_base: u64,
        slices: &[(u64, usize, &[i32])],
    ) -> Result<()> {
        if slices.is_empty() || self.row_bytes == 0 || self.tap_row_bytes == 0 {
            return Ok(());
        }
        let max_chunk = self.max_rows.clamp(1, DFLASH2_MAX_BATCH_ROWS);
        let tap_row_bytes = self.tap_row_bytes;
        for &(req_id, tok_offset, tokens) in slices {
            let mut sub_off = 0usize;
            while sub_off < tokens.len() {
                let take = (tokens.len() - sub_off).min(max_chunk);
                let tap_ptr = tap_base + ((tok_offset + sub_off) * tap_row_bytes) as u64;
                let ctx_ptr = self.project_plain_dflash_taps_rank0(t0, tap_ptr, take)?;
                self.append_plain_slices_chunked_d2h(
                    ctx_ptr,
                    &[(req_id, 0, &tokens[sub_off..sub_off + take])],
                    &t0.stream,
                )?;
                sub_off += take;
            }
        }
        Ok(())
    }

    pub(crate) fn prefill_max_rows(&self) -> usize {
        self.mtp_prefill_max_rows
    }

    /// Initializes a per-request MTP host memo for a request running in `step_plain`,
    /// non-destructively copying any aligned prefix from `self.memos`, active slot memos,
    /// or sibling `plain_memos` so concurrent requests sharing a long harness never steal
    /// from each other.
    pub(crate) fn init_plain_memo(&mut self, req_id: u64, prefix_tokens: &[i32]) {
        if self.row_bytes == 0 || self.plain_memos.contains_key(&req_id) {
            return;
        }
        if prefix_tokens.is_empty() {
            self.plain_memos.insert(
                req_id,
                PrefixMemo {
                    tokens: Vec::new(),
                    features: Vec::new(),
                },
            );
            return;
        }
        let Some(need_bytes) = prefix_tokens.len().checked_mul(self.row_bytes) else {
            return;
        };
        if !memo_fits_budget_limit(
            SpeculativeKind::Mtp,
            0,
            prefix_tokens.len(),
            self.row_bytes,
            self.memo_budget,
        ) {
            return;
        }
        let active_plain_bytes: usize = self
            .plain_memos
            .values()
            .map(|m| m.features.capacity())
            .sum();
        if active_plain_bytes.saturating_add(need_bytes) > self.memo_budget {
            return;
        }
        let donor_src: Option<&[u8]> = self
            .memos
            .iter()
            .rev()
            .chain(self.current.iter())
            .chain(
                self.mtp_batch_slots
                    .iter()
                    .filter_map(|s| s.current.as_ref()),
            )
            .chain(self.plain_memos.values())
            .find(|m| m.tokens.starts_with(prefix_tokens) && m.features.len() >= need_bytes)
            .map(|m| &m.features[..need_bytes]);
        if let Some(src_bytes) = donor_src {
            let mut features = Vec::new();
            grow_memo_features(&mut features, need_bytes, self.row_bytes);
            features.copy_from_slice(src_bytes);
            self.plain_memos.insert(
                req_id,
                PrefixMemo {
                    tokens: prefix_tokens.to_vec(),
                    features,
                },
            );
        }
    }

    pub(crate) fn has_plain_memo(&self, req_id: u64) -> bool {
        self.plain_memos.contains_key(&req_id)
    }

    /// Appends committed `(tokens, raw_hidden_bytes)` rows captured from `step_plain` to a
    /// request's in-flight MTP memo. Drops the memo cleanly if it would exceed `self.memo_budget`.
    pub(crate) fn append_plain_memo_bytes(
        &mut self,
        req_id: u64,
        tokens: &[i32],
        raw_hidden_bytes: &[u8],
    ) {
        if tokens.is_empty() || self.row_bytes == 0 {
            return;
        }
        if raw_hidden_bytes.len() != tokens.len().saturating_mul(self.row_bytes) {
            self.plain_memos.remove(&req_id);
            return;
        }
        let active_plain_bytes: usize = self
            .plain_memos
            .values()
            .map(|m| m.features.capacity())
            .sum();
        let Some(memo) = self.plain_memos.get_mut(&req_id) else {
            return;
        };
        let prev_cap = memo.features.capacity();
        if !memo_fits_budget_limit(
            SpeculativeKind::Mtp,
            memo.tokens.len(),
            tokens.len(),
            self.row_bytes,
            self.memo_budget,
        ) || active_plain_bytes
            .saturating_sub(prev_cap)
            .saturating_add(memo.features.len().saturating_add(raw_hidden_bytes.len()))
            > self.memo_budget
        {
            self.plain_memos.remove(&req_id);
            return;
        }
        let old_len = memo.features.len();
        grow_memo_features(
            &mut memo.features,
            old_len + raw_hidden_bytes.len(),
            self.row_bytes,
        );
        memo.features[old_len..].copy_from_slice(raw_hidden_bytes);
        memo.tokens.extend_from_slice(tokens);
    }

    /// Returns `true` when `total_rows` fits inside `memo_pinned_staging` so an asynchronous
    /// D2H from `last_final_hidden_f16` can overlap with `run_gpu_sampler_rows` without an
    /// extra `stream.synchronize()`.
    pub(crate) fn can_stage_plain_tap_async(&self, total_rows: usize) -> bool {
        let bytes = total_rows.saturating_mul(self.row_bytes);
        bytes > 0
            && self
                .memo_pinned_staging
                .as_ref()
                .is_some_and(|s| s.len_bytes >= bytes)
    }

    /// Enqueues an asynchronous D2H copy of `total_rows` FP16 hidden rows from `dev_src` into
    /// `memo_pinned_staging` on `stream` without synchronizing. Caller must ensure `stream`
    /// synchronizes (e.g. via `run_gpu_sampler_rows` or an explicit `stream.synchronize()`)
    /// before calling [`Self::commit_staged_plain_tap_slices`].
    pub(crate) fn enqueue_plain_tap_hidden_d2h(
        &mut self,
        dev_src: u64,
        total_rows: usize,
        stream: &crate::cuda::CudaStream,
    ) -> Result<()> {
        self.pending_plain_tap_rows = 0;
        let bytes = total_rows.saturating_mul(self.row_bytes);
        let Some(stage) = self.memo_pinned_staging.as_ref() else {
            return Err(loader_err("missing pinned staging buffer"));
        };
        if bytes == 0 || bytes > stage.len_bytes {
            return Err(loader_err("plain tap D2H exceeds pinned staging buffer"));
        }
        check_rc(
            // SAFETY: `stage.host_ptr` is pinned with `stage.len_bytes >= bytes`, and `dev_src`
            // holds `bytes` valid FP16 hidden bytes on `stream`.
            unsafe {
                ffi::warp34_memcpy_d2h_async(
                    stage.host_ptr,
                    dev_src as *const std::ffi::c_void,
                    bytes,
                    stream.handle,
                )
            },
            "plain MTP memo async pinned D2H",
        )?;
        self.pending_plain_tap_rows = total_rows;
        Ok(())
    }

    /// Commits tracked `(req_id, row_offset, tokens)` slices from `memo_pinned_staging` after
    /// `stream` has synchronized, avoiding any temporary `Vec<u8>` heap allocation.
    pub(crate) fn commit_staged_plain_tap_slices(&mut self, slices: &[(u64, usize, &[i32])]) {
        let staged_rows = std::mem::replace(&mut self.pending_plain_tap_rows, 0);
        if staged_rows == 0 || self.row_bytes == 0 {
            return;
        }
        let Some(stage_ptr) = self
            .memo_pinned_staging
            .as_ref()
            .map(|s| s.host_ptr.cast::<u8>() as usize)
        else {
            return;
        };
        let row_bytes = self.row_bytes;
        let max_bytes = staged_rows.saturating_mul(row_bytes);
        for &(req_id, tok_off, tokens) in slices {
            let byte_off = tok_off.saturating_mul(row_bytes);
            let byte_len = tokens.len().saturating_mul(row_bytes);
            if byte_len == 0 || byte_off.saturating_add(byte_len) > max_bytes {
                self.plain_memos.remove(&req_id);
                continue;
            }
            // SAFETY: `stage_ptr` belongs to `self.memo_pinned_staging` (unmodified during
            // `append_plain_memo_bytes`), and `byte_off + byte_len <= max_bytes <= stage.len_bytes`
            // after `stream` synchronized.
            let slice_bytes = unsafe {
                std::slice::from_raw_parts((stage_ptr + byte_off) as *const u8, byte_len)
            };
            self.append_plain_memo_bytes(req_id, tokens, slice_bytes);
        }
    }

    /// Streams tracked `(req_id, row_offset, tokens)` slices directly from `dev_src` into each
    /// request's `plain_memos` buffer in `memo_pinned_staging`-sized DMA chunks, avoiding both
    /// temporary `Vec<u8>` allocations and unpinned pageable `cudaMemcpyAsync` fallbacks on
    /// 2048-row prefill chunks.
    pub(crate) fn append_plain_slices_chunked_d2h(
        &mut self,
        dev_src: u64,
        slices: &[(u64, usize, &[i32])],
        stream: &crate::cuda::CudaStream,
    ) -> Result<()> {
        if self.row_bytes == 0 || slices.is_empty() {
            return Ok(());
        }
        let row_bytes = self.row_bytes;
        let budget = self.memo_budget;
        let stage_info = self
            .memo_pinned_staging
            .as_ref()
            .filter(|s| s.len_bytes >= row_bytes)
            .map(|s| (s.host_ptr.cast::<u8>(), s.len_bytes));
        for &(req_id, tok_off, tokens) in slices {
            if tokens.is_empty() {
                continue;
            }
            let active_plain_bytes: usize = self
                .plain_memos
                .values()
                .map(|m| m.features.capacity())
                .sum();
            let Some(memo) = self.plain_memos.get_mut(&req_id) else {
                continue;
            };
            let copy_bytes = tokens.len().saturating_mul(row_bytes);
            let prev_cap = memo.features.capacity();
            if !memo_fits_budget_limit(
                SpeculativeKind::Mtp,
                memo.tokens.len(),
                tokens.len(),
                row_bytes,
                budget,
            ) || active_plain_bytes
                .saturating_sub(prev_cap)
                .saturating_add(memo.features.len().saturating_add(copy_bytes))
                > budget
            {
                self.plain_memos.remove(&req_id);
                continue;
            }
            let old_len = memo.features.len();
            grow_memo_features(&mut memo.features, old_len + copy_bytes, row_bytes);
            let dev_base = dev_src + (tok_off.saturating_mul(row_bytes)) as u64;
            let copy_res: Result<()> = (|| {
                if let Some((stage_ptr, stage_cap)) = stage_info {
                    let mut off = 0usize;
                    while off < copy_bytes {
                        let take = (copy_bytes - off).min(stage_cap);
                        check_rc(
                            // SAFETY: `stage_ptr` is pinned with `take <= stage_cap`, and
                            // `dev_base + off` spans valid FP16 hidden bytes on `stream`.
                            unsafe {
                                ffi::warp34_memcpy_d2h_async(
                                    stage_ptr.cast(),
                                    (dev_base + off as u64) as *const std::ffi::c_void,
                                    take,
                                    stream.handle,
                                )
                            },
                            "plain MTP memo chunked pinned D2H",
                        )?;
                        stream.synchronize()?;
                        // SAFETY: `stream.synchronize()` completed the D2H into `stage_ptr`,
                        // and `memo.features` was resized to `old_len + copy_bytes`.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                stage_ptr,
                                memo.features.as_mut_ptr().add(old_len + off),
                                take,
                            );
                        }
                        off += take;
                    }
                } else {
                    check_rc(
                        // SAFETY: `memo.features[old_len..]` has `copy_bytes` bytes and we
                        // synchronize `stream` before returning.
                        unsafe {
                            ffi::warp34_memcpy_d2h_async(
                                memo.features.as_mut_ptr().add(old_len).cast(),
                                dev_base as *const std::ffi::c_void,
                                copy_bytes,
                                stream.handle,
                            )
                        },
                        "plain MTP memo fallback D2H",
                    )?;
                    stream.synchronize()?;
                }
                Ok(())
            })();
            if let Err(e) = copy_res {
                self.plain_memos.remove(&req_id);
                return Err(e);
            }
            memo.tokens.extend_from_slice(tokens);
        }
        Ok(())
    }

    /// Commits a finished or cancelled `step_plain` request's memo into `self.memos` so
    /// subsequent turns can immediately resume MTP-3 with 100% cache hit.
    pub(crate) fn finish_plain_memo(&mut self, req_id: u64) {
        if let Some(memo) = self.plain_memos.remove(&req_id) {
            push_mtp_memo_with_budget(&mut self.memos, memo, self.memo_budget);
        }
    }

    /// Discards an errored `step_plain` request's in-flight memo without polluting `self.memos`.
    pub(crate) fn discard_plain_memo(&mut self, req_id: u64) {
        self.plain_memos.remove(&req_id);
    }

    /// Clears all in-flight `step_plain` memos (e.g. on batch abort or worker shutdown).
    pub(crate) fn clear_plain_memos(&mut self) {
        self.pending_plain_tap_rows = 0;
        self.plain_memos.clear();
    }

    /// Exports cached speculative proposer features for `prefix_tokens` (if present in host
    /// memos) so another DP replica can hydrate both KV/GDN and the speculative memo together.
    pub(crate) fn export_shared_memo_features(&self, prefix_tokens: &[i32]) -> Option<Vec<u8>> {
        if prefix_tokens.is_empty() || self.row_bytes == 0 {
            return None;
        }
        let need_bytes = prefix_tokens.len().checked_mul(self.row_bytes)?;
        self.memos
            .iter()
            .chain(self.current.iter())
            .chain(self.plain_memos.values())
            .chain(
                self.mtp_batch_slots
                    .iter()
                    .filter_map(|s| s.current.as_ref()),
            )
            .find(|m| m.tokens.starts_with(prefix_tokens) && m.features.len() >= need_bytes)
            .map(|m| m.features[..need_bytes].to_vec())
    }

    /// Imports a shared speculative proposer prefix memo hydrated from [`crate::dp_prefix_pool::SharedDpPrefixPool`].
    pub(crate) fn import_shared_memo_features(&mut self, prefix_tokens: &[i32], features: &[u8]) {
        if prefix_tokens.is_empty()
            || self.row_bytes == 0
            || features.len() != prefix_tokens.len().saturating_mul(self.row_bytes)
        {
            return;
        }
        if self.can_restore_for_prompt(prefix_tokens, prefix_tokens.len().saturating_add(1)) {
            return;
        }
        push_mtp_memo_with_budget(
            &mut self.memos,
            PrefixMemo {
                tokens: prefix_tokens.to_vec(),
                features: features.to_vec(),
            },
            self.memo_budget,
        );
    }
}

struct StagedMemoAppend {
    prev_lens: Option<(usize, usize)>,
    pending_pinned: Option<(usize, usize)>,
}

impl NativeProposer {
    fn stage_current_memo_append(
        current: &mut Option<PrefixMemo>,
        pinned_staging: Option<&crate::cuda::MappedHostBuffer>,
        t0: &crate::model::Qwen3_5ModelRank,
        feature0_ptr: u64,
        tokens: &[i32],
        row_bytes: usize,
        memo_budget: usize,
    ) -> Result<StagedMemoAppend> {
        if let Some(memo) = current {
            let aligned = memo.features.len() == memo.tokens.len().saturating_mul(row_bytes);
            if !aligned
                || !memo_fits_budget_limit(
                    SpeculativeKind::DFlash2,
                    memo.tokens.len(),
                    tokens.len(),
                    row_bytes,
                    memo_budget,
                )
            {
                *current = None;
            }
        }
        let prev_lens = current.as_ref().map(|m| (m.tokens.len(), m.features.len()));
        let mut pending_pinned = None;
        if let Some(memo) = current {
            if let Err(e) = t0.device.make_current() {
                *current = None;
                return Err(e);
            }
            let len_bytes = tokens.len() * row_bytes;
            let old_len = memo.features.len();
            grow_memo_features(&mut memo.features, old_len + len_bytes, row_bytes);
            let copy_res = if let Some(pinned) = pinned_staging
                && len_bytes > 0
                && len_bytes <= pinned.len_bytes
            {
                let r = check_rc(
                    // SAFETY: `feature0_ptr` spans `len_bytes` on `t0.stream`, and `pinned`
                    // remains live until `t0.stream` synchronizes before `commit_staged_memo_append`.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            pinned.host_ptr,
                            feature0_ptr as *const _,
                            len_bytes,
                            t0.stream.handle,
                        )
                    },
                    "proposer pinned memo copy",
                );
                if r.is_ok() {
                    pending_pinned = Some((old_len, len_bytes));
                }
                r
            } else {
                check_rc(
                    // SAFETY: `feature0_ptr` spans `len_bytes` on `t0.stream`, which
                    // synchronizes before returning to caller.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            memo.features[old_len..].as_mut_ptr().cast(),
                            feature0_ptr as *const _,
                            len_bytes,
                            t0.stream.handle,
                        )
                    },
                    "proposer memo copy",
                )
            };
            if let Err(e) = copy_res {
                *current = None;
                return Err(e);
            }
            memo.tokens.extend_from_slice(tokens);
        }
        Ok(StagedMemoAppend {
            prev_lens,
            pending_pinned,
        })
    }

    fn commit_staged_memo_append(
        current: &mut Option<PrefixMemo>,
        pinned_staging: Option<&crate::cuda::MappedHostBuffer>,
        pending_pinned: Option<(usize, usize)>,
    ) {
        if let Some((old_len, len_bytes)) = pending_pinned
            && let (Some(memo), Some(pinned)) = (current, pinned_staging)
        {
            // SAFETY: caller synchronized `engine.ranks[0].stream`, completing the D2H into `pinned`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pinned.host_ptr.cast::<u8>(),
                    memo.features.as_mut_ptr().add(old_len),
                    len_bytes,
                );
            }
        }
    }

    fn rollback_staged_memo_append(
        current: &mut Option<PrefixMemo>,
        prev_lens: Option<(usize, usize)>,
    ) {
        if let (Some(memo), Some((pt, pf))) = (current, prev_lens) {
            memo.tokens.truncate(pt);
            memo.features.truncate(pf);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_dflash_memo_window(
        rs: &mut [DFlash2Rank],
        engine: &Qwen3_5Engine,
        features: &[u8],
        start_tok: usize,
        n: usize,
        row_bytes: usize,
        max_rows: usize,
    ) -> Result<()> {
        for r in rs.iter_mut() {
            r.truncate(0)?;
            r.set_committed(start_tok)?;
        }
        let step = max_rows.clamp(1, DFLASH2_MAX_BATCH_ROWS);
        let pool = engine.tp_workers.clone();
        for c0 in (start_tok..n).step_by(step) {
            let c1 = (c0 + step).min(n);
            let chunk_rows = c1 - c0;
            let bytes = &features[c0 * row_bytes..c1 * row_bytes];
            let mut ptrs = Vec::with_capacity(engine.tp_size);
            for t in &engine.ranks {
                let stage = t
                    .tap_aux_bf16
                    .as_ref()
                    .ok_or_else(|| loader_err("missing target feature capture"))?;
                stage.copy_from_host(bytes, &t.stream)?;
                ptrs.push(stage.addr());
            }
            crate::coordinated::run_tp_closures(
                pool.as_deref(),
                rs.iter_mut()
                    .zip(&engine.ranks)
                    .zip(&ptrs)
                    .map(|((r, t), &ptr)| move || r.ingest_ctx(t, ptr, chunk_rows)),
            )
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        }
        Ok(())
    }

    fn clone_aligned_memo_prefix(
        memos: &mut Vec<PrefixMemo>,
        tokens: &[i32],
        need_bytes: usize,
        row_bytes: usize,
        memo_budget: usize,
    ) -> PrefixMemo {
        let n = tokens.len();
        if let Some(idx) = memos
            .iter()
            .rposition(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_bytes)
        {
            let mut memo = memos.remove(idx);
            if n > 4096 && memo.tokens.len().saturating_sub(n) < 64 {
                memo.tokens.truncate(n);
                memo.features.truncate(need_bytes);
                return memo;
            }
            let mut f = Vec::new();
            grow_memo_features(&mut f, need_bytes, row_bytes);
            f.copy_from_slice(&memo.features[..need_bytes]);
            push_mtp_memo_with_budget(memos, memo, memo_budget);
            PrefixMemo {
                tokens: tokens.to_vec(),
                features: f,
            }
        } else {
            PrefixMemo {
                tokens: tokens.to_vec(),
                features: vec![],
            }
        }
    }

    fn restore_dflash(&mut self, engine: &mut Qwen3_5Engine, tokens: &[i32]) -> Result<()> {
        let Ranks::DFlash(rs) = &mut self.ranks else {
            return Err(loader_err("expected DFlash2 ranks"));
        };
        let n = tokens.len();
        let need_dflash_bytes = n.saturating_mul(self.row_bytes);
        if !self
            .memos
            .iter()
            .any(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_dflash_bytes)
            && let Some(donor_memo) = self
                .mtp_batch_slots
                .iter()
                .filter_map(|s| s.current.as_ref())
                .chain(self.plain_memos.values())
                .find(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_dflash_bytes)
        {
            let mut f = Vec::new();
            grow_memo_features(&mut f, need_dflash_bytes, self.row_bytes);
            f.copy_from_slice(&donor_memo.features[..need_dflash_bytes]);
            push_mtp_memo_with_budget(
                &mut self.memos,
                PrefixMemo {
                    tokens: tokens.to_vec(),
                    features: f,
                },
                self.memo_budget,
            );
        }
        let ring_pages = rs.first().and_then(|r| r.ring_phys_pages_per_slot());
        let window = rs.first().map_or(0, |r| r.sliding_window());
        let self_eff = self.resident_tokens.starts_with(tokens).then(|| {
            dflash_effective_valid_from(
                self.resident_tokens.len(),
                self.resident_valid_from,
                ring_pages,
            )
        });
        let slot_count = rs.first().map_or(0, |r| r.batch_slot_count());
        let best_donor = rs
            .first()
            .is_some_and(|r| r.is_batch_mode())
            .then(|| {
                (0..slot_count)
                    .filter(|&s| {
                        s != self.mtp_batch_active_slot
                            && self.mtp_batch_slots[s].resident_tokens.starts_with(tokens)
                    })
                    .map(|s| {
                        let eff = dflash_effective_valid_from(
                            self.mtp_batch_slots[s].resident_tokens.len(),
                            self.mtp_batch_slots[s].resident_valid_from,
                            ring_pages,
                        );
                        (s, eff)
                    })
                    .min_by_key(|&(_, eff)| eff)
            })
            .flatten();
        if let Some((donor, donor_eff)) = best_donor
            && donor_eff < n
            && self_eff.is_none_or(|se| donor_eff < se)
        {
            let dst = self.mtp_batch_active_slot;
            for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                r.copy_kv_from_slot(t, donor, dst, n)?;
            }
            self.resident_tokens.clear();
            self.resident_tokens.extend_from_slice(tokens);
            self.resident_valid_from = donor_eff;
            self.current = Some(Self::clone_aligned_memo_prefix(
                &mut self.memos,
                tokens,
                need_dflash_bytes,
                self.row_bytes,
                self.memo_budget,
            ));
        } else if let Some(se) = self_eff
            && se < n
        {
            for r in rs {
                r.truncate(n)?;
            }
            self.resident_tokens.truncate(n);
            self.resident_valid_from = se;
            self.current = Some(Self::clone_aligned_memo_prefix(
                &mut self.memos,
                tokens,
                need_dflash_bytes,
                self.row_bytes,
                self.memo_budget,
            ));
        } else if let Some(idx) = self
            .memos
            .iter()
            .rposition(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_dflash_bytes)
        {
            let mut memo = self.memos.remove(idx);
            let start_tok = if window > 0 && ring_pages.is_some() {
                (n.saturating_sub(window) / 16) * 16
            } else {
                0
            };
            if let Err(e) = Self::replay_dflash_memo_window(
                rs,
                engine,
                &memo.features,
                start_tok,
                n,
                self.row_bytes,
                self.max_rows,
            ) {
                push_mtp_memo_with_budget(&mut self.memos, memo, self.memo_budget);
                let _ = self.reset();
                return Err(e);
            }
            self.resident_tokens.clear();
            self.resident_tokens.extend_from_slice(tokens);
            self.resident_valid_from = start_tok;
            if n > 4096 && memo.tokens.len().saturating_sub(n) < 64 {
                memo.tokens.truncate(n);
                memo.features.truncate(need_dflash_bytes);
                self.current = Some(memo);
            } else {
                let mut f = Vec::new();
                grow_memo_features(&mut f, need_dflash_bytes, self.row_bytes);
                f.copy_from_slice(&memo.features[..need_dflash_bytes]);
                push_mtp_memo_with_budget(&mut self.memos, memo, self.memo_budget);
                self.current = Some(PrefixMemo {
                    tokens: tokens.to_vec(),
                    features: f,
                });
            }
        } else if window > 0 && ring_pages.is_some() {
            for r in rs {
                r.set_committed(n)?;
            }
            self.resident_tokens.clear();
            self.resident_tokens.extend_from_slice(tokens);
            self.resident_valid_from = n;
            self.current = Some(PrefixMemo {
                tokens: tokens.to_vec(),
                features: vec![],
            });
        } else {
            return Err(loader_err("missing DFlash2 resident prefix"));
        }
        self.processed = n;
        self.last_accepted.clear();
        self.last_restored_tokens = n;
        Ok(())
    }

    fn restore(&mut self, engine: &mut Qwen3_5Engine, tokens: &[i32]) -> Result<()> {
        self.commit_current_memo();
        if tokens.is_empty() {
            self.reset()?;
            self.current = Some(PrefixMemo {
                tokens: vec![],
                features: vec![],
            });
            return Ok(());
        }
        if matches!(self.ranks, Ranks::DFlash(_)) {
            return self.restore_dflash(engine, tokens);
        }
        let need_mtp_bytes = matches!(self.ranks, Ranks::Mtp(_))
            .then(|| tokens.len().saturating_mul(self.row_bytes))
            .unwrap_or(0);
        // If another MTP batch slot already holds `tokens` in its resident GPU KV cache,
        // clone its `(n - 1)` KV tokens via D2D copy so we hit the O(1) GPU fast path below.
        if !self.resident_tokens.starts_with(tokens)
            && let Ranks::Mtp(rs) = &mut self.ranks
            && rs.first().is_some_and(|r| r.is_batch_mode())
        {
            let slot_count = rs[0].batch_slot_count();
            if let Some(donor) = (0..slot_count).find(|&s| {
                s != self.mtp_batch_active_slot
                    && self.mtp_batch_slots[s].resident_tokens.starts_with(tokens)
            }) {
                let dst = self.mtp_batch_active_slot;
                let n = tokens.len();
                for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                    r.copy_kv_from_slot(t, donor, dst, n.saturating_sub(1))?;
                }
                self.resident_tokens.clear();
                self.resident_tokens.extend_from_slice(tokens);
            }
        }
        // If the matching memo is currently held in another active slot's `current` or an
        // in-flight `step_plain` memo, clone its `n`-token prefix into `self.memos` so
        // `restore()` is symmetric with `can_restore_for_prompt()`.
        if !self
            .memos
            .iter()
            .any(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_mtp_bytes)
            && let Some(donor_memo) = self
                .mtp_batch_slots
                .iter()
                .filter_map(|s| s.current.as_ref())
                .chain(self.plain_memos.values())
                .find(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_mtp_bytes)
        {
            let n = tokens.len();
            let mut f = Vec::new();
            grow_memo_features(&mut f, n * self.row_bytes, self.row_bytes);
            f.copy_from_slice(&donor_memo.features[..n * self.row_bytes]);
            push_mtp_memo_with_budget(
                &mut self.memos,
                PrefixMemo {
                    tokens: tokens.to_vec(),
                    features: f,
                },
                self.memo_budget,
            );
        }
        // Fast path: if the drafter's resident GPU KV cache already starts with `tokens`,
        // truncate the drafter KV cache in-place and restore MTP's 1-row tail without
        // re-running drafter prefill over the cached prefix.
        if self.resident_tokens.starts_with(tokens)
            && let Some(idx) = self
                .memos
                .iter()
                .rposition(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_mtp_bytes)
        {
            let mut memo = self.memos.remove(idx);
            let n = tokens.len();
            match &mut self.ranks {
                Ranks::DFlash(rs) => {
                    for r in rs {
                        r.truncate(n)?;
                    }
                }
                Ranks::Mtp(rs) => {
                    let tail_bytes = &memo.features[(n - 1) * self.row_bytes..n * self.row_bytes];
                    for ((r, t), tail) in rs.iter_mut().zip(&engine.ranks).zip(&self.tail) {
                        r.truncate(n - 1)?;
                        tail.copy_from_host(tail_bytes, &t.stream)?;
                    }
                }
            }
            self.resident_tokens.truncate(n);
            self.processed = n;
            self.last_accepted.clear();
            self.last_restored_tokens = n;
            // For long session-specific prefixes (> 4096 tokens) where `memo` is this turn's
            // own Mamba-chunk frontier (`memo.tokens.len() - n < 64`), move `memo` directly into
            // `self.current` in O(1) without cloning or page-faulting a new buffer. Shared
            // harnesses (`<= 4096`) and further-ahead sibling sessions (`>= 64` suffix) keep
            // `memo` intact in `self.memos` so no session's Turn-1 suffix is truncated.
            if n > 4096 && memo.tokens.len().saturating_sub(n) < 64 {
                memo.tokens.truncate(n);
                if matches!(self.ranks, Ranks::Mtp(_)) {
                    memo.features.truncate(n * self.row_bytes);
                } else {
                    memo.features.clear();
                }
                self.current = Some(memo);
            } else {
                let features = if matches!(self.ranks, Ranks::Mtp(_))
                    && memo.features.len() >= n * self.row_bytes
                {
                    let mut f = Vec::new();
                    grow_memo_features(&mut f, n * self.row_bytes, self.row_bytes);
                    f.copy_from_slice(&memo.features[..n * self.row_bytes]);
                    f
                } else {
                    vec![]
                };
                push_mtp_memo_with_budget(&mut self.memos, memo, self.memo_budget);
                self.current = Some(PrefixMemo {
                    tokens: tokens.to_vec(),
                    features,
                });
            }
            return Ok(());
        }
        let idx = self
            .memos
            .iter()
            .rposition(|m| m.tokens.starts_with(tokens) && m.features.len() >= need_mtp_bytes)
            .ok_or_else(|| loader_err("missing proposer prefix"))?;
        let mut memo = self.memos.remove(idx);
        let start_tok = match self.prepare_mtp_partial_lcp_restore(engine, tokens, &memo.features) {
            Ok(reused) => reused,
            Err(e) => {
                push_mtp_memo_with_budget(&mut self.memos, memo, self.memo_budget);
                return Err(e);
            }
        };
        let step = self.max_rows.min(MTP_STAGE_ROWS);
        let result = (|| {
            for start in (start_tok..tokens.len()).step_by(step) {
                let end = (start + step).min(tokens.len());
                let bytes = &memo.features[start * self.row_bytes..end * self.row_bytes];
                let mut pointers = Vec::with_capacity(engine.tp_size);
                for t in &engine.ranks {
                    let stage = t
                        .last_final_hidden_f16
                        .as_ref()
                        .ok_or_else(|| loader_err("missing target feature capture"))?;
                    stage.copy_from_host(bytes, &t.stream)?;
                    pointers.push(stage.addr());
                }
                self.ingest(engine, &tokens[start..end], &pointers)?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = self.reset();
        }
        if result.is_ok()
            && tokens.len() > 4096
            && memo.tokens.len().saturating_sub(tokens.len()) < 64
        {
            memo.tokens.truncate(tokens.len());
            if matches!(self.ranks, Ranks::Mtp(_)) {
                memo.features.truncate(tokens.len() * self.row_bytes);
            } else {
                memo.features.clear();
            }
            self.current = Some(memo);
        } else {
            let restored_features = if result.is_ok() {
                let prefix_bytes = tokens.len() * self.row_bytes;
                let mut f = Vec::new();
                grow_memo_features(&mut f, prefix_bytes, self.row_bytes);
                f.copy_from_slice(&memo.features[..prefix_bytes]);
                Some(f)
            } else {
                None
            };
            push_mtp_memo_with_budget(&mut self.memos, memo, self.memo_budget);
            if let Some(features) = restored_features {
                self.current = Some(PrefixMemo {
                    tokens: tokens.to_vec(),
                    features,
                });
            }
        }
        result?;
        self.last_restored_tokens = tokens.len();
        Ok(())
    }

    fn prepare_mtp_partial_lcp_restore(
        &mut self,
        engine: &Qwen3_5Engine,
        tokens: &[i32],
        memo_features: &[u8],
    ) -> Result<usize> {
        if let Ranks::Mtp(rs) = &mut self.ranks {
            let self_lcp = self
                .resident_tokens
                .iter()
                .zip(tokens)
                .take_while(|(a, b)| a == b)
                .count();
            if rs.first().is_some_and(|r| r.is_batch_mode()) {
                let slot_count = rs[0].batch_slot_count();
                let best_donor = (0..slot_count)
                    .filter(|&s| s != self.mtp_batch_active_slot)
                    .map(|s| {
                        let lcp = self.mtp_batch_slots[s]
                            .resident_tokens
                            .iter()
                            .zip(tokens)
                            .take_while(|(a, b)| a == b)
                            .count();
                        (s, lcp)
                    })
                    .max_by_key(|&(_, lcp)| lcp);
                if let Some((donor, donor_lcp)) = best_donor
                    && donor_lcp >= 16
                    && donor_lcp >= self_lcp.saturating_add(256)
                {
                    let dst = self.mtp_batch_active_slot;
                    for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                        r.copy_kv_from_slot(t, donor, dst, donor_lcp - 1)?;
                    }
                    self.resident_tokens.clear();
                    self.resident_tokens.extend_from_slice(&tokens[..donor_lcp]);
                }
            }
            let reuse_lcp = self
                .resident_tokens
                .iter()
                .zip(tokens)
                .take_while(|(a, b)| a == b)
                .count();
            if reuse_lcp >= 16 {
                let tail_bytes =
                    &memo_features[(reuse_lcp - 1) * self.row_bytes..reuse_lcp * self.row_bytes];
                for ((r, t), tail) in rs.iter_mut().zip(&engine.ranks).zip(&self.tail) {
                    r.truncate(reuse_lcp - 1)?;
                    tail.copy_from_host(tail_bytes, &t.stream)?;
                }
                self.resident_tokens.truncate(reuse_lcp);
                self.processed = reuse_lcp;
                self.last_accepted.clear();
                self.current = None;
                return Ok(reuse_lcp);
            }
        }
        self.reset()?;
        self.current = None;
        Ok(0)
    }
    fn captured(&self, engine: &Qwen3_5Engine) -> Result<Vec<u64>> {
        engine
            .ranks
            .iter()
            .map(|t| {
                match &self.ranks {
                    Ranks::Mtp(_) => t.last_final_hidden_f16.as_ref(),
                    Ranks::DFlash(_) => t.tap_aux_bf16.as_ref(),
                }
                .map(DeviceBuffer::addr)
                .ok_or_else(|| loader_err("missing target feature capture"))
            })
            .collect()
    }
    pub(crate) fn captured_for_rows(
        &self,
        engine: &Qwen3_5Engine,
        rows: usize,
    ) -> Result<Vec<u64>> {
        engine
            .ranks
            .iter()
            .map(|t| {
                match &self.ranks {
                    Ranks::Mtp(_) => {
                        if rows <= t.max_tap_rows {
                            t.last_final_hidden_f16.as_ref().map(DeviceBuffer::addr)
                        } else if t.last_mtp_hidden_ptr != 0 && t.last_mtp_hidden_rows >= rows {
                            Some(t.last_mtp_hidden_ptr)
                        } else {
                            None
                        }
                    }
                    Ranks::DFlash(_) => (rows <= t.max_tap_rows)
                        .then(|| t.tap_aux_bf16.as_ref().map(DeviceBuffer::addr))
                        .flatten(),
                }
                .ok_or_else(|| loader_err("missing target feature capture"))
            })
            .collect()
    }
    pub(crate) fn ingest_for_prompt_len(
        &mut self,
        engine: &mut Qwen3_5Engine,
        tokens: &[i32],
        features: &[u64],
        prompt_len: usize,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let dflash_win = match &self.ranks {
            Ranks::DFlash(rs) => rs
                .first()
                .and_then(|r| r.ring_phys_pages_per_slot().map(|_| r.sliding_window()))
                .filter(|&w| w > 0),
            Ranks::Mtp(_) => None,
        };
        if let Some(window) = dflash_win {
            let win_keep_tok = (prompt_len.saturating_sub(window) / 16) * 16;
            let chunk_start = self.processed;
            if chunk_start < win_keep_tok {
                if features.len() != engine.tp_size
                    || tokens.len() > self.max_rows
                    || self.processed + tokens.len() > self.capacity
                {
                    return Err(loader_err("invalid committed proposer span"));
                }
                let chunk_end = chunk_start + tokens.len();
                let skip_end = chunk_end.min(win_keep_tok);
                let skip_rows = skip_end - chunk_start;
                if let Some(memo) = &mut self.current {
                    let aligned =
                        memo.features.len() == memo.tokens.len().saturating_mul(self.row_bytes);
                    let within_cap = memo_fits_budget_limit(
                        SpeculativeKind::DFlash2,
                        memo.tokens.len(),
                        skip_rows,
                        self.row_bytes,
                        self.memo_budget,
                    );
                    if aligned && within_cap {
                        let t0 = &engine.ranks[0];
                        let step = self.max_rows.clamp(1, DFLASH2_MAX_BATCH_ROWS);
                        let row_bytes = self.row_bytes;
                        let tap_row_bytes = self.tap_row_bytes;
                        let copy_res: Result<()> = (|| {
                            let Ranks::DFlash(rs) = &mut self.ranks else {
                                return Err(loader_err("expected DFlash2 ranks"));
                            };
                            t0.device.make_current()?;
                            let len_bytes = skip_rows * row_bytes;
                            let old_len = memo.features.len();
                            grow_memo_features(&mut memo.features, old_len + len_bytes, row_bytes);
                            let mut off_rows = 0usize;
                            while off_rows < skip_rows {
                                let take = (skip_rows - off_rows).min(step);
                                let tap_ptr = features[0] + (off_rows * tap_row_bytes) as u64;
                                let ctx_ptr =
                                    rs[0].project_taps_to_ctx_no_sync(t0, tap_ptr, take)?;
                                let dst_off = old_len + off_rows * row_bytes;
                                check_rc(
                                    // SAFETY: `ctx_ptr` spans `take * row_bytes` on `t0.stream`,
                                    // synchronized immediately below before `scratch.reset()`.
                                    unsafe {
                                        ffi::warp34_memcpy_d2h_async(
                                            memo.features[dst_off..].as_mut_ptr().cast(),
                                            ctx_ptr as *const _,
                                            take * row_bytes,
                                            t0.stream.handle,
                                        )
                                    },
                                    "DFlash prefill projected ctx memo copy",
                                )?;
                                t0.stream.synchronize()?;
                                off_rows += take;
                            }
                            Ok(())
                        })();
                        if let Err(e) = copy_res {
                            self.current = None;
                            return Err(e);
                        }
                        memo.tokens.extend_from_slice(&tokens[..skip_rows]);
                    } else {
                        self.current = None;
                    }
                }
                let prev_processed = self.processed;
                let prev_valid_from = self.resident_valid_from;
                let prev_res_len = self.resident_tokens.len();
                if let Ranks::DFlash(rs) = &mut self.ranks
                    && let Err(e) = rs.iter_mut().try_for_each(|r| r.set_committed(skip_end))
                {
                    for r in rs.iter_mut() {
                        let _ = r.set_committed(prev_processed);
                    }
                    self.current = None;
                    return Err(e);
                }
                self.resident_tokens.extend_from_slice(&tokens[..skip_rows]);
                self.processed = skip_end;
                self.resident_valid_from = self.resident_valid_from.max(skip_end);
                if skip_rows == tokens.len() {
                    return Ok(());
                }
                let tail_features: Vec<u64> = features
                    .iter()
                    .map(|&p| p + (skip_rows * self.tap_row_bytes) as u64)
                    .collect();
                if let Err(e) = self.ingest(engine, &tokens[skip_rows..], &tail_features) {
                    if let Ranks::DFlash(rs) = &mut self.ranks {
                        for r in rs.iter_mut() {
                            let _ = r.set_committed(prev_processed);
                        }
                    }
                    self.resident_tokens.truncate(prev_res_len);
                    self.processed = prev_processed;
                    self.resident_valid_from = prev_valid_from;
                    self.current = None;
                    return Err(e);
                }
                return Ok(());
            }
        }
        self.ingest(engine, tokens, features)
    }

    fn ingest(
        &mut self,
        engine: &mut Qwen3_5Engine,
        tokens: &[i32],
        features: &[u64],
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let stage_cap = match self.ranks {
            Ranks::DFlash(_) => self.max_rows.min(DFLASH2_MAX_BATCH_ROWS),
            Ranks::Mtp(_) => self.max_rows.min(MTP_STAGE_ROWS),
        };
        if tokens.len() > stage_cap {
            let mut offset = 0;
            while offset < tokens.len() {
                let take = (tokens.len() - offset).min(stage_cap);
                let sub_features: Vec<u64> = features
                    .iter()
                    .map(|&p| p + (offset * self.tap_row_bytes) as u64)
                    .collect();
                self.ingest(engine, &tokens[offset..offset + take], &sub_features)?;
                offset += take;
            }
            return Ok(());
        }
        if features.len() != engine.tp_size
            || tokens.len() > self.max_rows
            || self.processed + tokens.len() > self.capacity
        {
            return Err(loader_err("invalid committed proposer span"));
        }
        // Both MTP and DFlash2 record aligned host feature memos up to `memo_budget` so
        // multi-prefix workloads (`prefix-primed`, multi-session agents) never reject a warm
        // target `MambaRadixCache` hit when the drafter's active GPU slot switched prefixes.
        if let Some(memo) = &self.current {
            let kind = match self.ranks {
                Ranks::DFlash(_) => SpeculativeKind::DFlash2,
                Ranks::Mtp(_) => SpeculativeKind::Mtp,
            };
            let aligned = memo.features.len() == memo.tokens.len().saturating_mul(self.row_bytes);
            if !aligned
                || !memo_fits_budget_limit(
                    kind,
                    memo.tokens.len(),
                    tokens.len(),
                    self.row_bytes,
                    self.memo_budget,
                )
            {
                self.current = None;
            }
        }
        let mut pending_pinned_memo_copy: Option<(usize, usize)> = None;
        if matches!(self.ranks, Ranks::Mtp(_))
            && let Some(memo) = &mut self.current
        {
            let t0 = &engine.ranks[0];
            if let Err(e) = t0.device.make_current() {
                self.current = None;
                return Err(e);
            }
            let len_bytes = tokens.len() * self.row_bytes;
            let old_len = memo.features.len();
            grow_memo_features(&mut memo.features, old_len + len_bytes, self.row_bytes);
            let copy_res = if let Some(pinned) = &self.memo_pinned_staging
                && len_bytes > 0
                && len_bytes <= pinned.len_bytes
            {
                let r = check_rc(
                    // SAFETY: `features[0]` spans `len_bytes` on `t0.stream`, and `pinned`
                    // remains live until `t0.stream` synchronizes at the end of `ingest`.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            pinned.host_ptr,
                            features[0] as *const _,
                            len_bytes,
                            t0.stream.handle,
                        )
                    },
                    "proposer pinned memo copy",
                );
                if r.is_ok() {
                    pending_pinned_memo_copy = Some((old_len, len_bytes));
                }
                r
            } else {
                check_rc(
                    // SAFETY: features[0] spans rows*row_bytes on rank 0's stream, and
                    // rank 0's stream is synchronized before ingest returns below.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            memo.features[old_len..].as_mut_ptr().cast(),
                            features[0] as *const _,
                            len_bytes,
                            t0.stream.handle,
                        )
                    },
                    "proposer memo copy",
                )
            };
            if let Err(e) = copy_res {
                self.current = None;
                return Err(e);
            }
            memo.tokens.extend_from_slice(tokens);
        }
        let pool = engine.tp_workers.clone();
        let ingest_res: Result<()> = match &mut self.ranks {
            Ranks::DFlash(rs) => {
                let n = tokens.len();
                let res = crate::coordinated::run_tp_closures(
                    pool.as_deref(),
                    rs.iter_mut()
                        .zip(&engine.ranks)
                        .zip(features)
                        .map(|((r, t), &ptr)| move || r.ingest_no_sync(t, ptr, n)),
                )
                .into_iter()
                .collect::<Result<Vec<_>>>();
                if let Err(e) = res {
                    Err(e)
                } else {
                    if let Some(memo) = &mut self.current {
                        let t0 = &engine.ranks[0];
                        let ctx_ptr = rs[0].last_ctx_ptr();
                        let len_bytes = n * self.row_bytes;
                        let old_len = memo.features.len();
                        grow_memo_features(&mut memo.features, old_len + len_bytes, self.row_bytes);
                        let copy_res = t0.device.make_current().and_then(|()| {
                            if let Some(pinned) = &self.memo_pinned_staging
                                && len_bytes > 0
                                && len_bytes <= pinned.len_bytes
                            {
                                let r = check_rc(
                                    // SAFETY: `ctx_ptr` spans `len_bytes` on `t0.stream`,
                                    // synchronized immediately below before `scratch` reuse.
                                    unsafe {
                                        ffi::warp34_memcpy_d2h_async(
                                            pinned.host_ptr,
                                            ctx_ptr as *const _,
                                            len_bytes,
                                            t0.stream.handle,
                                        )
                                    },
                                    "DFlash pinned ctx memo copy",
                                );
                                if r.is_ok() {
                                    pending_pinned_memo_copy = Some((old_len, len_bytes));
                                }
                                r
                            } else {
                                check_rc(
                                    // SAFETY: `ctx_ptr` spans `len_bytes` on `t0.stream`,
                                    // synchronized immediately below.
                                    unsafe {
                                        ffi::warp34_memcpy_d2h_async(
                                            memo.features[old_len..].as_mut_ptr().cast(),
                                            ctx_ptr as *const _,
                                            len_bytes,
                                            t0.stream.handle,
                                        )
                                    },
                                    "DFlash ctx memo copy",
                                )
                            }
                        });
                        if let Err(e) = copy_res {
                            self.current = None;
                            return Err(e);
                        }
                        memo.tokens.extend_from_slice(tokens);
                    }
                    engine.ranks.iter().try_for_each(|t| {
                        t.device
                            .make_current()
                            .and_then(|()| t.stream.synchronize())
                    })
                }
            }
            Ranks::Mtp(rs) => {
                let row_bytes = self.row_bytes;
                let before = self.processed;
                let comm = engine.tp2_comm.as_deref();
                crate::coordinated::run_tp_closures(
                    pool.as_deref(),
                    rs.iter_mut()
                        .zip(&mut engine.ranks)
                        .zip(&self.tail)
                        .zip(&self.shifted)
                        .zip(features)
                        .map(|((((r, t), tail), shifted), &ptr)| {
                            move || -> Result<()> {
                                t.device.make_current()?;
                                r.truncate(before.saturating_sub(1))?;
                                let (ids, input) = if before == 0 {
                                    (&tokens[1..], ptr)
                                } else {
                                    check_rc(
                                        // SAFETY: both resident buffers hold at least one MTP hidden row.
                                        unsafe {
                                            ffi::warp34_memcpy_d2d_async(
                                                shifted.addr() as *mut _,
                                                tail.addr() as *const _,
                                                row_bytes,
                                                t.stream.handle,
                                            )
                                        },
                                        "MTP preceding hidden",
                                    )?;
                                    if tokens.len() > 1 {
                                        check_rc(
                                            // SAFETY: rows<=max_rows and the final shifted row fits;
                                            // source is the owned live capture for this whole batch.
                                            unsafe {
                                                ffi::warp34_memcpy_d2d_async(
                                                    (shifted.addr() + row_bytes as u64) as *mut _,
                                                    ptr as *const _,
                                                    (tokens.len() - 1) * row_bytes,
                                                    t.stream.handle,
                                                )
                                            },
                                            "MTP shifted hidden",
                                        )?;
                                    }
                                    (tokens, shifted.addr())
                                };
                                r.ingest_kv_only_no_sync(t, ids, input, comm)?;
                                check_rc(
                                    // SAFETY: tokens is nonempty and ptr spans every captured row;
                                    // tail owns exactly one MTP hidden row and survives synchronization.
                                    unsafe {
                                        ffi::warp34_memcpy_d2d_async(
                                            tail.addr() as *mut _,
                                            (ptr + ((tokens.len() - 1) * row_bytes) as u64)
                                                as *const _,
                                            row_bytes,
                                            t.stream.handle,
                                        )
                                    },
                                    "MTP tail hidden",
                                )?;
                                t.stream.synchronize()
                            }
                        }),
                )
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .map(|_| ())
            }
        };
        if let Err(e) = ingest_res {
            self.current = None;
            let _ = engine.ranks[0].device.make_current();
            let _ = engine.ranks[0].stream.synchronize();
            return Err(e);
        }
        if let Some((old_len, len_bytes)) = pending_pinned_memo_copy
            && let (Some(memo), Some(pinned)) = (&mut self.current, &self.memo_pinned_staging)
        {
            // SAFETY: `Mtp::ingest` synchronized `engine.ranks[0].stream`, completing the D2H
            // into `pinned.host_ptr` (`len_bytes <= pinned.len_bytes`), and `memo.features`
            // was grown to `old_len + len_bytes`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    pinned.host_ptr.cast::<u8>(),
                    memo.features.as_mut_ptr().add(old_len),
                    len_bytes,
                );
            }
        }
        self.resident_tokens.extend_from_slice(tokens);
        self.processed += tokens.len();
        Ok(())
    }
    fn ingest_and_propose(
        &mut self,
        engine: &mut Qwen3_5Engine,
        tokens: &[i32],
        features: &[u64],
        anchor: i32,
        k: usize,
    ) -> Result<Vec<i32>> {
        if tokens.is_empty() {
            return self.propose(engine, anchor, k);
        }
        if k == 0 {
            self.ingest(engine, tokens, features)?;
            return Ok(vec![]);
        }
        if features.len() != engine.tp_size
            || tokens.len() > self.max_rows
            || self.processed + tokens.len() > self.capacity
        {
            return Err(loader_err("invalid committed proposer span"));
        }
        match &mut self.ranks {
            Ranks::DFlash(rs) => {
                if tokens.len() > self.max_rows.min(DFLASH2_MAX_BATCH_ROWS) {
                    self.ingest(engine, tokens, features)?;
                    return self.propose(engine, anchor, k);
                }
                let n = tokens.len();
                let use_graph = engine.use_cuda_graph;
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph;
                }
                if rs.len() == 1 {
                    rs[0].ingest_no_sync(&engine.ranks[0], features[0], n)?;
                    let staged_memo = Self::stage_current_memo_append(
                        &mut self.current,
                        self.memo_pinned_staging.as_ref(),
                        &engine.ranks[0],
                        rs[0].last_ctx_ptr(),
                        tokens,
                        self.row_bytes,
                        self.memo_budget,
                    )?;
                    let (memo_prev_lens, pending_pinned_memo_copy) =
                        (staged_memo.prev_lens, staged_memo.pending_pinned);
                    let res = rs[0].forward_select_greedy(&engine.ranks[0], anchor);
                    let mut ids = match res {
                        Ok(ids) => ids,
                        Err(e) => {
                            let _ = engine.ranks[0].device.make_current();
                            let _ = engine.ranks[0].stream.synchronize();
                            Self::rollback_staged_memo_append(&mut self.current, memo_prev_lens);
                            return Err(e);
                        }
                    };
                    Self::commit_staged_memo_append(
                        &mut self.current,
                        self.memo_pinned_staging.as_ref(),
                        pending_pinned_memo_copy,
                    );
                    self.resident_tokens.extend_from_slice(tokens);
                    self.processed += tokens.len();
                    ids.truncate(k);
                    return Ok(ids);
                }
                let comm = engine.tp2_comm.as_deref();
                let dflash_graph =
                    use_graph && rs.iter().all(DFlash2Rank::can_capture_select_greedy);
                let mut memo_prev_lens = None;
                let mut pending_pinned_memo_copy = None;
                let outputs_res: Result<Vec<Vec<i32>>> = if dflash_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let current = &mut self.current;
                    let pinned = self.memo_pinned_staging.as_ref();
                    let row_bytes = self.row_bytes;
                    let memo_budget = self.memo_budget;
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        let ptr = features[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.ingest_no_sync(t, ptr, n)?;
                                if rank == 0 {
                                    let staged = Self::stage_current_memo_append(
                                        current,
                                        pinned,
                                        t,
                                        r.last_ctx_ptr(),
                                        tokens,
                                        row_bytes,
                                        memo_budget,
                                    )?;
                                    memo_prev_lens = staged.prev_lens;
                                    pending_pinned_memo_copy = staged.pending_pinned;
                                }
                                r.prepare_select_greedy_graph_inputs_async(t, anchor)
                            }
                            ReplayPhase::Capture => r.ensure_select_greedy_graph(t, comm),
                            ReplayPhase::Launch => r.replay_select_greedy_graph_async(t),
                        }
                    });
                    match replay_res {
                        Ok(()) => rs
                            .iter_mut()
                            .zip(&engine.ranks)
                            .map(|(r, t)| r.collect_select_greedy_host(t, anchor))
                            .collect::<Result<Vec<_>>>(),
                        Err(error) => {
                            if error.partial_launch {
                                eprintln!(
                                    "[warp34] fatal: asymmetric DFlash TP graph launch failure: {}",
                                    error.source
                                );
                                std::process::abort();
                            }
                            Err(error.source)
                        }
                    }
                } else {
                    let pool = engine.tp_workers.clone();
                    let ingest_res = crate::coordinated::run_tp_closures(
                        pool.as_deref(),
                        rs.iter_mut()
                            .zip(&engine.ranks)
                            .zip(features)
                            .map(|((r, t), &ptr)| move || r.ingest_no_sync(t, ptr, n)),
                    )
                    .into_iter()
                    .collect::<Result<Vec<_>>>();
                    match ingest_res {
                        Err(e) => Err(e),
                        Ok(_) => {
                            let staged = Self::stage_current_memo_append(
                                &mut self.current,
                                self.memo_pinned_staging.as_ref(),
                                &engine.ranks[0],
                                rs[0].last_ctx_ptr(),
                                tokens,
                                self.row_bytes,
                                self.memo_budget,
                            )?;
                            memo_prev_lens = staged.prev_lens;
                            pending_pinned_memo_copy = staged.pending_pinned;
                            crate::coordinated::run_tp_closures(
                                pool.as_deref(),
                                rs.iter_mut().zip(&mut engine.ranks).map(|(r, t)| {
                                    move || r.forward_select_greedy_tp(t, anchor, comm)
                                }),
                            )
                            .into_iter()
                            .collect::<Result<Vec<_>>>()
                        }
                    }
                };
                let mut outputs = match outputs_res {
                    Ok(o) => o,
                    Err(e) => {
                        for t in &engine.ranks {
                            let _ = t.device.make_current();
                            let _ = t.stream.synchronize();
                        }
                        Self::rollback_staged_memo_append(&mut self.current, memo_prev_lens);
                        return Err(e);
                    }
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    Self::rollback_staged_memo_append(&mut self.current, memo_prev_lens);
                    return Err(loader_err("DFlash TP token disagreement"));
                }
                Self::commit_staged_memo_append(
                    &mut self.current,
                    self.memo_pinned_staging.as_ref(),
                    pending_pinned_memo_copy,
                );
                self.resident_tokens.extend_from_slice(tokens);
                self.processed += tokens.len();
                let mut ids = outputs.swap_remove(0);
                ids.truncate(k);
                Ok(ids)
            }
            Ranks::Mtp(rs) => {
                if tokens.len() > self.max_rows.min(MTP_STAGE_ROWS) {
                    self.ingest(engine, tokens, features)?;
                    return self.propose(engine, anchor, k);
                }
                if let Some(memo) = &self.current
                    && (memo.tokens.len() + tokens.len()) * self.row_bytes > MEMO_BUDGET
                {
                    self.current = None;
                }
                let memo_prev_lens = self
                    .current
                    .as_ref()
                    .map(|m| (m.tokens.len(), m.features.len()));
                let mut pending_pinned_memo_copy: Option<(usize, usize)> = None;
                if let Some(memo) = &mut self.current {
                    let t0 = &engine.ranks[0];
                    t0.device.make_current()?;
                    let len_bytes = tokens.len() * self.row_bytes;
                    let old_len = memo.features.len();
                    grow_memo_features(&mut memo.features, old_len + len_bytes, self.row_bytes);
                    if let Some(pinned) = &self.memo_pinned_staging
                        && len_bytes > 0
                        && len_bytes <= pinned.len_bytes
                    {
                        check_rc(
                            // SAFETY: `features[0]` spans `len_bytes` on `t0.stream`, and `pinned`
                            // remains live until `t0.stream` synchronizes during MTP draft readback.
                            unsafe {
                                ffi::warp34_memcpy_d2h_async(
                                    pinned.host_ptr,
                                    features[0] as *const _,
                                    len_bytes,
                                    t0.stream.handle,
                                )
                            },
                            "proposer pinned memo copy",
                        )?;
                        pending_pinned_memo_copy = Some((old_len, len_bytes));
                    } else {
                        check_rc(
                            // SAFETY: features[0] spans rows*row_bytes on rank 0's stream, and
                            // rank 0's stream is synchronized before ingest_and_propose returns below.
                            unsafe {
                                ffi::warp34_memcpy_d2h_async(
                                    memo.features[old_len..].as_mut_ptr().cast(),
                                    features[0] as *const _,
                                    len_bytes,
                                    t0.stream.handle,
                                )
                            },
                            "proposer memo copy",
                        )?;
                    }
                    memo.tokens.extend_from_slice(tokens);
                }
                let use_graph = engine.use_cuda_graph;
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                }
                let mtp_graph = use_graph && rs.iter().all(Qwen3_5MtpRank::supports_cuda_graph);
                let row_bytes = self.row_bytes;
                let before = self.processed;
                let comm = engine.tp2_comm.as_deref();
                let outputs_res: Result<Vec<Vec<i32>>> = if mtp_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        let tail = &self.tail[rank];
                        let shifted = &self.shifted[rank];
                        let ptr = features[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.truncate(before.saturating_sub(1))?;
                                let (ids, input) = if before == 0 {
                                    (&tokens[1..], ptr)
                                } else {
                                    check_rc(
                                        // SAFETY: both resident buffers hold at least one MTP hidden row.
                                        unsafe {
                                            ffi::warp34_memcpy_d2d_async(
                                                shifted.addr() as *mut _,
                                                tail.addr() as *const _,
                                                row_bytes,
                                                t.stream.handle,
                                            )
                                        },
                                        "MTP preceding hidden",
                                    )?;
                                    if tokens.len() > 1 {
                                        check_rc(
                                            // SAFETY: rows<=max_rows and the final shifted row fits;
                                            // source is the owned live capture for this whole batch.
                                            unsafe {
                                                ffi::warp34_memcpy_d2d_async(
                                                    (shifted.addr() + row_bytes as u64) as *mut _,
                                                    ptr as *const _,
                                                    (tokens.len() - 1) * row_bytes,
                                                    t.stream.handle,
                                                )
                                            },
                                            "MTP shifted hidden",
                                        )?;
                                    }
                                    (tokens, shifted.addr())
                                };
                                r.ingest_kv_only_no_sync(t, ids, input, comm)?;
                                check_rc(
                                    // SAFETY: tokens is nonempty and ptr spans every captured row;
                                    // tail owns exactly one MTP hidden row on this device.
                                    unsafe {
                                        ffi::warp34_memcpy_d2d_async(
                                            tail.addr() as *mut _,
                                            (ptr + ((tokens.len() - 1) * row_bytes) as u64)
                                                as *const _,
                                            row_bytes,
                                            t.stream.handle,
                                        )
                                    },
                                    "MTP tail hidden",
                                )?;
                                r.prepare_propose_graph_inputs_async(t, anchor, tail.addr(), k)
                            }
                            ReplayPhase::Capture => r.ensure_propose_graph(t, k, comm),
                            ReplayPhase::Launch => r.replay_propose_graph_async(t, k),
                        }
                    });
                    match replay_res {
                        Ok(()) => rs
                            .iter_mut()
                            .zip(&engine.ranks)
                            .map(|(r, t)| r.collect_propose_host(t, k))
                            .collect::<Result<Vec<_>>>(),
                        Err(error) => {
                            if error.partial_launch {
                                eprintln!(
                                    "[warp34] fatal: asymmetric MTP TP graph launch failure: {}",
                                    error.source
                                );
                                std::process::abort();
                            }
                            Err(error.source)
                        }
                    }
                } else {
                    let pool = engine.tp_workers.clone();
                    crate::coordinated::run_tp_closures(
                        pool.as_deref(),
                        rs.iter_mut()
                            .zip(&mut engine.ranks)
                            .zip(&self.tail)
                            .zip(&self.shifted)
                            .zip(features)
                            .map(|((((r, t), tail), shifted), &ptr)| {
                                move || -> Result<Vec<i32>> {
                                    t.device.make_current()?;
                                    r.truncate(before.saturating_sub(1))?;
                                    let (ids, input) = if before == 0 {
                                        (&tokens[1..], ptr)
                                    } else {
                                        check_rc(
                                            // SAFETY: both resident buffers hold at least one MTP hidden row.
                                            unsafe {
                                                ffi::warp34_memcpy_d2d_async(
                                                    shifted.addr() as *mut _,
                                                    tail.addr() as *const _,
                                                    row_bytes,
                                                    t.stream.handle,
                                                )
                                            },
                                            "MTP preceding hidden",
                                        )?;
                                        if tokens.len() > 1 {
                                            check_rc(
                                                // SAFETY: rows<=max_rows and the final shifted row fits;
                                                // source is the owned live capture for this whole batch.
                                                unsafe {
                                                    ffi::warp34_memcpy_d2d_async(
                                                        (shifted.addr() + row_bytes as u64)
                                                            as *mut _,
                                                        ptr as *const _,
                                                        (tokens.len() - 1) * row_bytes,
                                                        t.stream.handle,
                                                    )
                                                },
                                                "MTP shifted hidden",
                                            )?;
                                        }
                                        (tokens, shifted.addr())
                                    };
                                    r.ingest_kv_only_no_sync(t, ids, input, comm)?;
                                    check_rc(
                                        // SAFETY: tokens is nonempty and ptr spans every captured row;
                                        // tail owns exactly one MTP hidden row on this device.
                                        unsafe {
                                            ffi::warp34_memcpy_d2d_async(
                                                tail.addr() as *mut _,
                                                (ptr + ((tokens.len() - 1) * row_bytes) as u64)
                                                    as *const _,
                                                row_bytes,
                                                t.stream.handle,
                                            )
                                        },
                                        "MTP tail hidden",
                                    )?;
                                    r.propose(t, anchor, tail.addr(), k, comm)
                                }
                            }),
                    )
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
                };
                let outputs = match outputs_res {
                    Ok(o) => o,
                    Err(e) => {
                        for (r, t) in rs.iter_mut().zip(&engine.ranks) {
                            let _ = t.device.make_current();
                            let _ = t.stream.synchronize();
                            let _ = r.truncate(before.saturating_sub(1));
                        }
                        if let Some((prev_tok_len, prev_feat_len)) = memo_prev_lens
                            && let Some(memo) = &mut self.current
                        {
                            memo.tokens.truncate(prev_tok_len);
                            memo.features.truncate(prev_feat_len);
                        }
                        return Err(e);
                    }
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("MTP TP token disagreement"));
                }
                if let Some((old_len, len_bytes)) = pending_pinned_memo_copy
                    && let (Some(memo), Some(pinned)) =
                        (&mut self.current, &self.memo_pinned_staging)
                {
                    // SAFETY: MTP draft readback synchronized `engine.ranks[0].stream`, completing
                    // the D2H into `pinned.host_ptr`, and `memo.features` spans `old_len + len_bytes`.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            pinned.host_ptr.cast::<u8>(),
                            memo.features.as_mut_ptr().add(old_len),
                            len_bytes,
                        );
                    }
                }
                self.resident_tokens.extend_from_slice(tokens);
                self.processed += tokens.len();
                Ok(outputs.into_iter().next().unwrap())
            }
        }
    }

    fn propose(&mut self, engine: &mut Qwen3_5Engine, anchor: i32, k: usize) -> Result<Vec<i32>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let use_graph = engine.use_cuda_graph;
        let comm = engine.tp2_comm.as_deref();
        match &mut self.ranks {
            Ranks::Mtp(rs) => {
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph && r.supports_cuda_graph();
                }
                let mtp_graph = use_graph && rs.iter().all(|r| r.supports_cuda_graph());
                let outputs = if mtp_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        let tail = &self.tail[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.prepare_propose_graph_inputs_async(t, anchor, tail.addr(), k)
                            }
                            ReplayPhase::Capture => r.ensure_propose_graph(t, k, comm),
                            ReplayPhase::Launch => r.replay_propose_graph_async(t, k),
                        }
                    });
                    if let Err(error) = replay_res {
                        if error.partial_launch {
                            eprintln!(
                                "[warp34] fatal: asymmetric MTP TP graph launch failure: {}",
                                error.source
                            );
                            std::process::abort();
                        }
                        return Err(error.source);
                    }
                    rs.iter_mut()
                        .zip(&engine.ranks)
                        .map(|(r, t)| r.collect_propose_host(t, k))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    let pool = engine.tp_workers.clone();
                    crate::coordinated::run_tp_closures(
                        pool.as_deref(),
                        rs.iter_mut().zip(&mut engine.ranks).zip(&self.tail).map(
                            |((r, t), tail)| move || r.propose(t, anchor, tail.addr(), k, comm),
                        ),
                    )
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("MTP TP token disagreement"));
                }
                Ok(outputs[0].clone())
            }
            Ranks::DFlash(rs) => {
                for r in rs.iter_mut() {
                    r.use_cuda_graph = use_graph;
                }
                if rs.len() == 1 {
                    let mut ids = rs[0].forward_select_greedy(&engine.ranks[0], anchor)?;
                    ids.truncate(k);
                    return Ok(ids);
                }
                let dflash_graph =
                    use_graph && rs.iter().all(DFlash2Rank::can_capture_select_greedy);
                let mut outputs = if dflash_graph {
                    use crate::coordinated::{ReplayPhase, replay_ranks};
                    let replay_res = replay_ranks(rs.len(), |phase, rank| {
                        let r = &mut rs[rank];
                        let t = &engine.ranks[rank];
                        t.device.make_current()?;
                        match phase {
                            ReplayPhase::Upload => {
                                r.prepare_select_greedy_graph_inputs_async(t, anchor)
                            }
                            ReplayPhase::Capture => r.ensure_select_greedy_graph(t, comm),
                            ReplayPhase::Launch => r.replay_select_greedy_graph_async(t),
                        }
                    });
                    if let Err(error) = replay_res {
                        if error.partial_launch {
                            eprintln!(
                                "[warp34] fatal: asymmetric DFlash TP graph launch failure: {}",
                                error.source
                            );
                            std::process::abort();
                        }
                        return Err(error.source);
                    }
                    rs.iter_mut()
                        .zip(&engine.ranks)
                        .map(|(r, t)| r.collect_select_greedy_host(t, anchor))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    let pool = engine.tp_workers.clone();
                    crate::coordinated::run_tp_closures(
                        pool.as_deref(),
                        rs.iter_mut()
                            .zip(&mut engine.ranks)
                            .map(|(r, t)| move || r.forward_select_greedy_tp(t, anchor, comm)),
                    )
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
                };
                if outputs.windows(2).any(|p| p[0] != p[1]) {
                    return Err(loader_err("DFlash TP token disagreement"));
                }
                let mut ids = outputs.swap_remove(0);
                ids.truncate(k);
                Ok(ids)
            }
        }
    }

    #[inline]
    fn slot_processed(&self, slot: usize) -> usize {
        if slot == self.mtp_batch_active_slot {
            self.processed
        } else {
            self.mtp_batch_slots[slot].processed
        }
    }

    #[inline]
    fn slot_tail_addr(&self, slot: usize, rank_idx: usize) -> u64 {
        if slot == self.mtp_batch_active_slot {
            self.tail[rank_idx].addr()
        } else {
            self.mtp_batch_slots[slot].tail[rank_idx].addr()
        }
    }

    fn stage_batched_mtp_memo_d2h(
        &mut self,
        t0: &crate::model::Qwen3_5ModelRank,
        slots: &[usize],
        pending_ingests: &[&[i32]],
    ) -> Result<Vec<Option<(usize, usize, usize)>>> {
        let mut staged = vec![None; slots.len()];
        let mut pinned_cursor = 0usize;
        let pinned_cap = self.memo_pinned_staging.as_ref().map_or(0, |p| p.len_bytes);
        let pinned_ptr = self
            .memo_pinned_staging
            .as_ref()
            .map_or(core::ptr::null_mut(), |p| p.host_ptr);
        for (b, (&slot, &toks)) in slots.iter().zip(pending_ingests).enumerate() {
            if toks.is_empty() {
                continue;
            }
            let src = self.mtp_batch_slots[slot].captured_verify_hidden[0].addr();
            let memo_opt = if slot == self.mtp_batch_active_slot {
                &mut self.current
            } else {
                &mut self.mtp_batch_slots[slot].current
            };
            let Some(memo) = memo_opt else {
                continue;
            };
            t0.device.make_current()?;
            let len_bytes = toks.len() * self.row_bytes;
            let old_len = memo.features.len();
            grow_memo_features(&mut memo.features, old_len + len_bytes, self.row_bytes);
            if !pinned_ptr.is_null() && pinned_cursor + len_bytes <= pinned_cap {
                check_rc(
                    // SAFETY: src spans len_bytes on t0.stream; pinned_ptr + pinned_cursor stays
                    // live until collect_batched_propose_host synchronizes t0.stream.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            pinned_ptr.cast::<u8>().add(pinned_cursor).cast(),
                            src as *const _,
                            len_bytes,
                            t0.stream.handle,
                        )
                    },
                    "batched MTP pinned memo copy",
                )?;
                staged[b] = Some((pinned_cursor, old_len, len_bytes));
                pinned_cursor += len_bytes;
            } else {
                check_rc(
                    // SAFETY: memo.features[old_len..] spans len_bytes and stays live until sync.
                    unsafe {
                        ffi::warp34_memcpy_d2h_async(
                            memo.features[old_len..].as_mut_ptr().cast(),
                            src as *const _,
                            len_bytes,
                            t0.stream.handle,
                        )
                    },
                    "batched MTP memo copy",
                )?;
            }
        }
        Ok(staged)
    }

    fn finish_batched_mtp_slot_commits(
        &mut self,
        slots: &[usize],
        pending_ingests: &[&[i32]],
        staged_memos: &[Option<(usize, usize, usize)>],
    ) {
        let pinned_ptr = self
            .memo_pinned_staging
            .as_ref()
            .map_or(core::ptr::null(), |p| p.host_ptr.cast::<u8>());
        for (b, (&slot, &toks)) in slots.iter().zip(pending_ingests).enumerate() {
            if !toks.is_empty() {
                let memo_opt = if slot == self.mtp_batch_active_slot {
                    &mut self.current
                } else {
                    &mut self.mtp_batch_slots[slot].current
                };
                if let Some(memo) = memo_opt {
                    if let Some((pinned_off, old_len, len_bytes)) = staged_memos[b]
                        && !pinned_ptr.is_null()
                    {
                        // SAFETY: t0.stream synchronized during draft readback; both ranges valid.
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                pinned_ptr.add(pinned_off),
                                memo.features.as_mut_ptr().add(old_len),
                                len_bytes,
                            );
                        }
                    }
                    memo.tokens.extend_from_slice(toks);
                }
                if slot == self.mtp_batch_active_slot {
                    self.resident_tokens.extend_from_slice(toks);
                    self.processed += toks.len();
                } else {
                    self.mtp_batch_slots[slot]
                        .resident_tokens
                        .extend_from_slice(toks);
                    self.mtp_batch_slots[slot].processed += toks.len();
                }
            }
            self.batch_slot_tick_counter += 1;
            if slot == self.mtp_batch_active_slot {
                self.active_slot_last_used_tick = self.batch_slot_tick_counter;
            } else {
                self.mtp_batch_slots[slot].last_used_tick = self.batch_slot_tick_counter;
            }
        }
    }

    fn stage_dflash_slot_ctx_d2h(
        &mut self,
        t0: &crate::model::Qwen3_5ModelRank,
        slot: usize,
        toks: &[i32],
        ctx_ptr: u64,
        pinned_cursor: &mut usize,
    ) -> Result<Option<(usize, usize, usize)>> {
        if toks.is_empty() || self.row_bytes == 0 {
            return Ok(None);
        }
        let memo_opt = if slot == self.mtp_batch_active_slot {
            &mut self.current
        } else {
            &mut self.mtp_batch_slots[slot].current
        };
        let Some(memo) = memo_opt.as_mut() else {
            return Ok(None);
        };
        let aligned = memo.features.len() == memo.tokens.len().saturating_mul(self.row_bytes);
        if !aligned
            || !memo_fits_budget_limit(
                SpeculativeKind::DFlash2,
                memo.tokens.len(),
                toks.len(),
                self.row_bytes,
                self.memo_budget,
            )
        {
            *memo_opt = None;
            return Ok(None);
        }
        let len_bytes = toks.len() * self.row_bytes;
        let old_len = memo.features.len();
        grow_memo_features(&mut memo.features, old_len + len_bytes, self.row_bytes);
        if let Some(pinned) = &self.memo_pinned_staging
            && len_bytes > 0
            && pinned_cursor.saturating_add(len_bytes) <= pinned.len_bytes
        {
            let cur = *pinned_cursor;
            check_rc(
                // SAFETY: ctx_ptr spans len_bytes on t0.stream, synchronized before commit.
                unsafe {
                    ffi::warp34_memcpy_d2h_async(
                        pinned.host_ptr.cast::<u8>().add(cur).cast(),
                        ctx_ptr as *const _,
                        len_bytes,
                        t0.stream.handle,
                    )
                },
                "batched DFlash pinned ctx memo copy",
            )?;
            *pinned_cursor = cur + len_bytes;
            Ok(Some((cur, old_len, len_bytes)))
        } else {
            check_rc(
                // SAFETY: memo.features[old_len..] spans len_bytes and stays live until sync.
                unsafe {
                    ffi::warp34_memcpy_d2h_async(
                        memo.features[old_len..].as_mut_ptr().cast(),
                        ctx_ptr as *const _,
                        len_bytes,
                        t0.stream.handle,
                    )
                },
                "batched DFlash ctx memo copy",
            )?;
            Ok(None)
        }
    }

    fn try_ingest_and_propose_dflash_batch(
        &mut self,
        engine: &mut Qwen3_5Engine,
        slots: &[usize],
        pending_ingests: &[&[i32]],
        anchors: &[i32],
        k_budgets: &[usize],
    ) -> Result<Option<Vec<Vec<i32>>>> {
        let b_reqs = slots.len();
        if !(2..=crate::dflash::MAX_DFLASH_DRAFT_BATCH).contains(&b_reqs)
            || pending_ingests.len() != b_reqs
            || anchors.len() != b_reqs
            || k_budgets.len() != b_reqs
            || k_budgets.iter().all(|&k| k == 0)
        {
            return Ok(None);
        }
        let max_ingest_per_slot = self.max_rows.min(DFLASH2_MAX_BATCH_ROWS);
        let can_batch = {
            let Ranks::DFlash(rs) = &self.ranks else {
                return Ok(None);
            };
            let slot_count = rs[0].batch_slot_count().min(self.mtp_batch_slots.len());
            let slot_cap = rs[0].batch_slot_capacity();
            let mut seen = 0u16;
            rs.iter()
                .zip(&engine.ranks)
                .all(|(r, t)| r.can_batch_select_greedy(t, b_reqs))
                && slots.iter().zip(pending_ingests).all(|(&s, &toks)| {
                    let ok = s < slot_count
                        && s < 16
                        && (seen & (1u16 << s)) == 0
                        && toks.len() <= max_ingest_per_slot
                        && self.slot_processed(s) + toks.len() <= slot_cap;
                    if s < 16 {
                        seen |= 1u16 << s;
                    }
                    ok
                })
        };
        if !can_batch {
            return Ok(None);
        }

        let prev_active_slot = self.mtp_batch_active_slot;
        let prev_slot_committed: Vec<usize> =
            slots.iter().map(|&s| self.slot_processed(s)).collect();
        let num_ranks = engine.ranks.len();
        let captured_ptrs: Vec<Vec<u64>> = (0..num_ranks)
            .map(|ri| {
                slots
                    .iter()
                    .map(|&s| self.mtp_batch_slots[s].captured_verify_hidden[ri].addr())
                    .collect()
            })
            .collect();
        let prev_memo_lens: Vec<Option<(usize, usize)>> = slots
            .iter()
            .map(|&s| {
                let m = if s == prev_active_slot {
                    self.current.as_ref()
                } else {
                    self.mtp_batch_slots[s].current.as_ref()
                };
                m.map(|memo| (memo.tokens.len(), memo.features.len()))
            })
            .collect();
        let rollback_batch = |this: &mut Self, engine: &Qwen3_5Engine, sync_streams: bool| {
            if sync_streams {
                for t in &engine.ranks {
                    let _ = t.device.make_current();
                    let _ = t.stream.synchronize();
                }
            }
            if let Ranks::DFlash(rs) = &mut this.ranks {
                for r in rs.iter_mut() {
                    for (&slot, &prev_committed) in slots.iter().zip(&prev_slot_committed) {
                        if r.select_batch_slot(slot).is_ok() {
                            let _ = r.set_committed(prev_committed);
                        }
                    }
                    let _ = r.select_batch_slot(prev_active_slot);
                }
            }
            for (&slot, prev) in slots.iter().zip(&prev_memo_lens) {
                let memo_opt = if slot == prev_active_slot {
                    &mut this.current
                } else {
                    &mut this.mtp_batch_slots[slot].current
                };
                if let (Some(memo), Some((tl, fl))) = (memo_opt.as_mut(), *prev) {
                    memo.tokens.truncate(tl);
                    memo.features.truncate(fl);
                } else {
                    *memo_opt = None;
                }
            }
        };
        let mut staged_memos: Vec<Option<(usize, usize, usize)>> = vec![None; b_reqs];
        let mut pinned_cursor = 0usize;

        // Enqueue all slot ingests asynchronously on every rank (and stage rank 0 projected ctx D2H).
        let ingest_res: Result<()> = (|| {
            for rank in 0..num_ranks {
                let t = &engine.ranks[rank];
                t.device.make_current()?;
                for (b, (&slot, &toks)) in slots.iter().zip(pending_ingests).enumerate() {
                    if toks.is_empty() {
                        continue;
                    }
                    let ctx_ptr = {
                        let Ranks::DFlash(rs) = &mut self.ranks else {
                            unreachable!()
                        };
                        let r = &mut rs[rank];
                        r.select_batch_slot(slot)?;
                        r.ingest_no_sync(t, captured_ptrs[rank][b], toks.len())?;
                        r.last_ctx_ptr()
                    };
                    if rank == 0 {
                        staged_memos[b] = self.stage_dflash_slot_ctx_d2h(
                            t,
                            slot,
                            toks,
                            ctx_ptr,
                            &mut pinned_cursor,
                        )?;
                    }
                }
                let Ranks::DFlash(rs) = &mut self.ranks else {
                    unreachable!()
                };
                rs[rank].select_batch_slot(prev_active_slot)?;
            }
            Ok(())
        })();
        if let Err(e) = ingest_res {
            rollback_batch(self, engine, true);
            return Err(e);
        }

        let comm = engine.tp2_comm.as_deref();
        let use_graph = engine.use_cuda_graph;
        let outputs_res: Result<Vec<Vec<Vec<i32>>>> = {
            let Ranks::DFlash(rs) = &mut self.ranks else {
                unreachable!()
            };
            for r in rs.iter_mut() {
                r.use_cuda_graph = use_graph;
            }
            if use_graph {
                use crate::coordinated::{ReplayPhase, replay_ranks};
                let replay_res = replay_ranks(rs.len(), |phase, rank| {
                    let r = &mut rs[rank];
                    let t = &engine.ranks[rank];
                    t.device.make_current()?;
                    match phase {
                        ReplayPhase::Upload => {
                            r.prepare_batched_select_greedy_inputs_async(t, slots, anchors)
                        }
                        ReplayPhase::Capture => {
                            r.ensure_batched_select_greedy_graph(t, b_reqs, comm)
                        }
                        ReplayPhase::Launch => {
                            r.replay_batched_select_greedy_graph_async(t, b_reqs)
                        }
                    }
                });
                match replay_res {
                    Ok(()) => rs
                        .iter_mut()
                        .zip(&engine.ranks)
                        .map(|(r, t)| r.collect_batched_select_greedy_host(t, b_reqs))
                        .collect::<Result<Vec<_>>>(),
                    Err(error) => {
                        if error.partial_launch {
                            eprintln!(
                                "[warp34] fatal: asymmetric DFlash batched TP graph launch failure: {}",
                                error.source
                            );
                            std::process::abort();
                        }
                        Err(error.source)
                    }
                }
            } else {
                let pool = engine.tp_workers.clone();
                crate::coordinated::run_tp_closures(
                    pool.as_deref(),
                    rs.iter_mut().zip(&engine.ranks).map(|(r, t)| {
                        move || -> Result<Vec<Vec<i32>>> {
                            t.device.make_current()?;
                            r.prepare_batched_select_greedy_inputs_async(t, slots, anchors)?;
                            r.forward_batched_select_greedy_eager_async(t, b_reqs, comm)?;
                            r.collect_batched_select_greedy_host(t, b_reqs)
                        }
                    }),
                )
                .into_iter()
                .collect::<Result<Vec<_>>>()
            }
        };

        let mut outputs = match outputs_res {
            Ok(o) => o,
            Err(e) => {
                rollback_batch(self, engine, true);
                return Err(e);
            }
        };
        if outputs.windows(2).any(|p| p[0] != p[1]) {
            rollback_batch(self, engine, false);
            return Err(loader_err("DFlash batched TP token disagreement"));
        }

        self.finish_batched_mtp_slot_commits(slots, pending_ingests, &staged_memos);
        let mut drafts = outputs.swap_remove(0);
        for (d, &kb) in drafts.iter_mut().zip(k_budgets) {
            d.truncate(kb);
        }
        Ok(Some(drafts))
    }

    pub(crate) fn try_ingest_and_propose_mtp_batch(
        &mut self,
        engine: &mut Qwen3_5Engine,
        slots: &[usize],
        pending_ingests: &[&[i32]],
        anchors: &[i32],
        k_budgets: &[usize],
    ) -> Result<Option<Vec<Vec<i32>>>> {
        if matches!(self.ranks, Ranks::DFlash(_)) {
            return self.try_ingest_and_propose_dflash_batch(
                engine,
                slots,
                pending_ingests,
                anchors,
                k_budgets,
            );
        }
        let b_reqs = slots.len();
        let k_max = k_budgets.iter().copied().max().unwrap_or(0);
        if !engine.use_cuda_graph
            || !(2..=crate::mtp::MAX_MTP_BATCH_SLOTS).contains(&b_reqs)
            || !(1..=3).contains(&k_max)
            || pending_ingests.len() != b_reqs
            || anchors.len() != b_reqs
            || k_budgets.len() != b_reqs
        {
            return Ok(None);
        }
        let total_ingest: usize = pending_ingests.iter().map(|t| t.len()).sum();
        if total_ingest > self.max_rows {
            return Ok(None);
        }
        let can_batch = match &self.ranks {
            Ranks::Mtp(rs) => {
                let r0 = &rs[0];
                let slot_count = r0.batch_slot_count().min(self.mtp_batch_slots.len());
                let slot_cap = r0.batch_slot_capacity();
                let mut seen = 0u16;
                r0.is_batch_mode()
                    && r0.supports_cuda_graph()
                    && slots.iter().zip(pending_ingests).all(|(&s, &toks)| {
                        let ok = s < slot_count
                            && s < 16
                            && (seen & (1u16 << s)) == 0
                            && self.slot_processed(s) >= 1
                            && self.slot_processed(s) - 1 + toks.len() + k_max <= slot_cap;
                        if s < 16 {
                            seen |= 1u16 << s;
                        }
                        ok
                    })
            }
            Ranks::DFlash(_) => false,
        };
        if !can_batch {
            return Ok(None);
        }

        let staged_memos =
            self.stage_batched_mtp_memo_d2h(&engine.ranks[0], slots, pending_ingests)?;
        let num_ranks = engine.ranks.len();
        let tail_ptrs_per_rank: Vec<Vec<u64>> = (0..num_ranks)
            .map(|ri| slots.iter().map(|&s| self.slot_tail_addr(s, ri)).collect())
            .collect();
        let befores: Vec<usize> = slots.iter().map(|&s| self.slot_processed(s)).collect();

        let drafts_res: Result<Vec<Vec<i32>>> = (|| {
            let Ranks::Mtp(rs) = &mut self.ranks else {
                unreachable!()
            };
            for (&slot, &before) in slots.iter().zip(&befores) {
                for r in rs.iter_mut() {
                    r.truncate_slot(slot, before - 1)?;
                }
            }
            if total_ingest > 0 {
                let slices: Vec<crate::mtp::MtpIngestSlice<'_>> = slots
                    .iter()
                    .zip(pending_ingests)
                    .map(|(&slot, &tokens)| crate::mtp::MtpIngestSlice { slot, tokens })
                    .collect();
                for (ri, ((r, t), shifted)) in rs
                    .iter_mut()
                    .zip(&engine.ranks)
                    .zip(&self.shifted)
                    .enumerate()
                {
                    t.device.make_current()?;
                    let mut row_cursor = 0usize;
                    for (b, (&slot, &toks)) in slots.iter().zip(pending_ingests).enumerate() {
                        if toks.is_empty() {
                            continue;
                        }
                        let src = self.mtp_batch_slots[slot].captured_verify_hidden[ri].addr();
                        let tail_ptr = tail_ptrs_per_rank[ri][b];
                        let dst = shifted.addr() + (row_cursor * self.row_bytes) as u64;
                        check_rc(
                            // SAFETY: dst and tail_ptr span row_bytes on t.stream.
                            unsafe {
                                ffi::warp34_memcpy_d2d_async(
                                    dst as *mut _,
                                    tail_ptr as *const _,
                                    self.row_bytes,
                                    t.stream.handle,
                                )
                            },
                            "batched MTP shifted head",
                        )?;
                        if toks.len() > 1 {
                            check_rc(
                                // SAFETY: src and dst + row_bytes span (toks.len() - 1) * row_bytes.
                                unsafe {
                                    ffi::warp34_memcpy_d2d_async(
                                        (dst + self.row_bytes as u64) as *mut _,
                                        src as *const _,
                                        (toks.len() - 1) * self.row_bytes,
                                        t.stream.handle,
                                    )
                                },
                                "batched MTP shifted body",
                            )?;
                        }
                        check_rc(
                            // SAFETY: src tail row copied into tail_ptr after reading old tail_ptr above.
                            unsafe {
                                ffi::warp34_memcpy_d2d_async(
                                    tail_ptr as *mut _,
                                    (src + ((toks.len() - 1) * self.row_bytes) as u64) as *const _,
                                    self.row_bytes,
                                    t.stream.handle,
                                )
                            },
                            "batched MTP update tail",
                        )?;
                        row_cursor += toks.len();
                    }
                    r.ingest_kv_only_multi_slot_no_sync(t, &slices, shifted.addr())?;
                }
            }

            let comm = engine.tp2_comm.as_deref();
            let mut long_ctx_per_rank = [false; 2];
            use crate::coordinated::{ReplayPhase, replay_ranks};
            let replay_res = replay_ranks(rs.len(), |phase, ri| {
                let r = &mut rs[ri];
                let t = &engine.ranks[ri];
                match phase {
                    ReplayPhase::Upload => {
                        long_ctx_per_rank[ri] = r.prepare_batched_propose_graph_inputs_async(
                            t,
                            slots,
                            anchors,
                            &tail_ptrs_per_rank[ri],
                            k_max,
                        )?;
                        Ok(())
                    }
                    ReplayPhase::Capture => r.ensure_batched_propose_graph(
                        t,
                        b_reqs,
                        k_max,
                        long_ctx_per_rank[ri],
                        comm,
                    ),
                    ReplayPhase::Launch => r.replay_batched_propose_graph_async(
                        t,
                        b_reqs,
                        k_max,
                        long_ctx_per_rank[ri],
                    ),
                }
            });
            let mut outputs = match replay_res {
                Ok(()) => rs
                    .iter_mut()
                    .zip(&engine.ranks)
                    .map(|(r, t)| r.collect_batched_propose_host(t, slots, k_budgets, k_max))
                    .collect::<Result<Vec<_>>>()?,
                Err(error) => {
                    if error.partial_launch {
                        eprintln!(
                            "[warp34] fatal: asymmetric batched MTP TP graph launch failure: {}",
                            error.source
                        );
                        std::process::abort();
                    }
                    return Err(error.source);
                }
            };
            if outputs.windows(2).any(|w| w[0] != w[1]) {
                return Err(loader_err("batched MTP TP token disagreement"));
            }
            Ok(outputs.swap_remove(0))
        })();

        match drafts_res {
            Ok(drafts) => {
                self.finish_batched_mtp_slot_commits(slots, pending_ingests, &staged_memos);
                Ok(Some(drafts))
            }
            Err(e) => {
                for (b, (&slot, &toks)) in slots.iter().zip(pending_ingests).enumerate() {
                    if !toks.is_empty() {
                        let memo_opt = if slot == self.mtp_batch_active_slot {
                            &mut self.current
                        } else {
                            &mut self.mtp_batch_slots[slot].current
                        };
                        if let Some(memo) = memo_opt
                            && let Some((_, old_len, _)) = staged_memos[b]
                        {
                            memo.features.truncate(old_len);
                        }
                    }
                }
                Err(e)
            }
        }
    }
    /// Runs one diagnostic drafter ingest + proposal step on `prompt` and `anchor`,
    /// recording `"drafter_hidden"` and `draft_proposal_ids` into `dump`.
    pub fn captured_batched_mtp_propose_graph_shapes(&self) -> Vec<(usize, usize, bool)> {
        match &self.ranks {
            Ranks::Mtp(rs) => rs
                .first()
                .map_or_else(Vec::new, |r| r.captured_batched_propose_graph_shapes()),
            Ranks::DFlash(_) => Vec::new(),
        }
    }

    pub fn dump_step(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        anchor: i32,
        dump: &mut crate::model::ForwardDump,
    ) -> Result<()> {
        self.restore(engine, &[])?;
        let features = self.captured(engine)?;
        self.ingest(engine, prompt, &features)?;
        let drafts = self.propose(engine, anchor, self.k)?;
        dump.draft_proposal_ids = drafts;
        let t0 = &engine.ranks[0];
        let h = engine.cfg.hidden_size;
        t0.device.make_current()?;
        match &self.ranks {
            Ranks::Mtp(rs) => {
                let cols = self.row_bytes / 2;
                let mut bits = vec![0u16; cols];
                rs[0].hidden.copy_to_host(&mut bits, &t0.stream)?;
                let vals: Vec<f32> = bits.into_iter().map(f16_to_f32).collect();
                dump.tensors_f32
                    .insert("drafter_hidden".to_string(), (vec![1, cols], vals));
            }
            Ranks::DFlash(rs) => {
                let rows = rs[0].cfg.block_size;
                let mut bits = vec![0u16; rows * h];
                rs[0].hidden.copy_to_host(&mut bits, &t0.stream)?;
                let vals: Vec<f32> = bits
                    .into_iter()
                    .map(|b| f32::from_bits((b as u32) << 16))
                    .collect();
                dump.tensors_f32
                    .insert("drafter_hidden".to_string(), (vec![rows, h], vals));
            }
        }
        Ok(())
    }

    pub fn generate(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        max_new: usize,
        chunk: usize,
        stop_eos: bool,
    ) -> Result<RadixBenchMetrics> {
        self.generate_impl(engine, prompt, max_new, chunk, stop_eos, None)
    }
    /// Exact stochastic target verification with deterministic greedy proposals
    /// (q is a point mass), including processed penalties and grammar masks.
    pub fn generate_sampled(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        params: SamplingParams,
        constraint: Option<TokenBitmaskConstraint>,
        chunk: usize,
    ) -> Result<RadixBenchMetrics> {
        if !params.stop_strings.is_empty() {
            return Err(loader_err(
                "native sampling does not yet support stop strings",
            ));
        }
        if params
            .stop_token_ids
            .iter()
            .any(|&id| id < 0 || id as usize >= engine.cfg.vocab_size)
        {
            return Err(loader_err("invalid sampling stop token"));
        }
        params
            .validate()
            .map_err(|_| loader_err("invalid sampling parameters"))?;
        if let Some(ref c) = constraint
            && !(0..engine.cfg.vocab_size as i32).any(|tid| c.is_token_allowed(tid))
        {
            return Err(loader_err("grammar has no support in the vocabulary"));
        }
        let count = params.max_tokens;
        let sampler = SamplerV1::new(params, prompt, constraint);
        self.generate_impl(engine, prompt, count, chunk, true, Some(sampler))
    }
    fn generate_impl(
        &mut self,
        engine: &mut Qwen3_5Engine,
        prompt: &[i32],
        max_new: usize,
        chunk: usize,
        stop_eos: bool,
        configured_sampler: Option<SamplerV1>,
    ) -> Result<RadixBenchMetrics> {
        let total = prompt
            .len()
            .checked_add(max_new.max(1))
            .ok_or_else(|| loader_err("speculative length overflow"))?;
        if prompt.is_empty()
            || total > self.capacity
            || prompt
                .iter()
                .any(|&id| id < 0 || id as usize >= engine.cfg.vocab_size)
        {
            return Err(loader_err(
                "invalid speculative prompt or insufficient KV capacity",
            ));
        }
        let radix_prompt = crate::lora::salt_radix_key_for_lora(prompt, engine.active_lora_id);
        let lookup = engine
            .radix
            .match_prefix(&radix_prompt[..radix_prompt.len() - 1]);
        let hit = if self
            .can_restore_for_prompt(&radix_prompt[..lookup.matched_tokens], radix_prompt.len())
        {
            lookup
        } else {
            let mut cold = engine.radix.match_prefix(&[]);
            cold.mamba_branching_seqlen = lookup.mamba_branching_seqlen;
            cold
        };
        let cached = hit.matched_tokens;
        self.restore(engine, &radix_prompt[..cached])?;
        engine.radix.inc_lock_ref(hit.last_node);
        let Some((active, pp)) = engine.alloc_three_state_slots() else {
            engine.radix.dec_lock_ref(hit.last_node);
            return Err(loader_err("no speculative request state slots"));
        };
        engine.radix.dec_lock_ref(hit.last_node);
        let mut req = ReqRadixState::new(16, 64, active, pp);
        req.restore_prefix_hit(&mut engine.radix, &radix_prompt, &hit);
        let mut tx = StateTx::new(active);
        let processed_sampling = configured_sampler.is_some();
        let mut sampler = configured_sampler.unwrap_or_else(|| {
            SamplerV1::new(
                SamplingParams {
                    stop_token_ids: engine.cfg.eos_token_ids.clone(),
                    eos_token_ids: engine.cfg.eos_token_ids.clone(),
                    ..SamplingParams::default()
                },
                prompt,
                None,
            )
        });
        let mut prefill_donated_seq_len = 0usize;
        let started = Instant::now();
        let result = (|| -> Result<(Vec<i32>, Vec<crate::parser::StepLogprobs>, f64, f64)> {
            engine.materialize_pending_checkpoint(&mut req)?;
            engine.ensure_kv_pages_for_len(&mut req, total)?;
            let mut anchor = 0;
            let mut cursor = cached;
            while cursor < prompt.len() {
                let rem = prompt.len() - cursor;
                let mamba_chunk = req.mamba_chunk_size.max(1);
                let mut max_take = rem.min(chunk.max(1).min(self.max_rows));
                if let Some(branch_end) = hit.mamba_branching_seqlen {
                    let aligned_branch = (branch_end / mamba_chunk) * mamba_chunk;
                    if aligned_branch > cursor && aligned_branch < prompt.len() {
                        max_take = max_take.min(aligned_branch - cursor);
                    }
                }
                let take = if cursor + max_take < prompt.len() {
                    let next_aligned = ((cursor + max_take) / mamba_chunk) * mamba_chunk;
                    if next_aligned > cursor {
                        next_aligned - cursor
                    } else {
                        max_take
                    }
                } else {
                    max_take
                };
                let end = cursor + take;
                let tokens = &prompt[cursor..end];
                let radix_tokens = &radix_prompt[cursor..end];
                let planned = req.plan_checkpoints(tokens.len());
                let pairs: Vec<_> = planned
                    .iter()
                    .map(|p| (p.trigger_offset_1based, p.dst_slot))
                    .collect();
                let is_last_chunk = end == prompt.len();
                let emit_last = is_last_chunk && max_new > 0;
                let out = engine.step_batch_slices_for_sampler_filtered(
                    &[EngineBatchSlice {
                        tokens,
                        ctx_len_before: cursor,
                        block_table: &req.kv_pages,
                        state_in_slot: req.current_state_in_slot(),
                        state_out_slot: req.active_state_slot,
                        has_initial_state: req.has_initial_state,
                        spec_slots: &[],
                        checkpoints: &pairs,
                        emit_all_query_tokens: false,
                    }],
                    None,
                    Some(&[emit_last]),
                    processed_sampling && emit_last,
                )?;
                if emit_last && !processed_sampling {
                    anchor = out[0][0];
                }
                let features = self.captured(engine)?;
                self.ingest_for_prompt_len(engine, radix_tokens, &features, prompt.len())?;
                req.commit_tokens(radix_tokens, &planned);
                cursor = end;
                prefill_donated_seq_len =
                    prefill_donated_seq_len.max(engine.donate_intermediate_prefill_checkpoint(
                        &mut req,
                        cursor,
                        prompt.len(),
                        max_new,
                        false,
                    )?);
            }
            tx.committed_slot = req.active_state_slot;
            tx.commit_prefill(prompt.len());
            let ttft = started.elapsed().as_secs_f64() * 1000.;
            let decoding = Instant::now();
            let mut generated = Vec::with_capacity(max_new);
            let mut token_logprobs = Vec::new();
            if max_new > 0 {
                if processed_sampling {
                    let (tok, lp) = engine.sample_single_gpu_step(&mut sampler)?;
                    anchor = tok;
                    if let Some(lp) = lp {
                        token_logprobs.push(lp);
                    }
                } else {
                    sampler.commit_token(anchor);
                }
                generated.push(anchor);
            }
            let _decode_range = crate::profiling::Range::new(c"warp34.decode");
            let mut pending_ingest: Vec<i32> = Vec::new();
            while generated.len() < max_new
                && !generation_stopped(&sampler, anchor, stop_eos, &engine.cfg.eos_token_ids)
            {
                let k = self.k.min(max_new - generated.len() - 1);
                let proposer_range = crate::profiling::Range::new(c"warp34.proposer");
                let mut drafts = if pending_ingest.is_empty() {
                    self.propose(engine, anchor, k)?
                } else {
                    let features = self.captured(engine)?;
                    let out =
                        self.ingest_and_propose(engine, &pending_ingest, &features, anchor, k)?;
                    pending_ingest.clear();
                    out
                };
                drop(proposer_range);
                if stop_eos
                    && let Some(eos) = drafts.iter().enumerate().position(|(idx, id)| {
                        let pos_before = generated.len() + idx;
                        if pos_before < sampler.params.min_tokens {
                            sampler.is_token_soft_banned(*id, &engine.cfg.eos_token_ids)
                        } else {
                            sampler
                                .params
                                .is_stop_or_eos(*id, &engine.cfg.eos_token_ids)
                        }
                    })
                {
                    drafts.truncate(eos);
                }
                let mut extras = Vec::with_capacity(drafts.len());
                for _ in 0..drafts.len() {
                    let Some(s) = engine.alloc_state_slot() else {
                        break;
                    };
                    extras.push(s);
                }
                drafts.truncate(extras.len());
                let prepared_sampler = if processed_sampling {
                    Some(crate::model::PreparedSpecSamplerRows::prepare_drafts(
                        engine.cfg.vocab_size,
                        &engine.cfg.eos_token_ids,
                        &sampler,
                        &mut drafts,
                    )?)
                } else {
                    None
                };
                if extras.len() > drafts.len() {
                    let surplus: Vec<_> = extras.drain(drafts.len()..).collect();
                    engine.free_state_slots(&surplus);
                }
                let k = drafts.len();
                let mut verify = vec![anchor];
                verify.extend_from_slice(&drafts);
                let schedule = if k > 0 {
                    tx.begin_spec_verify(extras);
                    tx.verify_slot_schedule()
                } else {
                    vec![tx.committed_slot]
                };
                let before = req.token_ids.len();
                let planned: Vec<PlannedCheckpoint> = req.plan_checkpoints(verify.len());
                let pairs: Vec<_> = planned
                    .iter()
                    .map(|p| (p.trigger_offset_1based, p.dst_slot))
                    .collect();
                let out = engine.step_batch_slices_for_sampler(
                    &[EngineBatchSlice {
                        tokens: &verify,
                        ctx_len_before: before,
                        block_table: &req.kv_pages,
                        state_in_slot: tx.committed_slot,
                        state_out_slot: *schedule.last().unwrap(),
                        has_initial_state: true,
                        spec_slots: &schedule,
                        checkpoints: &pairs,
                        emit_all_query_tokens: true,
                    }],
                    None,
                    processed_sampling,
                )?;
                let (decision, round_lps) = if let Some(prepared) = prepared_sampler {
                    engine.verify_speculative_gpu_prepared(prepared, &sampler, &drafts)?
                } else {
                    (
                        RejectionSampler::greedy(&drafts, &out[0])
                            .map_err(|_| loader_err("greedy rejection geometry"))?,
                        Vec::new(),
                    )
                };
                let processed = decision.processed_rows();
                if k > 0 {
                    let freed = decision
                        .commit(&mut tx, &mut sampler, 0)
                        .map_err(|_| loader_err("speculative sampler commit"))?;
                    engine.free_state_slots(&freed);
                } else {
                    tx.kv_seq_len_before += 1;
                    sampler
                        .try_commit_token(decision.emitted_tokens()[0])
                        .map_err(|_| loader_err("sampler/grammar commit failed"))?;
                }
                req.active_state_slot = tx.committed_slot;
                req.commit_tokens(&verify[..processed], &planned);
                pending_ingest.extend_from_slice(&verify[..processed]);
                self.last_accepted.push(decision.accepted_drafts());
                generated.extend_from_slice(decision.emitted_tokens());
                token_logprobs.extend(round_lps);
                anchor = *generated.last().unwrap();
            }
            if !pending_ingest.is_empty() {
                let features = self.captured(engine)?;
                self.ingest(engine, &pending_ingest, &features)?;
            }
            debug_assert_eq!(sampler.output_history, generated);
            Ok((
                generated,
                token_logprobs,
                ttft,
                decoding.elapsed().as_secs_f64() * 1000.,
            ))
        })();
        let (generated, token_logprobs, ttft, decode) = match result {
            Ok(v) => v,
            Err(e) => {
                if !tx.spec_slots.is_empty() {
                    let freed = tx.resolve_spec_verify(1);
                    engine.free_state_slots(&freed);
                }
                req.active_state_slot = tx.committed_slot;
                let release = req.abort_and_release(&mut engine.radix);
                engine.free_kv_pages(&release.freed_kv_pages);
                engine.free_state_slots(&release.freed_mamba_slots);
                self.current = None;
                if matches!(self.ranks, Ranks::DFlash(_)) {
                    self.memos.clear();
                }
                return Err(e);
            }
        };
        let donation = {
            let mut radix = std::mem::replace(
                &mut engine.radix,
                crate::radix::MambaRadixCache::new(16, 64),
            );
            let res = engine.donate_req_checkpoint(&mut radix, &mut req, true);
            engine.radix = radix;
            res?
        };
        self.commit_current_memo();
        let steps = generated.len().saturating_sub(1);
        Ok(RadixBenchMetrics {
            prompt_len: prompt.len(),
            cached_prefix_tokens: cached,
            donated_seq_len: donation.donated_seq_len.max(prefill_donated_seq_len),
            gen_len: generated.len(),
            ttft_ms: ttft,
            prefill_tok_per_s: if ttft > 0. {
                (prompt.len() - cached) as f64 * 1000. / ttft
            } else {
                0.
            },
            decode_ms: decode,
            decode_tok_per_s: if decode > 0. {
                steps as f64 * 1000. / decode
            } else {
                0.
            },
            itl_ms: if steps > 0 { decode / steps as f64 } else { 0. },
            generated_token_ids: generated,
            token_logprobs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_stops_still_respect_min_tokens_and_stop_eos() {
        for constraint in [
            None,
            Some(TokenBitmaskConstraint::from_allowed_tokens(8, vec![4, 7])),
        ] {
            let mut sampler = SamplerV1::new(
                SamplingParams {
                    max_tokens: 4,
                    min_tokens: 2,
                    stop_token_ids: vec![4],
                    ..SamplingParams::default()
                },
                &[1],
                constraint,
            );
            sampler.try_commit_token(4).unwrap();
            assert!(!generation_stopped(&sampler, 4, true, &[7]));
            assert!(!generation_stopped(&sampler, 7, true, &[7]));
            sampler.try_commit_token(4).unwrap();
            for anchor in [4, 7] {
                assert!(generation_stopped(&sampler, anchor, true, &[7]));
                assert!(!generation_stopped(&sampler, anchor, false, &[7]));
            }
            assert!(!generation_stopped(&sampler, 3, true, &[7]));
        }
    }

    #[test]
    fn sampled_terminal_grammar_stops_independently() {
        for min_tokens in [0, 3] {
            for stop_eos in [false, true] {
                let constraint = TokenBitmaskConstraint::from_allowed_tokens(8, vec![4])
                    .with_terminal_eos_ids(vec![4]);
                let mut sampler = SamplerV1::new(
                    SamplingParams {
                        max_tokens: 4,
                        min_tokens,
                        ..SamplingParams::default()
                    },
                    &[1],
                    Some(constraint),
                );
                assert!(!generation_stopped(&sampler, 1, stop_eos, &[7]));
                sampler.try_commit_token(4).unwrap();
                assert!(sampler.constraint.as_ref().unwrap().is_terminal_eos());
                assert!(
                    generation_stopped(&sampler, 4, stop_eos, &[7]),
                    "terminal grammar must stop before another sample, min_tokens={min_tokens}, stop_eos={stop_eos}"
                );
            }
        }
    }

    #[test]
    fn mtp_memo_growth_bounds_slack_and_deduplicates_prefixes() {
        let row_bytes = 10_240;
        let mut buf = vec![1u8; 10_000 * row_bytes];
        buf.shrink_to_fit();
        // Appending 128 new SWE tool-observation tokens must grow by at most one 512-row
        // (~5 MiB) chunk rather than doubling a 100 MiB buffer to 200 MiB.
        grow_memo_features(&mut buf, 10_128 * row_bytes, row_bytes);
        assert_eq!(buf.len(), 10_128 * row_bytes);
        assert!(
            buf.capacity() <= (10_128 + MEMO_GROWTH_ROWS) * row_bytes,
            "unexpected Vec doubling: len={} cap={}",
            buf.len(),
            buf.capacity()
        );

        let mut memos = Vec::new();
        let mut m1_feat = Vec::new();
        grow_memo_features(&mut m1_feat, 512, 16);
        let expected_cap = m1_feat.capacity();
        push_mtp_memo(
            &mut memos,
            PrefixMemo {
                tokens: vec![1, 2, 3],
                features: m1_feat,
            },
        );
        assert_eq!(memos.len(), 1);
        assert_eq!(memos[0].features.capacity(), expected_cap);

        // Strict prefix extension replaces the shorter turn memo cleanly.
        push_mtp_memo(
            &mut memos,
            PrefixMemo {
                tokens: vec![1, 2, 3, 4, 5],
                features: vec![7; 1024],
            },
        );
        assert_eq!(memos.len(), 1);
        assert_eq!(memos[0].tokens, vec![1, 2, 3, 4, 5]);

        // Re-committing a shorter or empty-feature prefix never downgrades an existing memo.
        push_mtp_memo(
            &mut memos,
            PrefixMemo {
                tokens: vec![1, 2, 3, 4, 5],
                features: vec![],
            },
        );
        assert_eq!(memos.len(), 1);
        assert_eq!(memos[0].features.len(), 1024);
    }

    #[test]
    fn scheduled_suffix_lookup_matches_continuation_and_rejects_stop_or_short_history() {
        let sampler = SamplerV1::new(SamplingParams::default(), &[1], None);
        let mut history: Vec<i32> = (100..180).collect();
        history.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70]);
        history.extend_from_slice(&[200, 201, 202, 10, 20]);
        // Suffix ending with anchor=30 matches [10, 20, 30] and returns [40, 50, 60].
        assert_eq!(
            scheduled::try_lookup_scheduled_suffix_drafts(&history, 30, 3, 3, 5, &sampler, &[999],),
            Some(vec![40, 50, 60]),
        );
        // Stop/EOS token in looked-up continuation rejects the bypass so normal proposer handles truncation.
        assert_eq!(
            scheduled::try_lookup_scheduled_suffix_drafts(&history, 30, 3, 3, 5, &sampler, &[50],),
            None,
        );
        // Short history (< 64 tokens) does not trigger suffix lookup bypass.
        assert_eq!(
            scheduled::try_lookup_scheduled_suffix_drafts(
                &history[..32],
                130,
                3,
                3,
                5,
                &sampler,
                &[999],
            ),
            None,
        );
    }

    #[test]
    fn dflash_slot_has_prefix_for_prompt_handles_ring_wrap_and_suffix_window() {
        let res: Vec<i32> = (0..22_430).collect();
        let prompt: Vec<i32> = (0..22_670).collect();
        // When suffix prefill (`22,670 - 21,952 = 718`) advances decode past `22,670`,
        // the sliding window `[22,670 - 2,048 .. 21,952]` (`pages 1,288 .. 1,372`) is still
        // inside a 138-page (`2,208`-token) ring buffer (`1,402 - 1,288 = 114 <= 138`).
        assert!(dflash_slot_has_prefix_for_prompt(
            &res,
            0,
            &prompt[..21_952],
            prompt.len(),
            Some(138),
            2048,
        ));
        // If a prior restore took the `set_committed(21_952)` shortcut without writing
        // `[0..21_952]` into the ring buffer (`valid_from = 21_952`), a subsequent short-suffix
        // request (`prompt_len = 22_670`, needing KV from `20_608..21_952`) must reject it,
        // whereas a request whose suffix (`>= 2,048`) overwrites the entire window is allowed.
        assert!(!dflash_slot_has_prefix_for_prompt(
            &res,
            21_952,
            &prompt[..21_952],
            prompt.len(),
            Some(138),
            2048,
        ));
        assert!(dflash_slot_has_prefix_for_prompt(
            &res[..21_952],
            21_952,
            &prompt[..21_952],
            21_952 + 2_048,
            Some(138),
            2048,
        ));
        assert_eq!(
            dflash_effective_valid_from(22_430, 0, Some(138)),
            (22_430usize.div_ceil(16) - 138) * 16
        );
        // In single-slot mode (`!batch_mode`, e.g. `--max-num-seqs 1`), `DFlash2Rank::pages_dev`
        // wraps modulo `all_pages.len()` (`8 * 138 = 1,104` pages = `17,664` tokens on TP=1).
        // When a turn at `103,552` generates `2,680` tokens (`resident_len = 106,248`) and the
        // follow-up prompt (`104,528` tokens) strips `<think>...</think>` back to `103,552`,
        // the 1,104-page single-slot ring still holds `[88,592 .. 106,248]`, so `103,552` is
        // restored directly without stepping back to `102,464` (`admission_filter`).
        let long_res: Vec<i32> = (0..106_248).collect();
        let next_prompt: Vec<i32> = (0..104_528).collect();
        assert!(!dflash_slot_has_prefix_for_prompt(
            &long_res,
            0,
            &next_prompt[..103_552],
            next_prompt.len(),
            Some(138),
            2048,
        ));
        assert!(dflash_slot_has_prefix_for_prompt(
            &long_res,
            0,
            &next_prompt[..103_552],
            next_prompt.len(),
            Some(8 * 138),
            2048,
        ));
    }

    #[test]
    fn proposer_max_rows_preserves_2048_chunk_prefill_for_tp2_262k_serving() {
        for kind in [SpeculativeKind::Mtp, SpeculativeKind::DFlash2] {
            // Small test pools (< 4096 pages) use 256-row tap buffers.
            assert_eq!(plan_proposer_max_rows(false, kind, 1, 256), 256);
            assert_eq!(plan_proposer_max_rows(false, kind, 2, 256), 256);

            // Standard serving pools (4096..14336 pages = 64K..224K tokens) use 2048-row tap buffers.
            assert_eq!(plan_proposer_max_rows(false, kind, 1, 4096), 2048);
            assert_eq!(plan_proposer_max_rows(false, kind, 2, 8192), 2048);

            // TP=1 extreme long-context (>= 14336 pages = 229K..262K on 1x 24GB GPU with 336 MiB scratch)
            // uses 1024-row prefill chunks (with OOM fallback to 256), while TP>=2 262K serving keeps 2048.
            assert_eq!(plan_proposer_max_rows(false, kind, 1, 16384), 1024);
            assert_eq!(plan_proposer_max_rows(false, kind, 2, 16384), 2048);
        }
        assert_eq!(
            plan_proposer_max_rows(true, SpeculativeKind::Mtp, 2, 4096),
            6144
        );
        assert_eq!(
            plan_proposer_max_rows(true, SpeculativeKind::DFlash2, 2, 4096),
            2048
        );
    }

    #[test]
    fn dflash_prefill_window_skip_and_memo_cap_invariants() {
        // Skipping `[0..1024]` during a `1,550`-token prefill (`window = 512`, `phys_pages = 42`)
        // sets `valid_from = 1024`. Subsequent primed requests at `1,536` (`prompt_len = 1,550`,
        // `win_start_tok = 1,024`) still hit resident GPU KV, whereas shorter branches (`900`)
        // reject the partial GPU ring and replay from host `PrefixMemo`.
        let res: Vec<i32> = (0..1550).collect();
        assert!(dflash_slot_has_prefix_for_prompt(
            &res,
            1024,
            &res[..1536],
            1550,
            Some(42),
            512,
        ));
        assert!(!dflash_slot_has_prefix_for_prompt(
            &res,
            1024,
            &res[..896],
            900,
            Some(42),
            512,
        ));

        // DFlash2 host memos store projected `ctx = rms_norm(fc(tap_aux))` (`1 × H × 2B`,
        // `10,240 B/row` on 27B, 5× smaller than raw taps), so multi-turn SWE-bench agents
        // fit past 4K up to `MEMO_BUDGET` just like MTP.
        assert!(memo_fits_budget(
            SpeculativeKind::DFlash2,
            0,
            65_536,
            10_240
        ));
        assert!(!memo_fits_budget(
            SpeculativeKind::DFlash2,
            0,
            500_000,
            10_240
        ));
        assert!(memo_fits_budget(SpeculativeKind::Mtp, 0, 5000, 4_096));
    }

    #[test]
    fn mtp_27b_wide_hidden_memo_fits_64k_and_120k_swe_agent() {
        // Qwen3.8-27B MTP has `hidden_size = 5120` (`10,240 B/row` > `8,192 B/row`) and has no
        // sliding-window ring shortcut, so `memo_fits_budget` must permit 64K and 120K prefix
        // memos up to `MEMO_BUDGET` (`4 GiB`) instead of mistaking `row_bytes > 8192` for DFlash2.
        let mtp_27b_row_bytes = 5120 * 2;
        assert!(memo_fits_budget(
            SpeculativeKind::Mtp,
            0,
            65_536,
            mtp_27b_row_bytes
        ));
        assert!(memo_fits_budget(
            SpeculativeKind::Mtp,
            65_536,
            65_536,
            mtp_27b_row_bytes
        ));
    }

    #[test]
    fn dp_memo_budget_scales_per_replica_with_512m_floor() {
        assert_eq!(plan_memo_budget(1), 4 * 1024 * 1024 * 1024);
        assert_eq!(plan_memo_budget(2), 2 * 1024 * 1024 * 1024);
        assert_eq!(plan_memo_budget(4), 1024 * 1024 * 1024);
        assert_eq!(plan_memo_budget(8), 512 * 1024 * 1024);
        assert_eq!(plan_memo_budget(16), 512 * 1024 * 1024);
        assert!(!memo_fits_budget_limit(
            SpeculativeKind::Mtp,
            200_000,
            100_000,
            10_240,
            plan_memo_budget(2),
        ));
        assert!(memo_fits_budget_limit(
            SpeculativeKind::Mtp,
            0,
            65_536,
            10_240,
            plan_memo_budget(4),
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn clone_aligned_memo_prefix_preserves_shared_4k_harness_sibling_suffixes() {
        let row_bytes = 16;
        let budget = plan_memo_budget(1);
        // Agent A finished Turn 1 at 4,200 tokens (4,096 shared harness + 104 unique tokens).
        let agent_a_tokens: Vec<i32> = (0..4200).collect();
        let agent_a_feats = vec![0x5Au8; 4200 * row_bytes];
        let mut memos = vec![PrefixMemo {
            tokens: agent_a_tokens.clone(),
            features: agent_a_feats.clone(),
        }];

        // New Agent B arrives restoring the 4,096-token shared harness (4200 - 4096 = 104 < 256).
        // `clone_aligned_memo_prefix` MUST NOT truncate Agent A's 4,200-token memo in-place!
        let harness = &agent_a_tokens[..4096];
        let b_memo = NativeProposer::clone_aligned_memo_prefix(
            &mut memos,
            harness,
            4096 * row_bytes,
            row_bytes,
            budget,
        );
        assert_eq!(b_memo.tokens.len(), 4096);
        assert_eq!(b_memo.features.len(), 4096 * row_bytes);
        assert_eq!(
            memos.len(),
            1,
            "Agent A's memo must remain in memos for Turn 2"
        );
        assert_eq!(
            memos[0].tokens.len(),
            4200,
            "Agent A's 104 unique Turn-1 tokens must not be truncated by a shared-harness peer"
        );

        // Now Agent A itself returns for Turn 2 at 5,120 tokens where its stored memo is 5,150
        // (`n = 5120 > 4096` and `5150 - 5120 = 30 < 64` Mamba chunk tail) -> moves in O(1).
        let long_a: Vec<i32> = (0..5150).collect();
        memos = vec![PrefixMemo {
            tokens: long_a.clone(),
            features: vec![0x3Cu8; 5150 * row_bytes],
        }];
        let moved = NativeProposer::clone_aligned_memo_prefix(
            &mut memos,
            &long_a[..5120],
            5120 * row_bytes,
            row_bytes,
            budget,
        );
        assert_eq!(moved.tokens.len(), 5120);
        assert!(
            memos.is_empty(),
            "single-session Mamba chunk tail (<64 past >4K) should move in O(1)"
        );
    }
}
