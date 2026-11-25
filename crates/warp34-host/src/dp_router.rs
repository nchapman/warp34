//! In-process Data-Parallel (`--dp`) prefix directory and load/affinity request router.
//!
//! Designed for `DP = 1..=64` (primary targets: 2×, 3×, and 4× GPU servers, including
//! `DP=2 × TP=2` on 4 GPUs). Because all DP workers live in one Rust process:
//! - Each worker's [`crate::radix::MambaRadixCache`] updates [`DpPrefixDirectory`] directly
//!   on insert/evict/reset, giving exact zero-drift prefix visibility without an external
//!   indexer or network KV-events.
//! - Routing combines exact chained 64-token block-hash prefix matching (galloping + binary
//!   search in `O(log N)` under a read lock), quadratic length weighting (borrowed from
//!   `llm-d`'s `prefix-cache-scorer`), an idle-GPU spread guard (preventing staggered agents
//!   with a shallow shared harness prefix from piling onto GPU 0), router-local atomic
//!   in-flight tracking (borrowed from `sgl-router`), and hard KV/queue admission guards.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Tokens per chained prefix block in [`DpPrefixDirectory`] (matches `MambaRadixCache`
/// `mamba_chunk_size = 64` and `llm-d` `blockSizeTokens = 64`).
pub const DP_PREFIX_BLOCK_TOKENS: usize = 64;

/// Maximum supported in-process DP replicas (`u64` worker bitmask).
pub const MAX_DP_WORKERS: usize = 64;

/// Minimum matched prefix tokens required to override an idle GPU when no idle worker
/// already holds the shared harness prefix.
const IDLE_SPREAD_MIN_MATCHED_TOKENS: usize = 4096;
/// Minimum fraction of prompt tokens matched required to override an idle GPU.
const IDLE_SPREAD_MIN_MATCH_RATIO: f64 = 0.75;
/// Minimum fraction of prompt tokens matched to keep a multi-turn continuation pinned when
/// an idle worker already holds the shallow shared harness (`best_idle_match > 0`).
const IDLE_CONTINUATION_MIN_MATCH_RATIO: f64 = 0.65;
/// Maximum shallow shared-harness match (tokens) that should still spread across cold idle
/// workers when multiple workers are idle.
const SHALLOW_HARNESS_MAX_TOKENS: usize = 2048;
/// Minimum unique prefix-match advantage (tokens) required to award a non-zero prefix bonus.
const MIN_PREFIX_BONUS_DELTA_TOKENS: usize = 256;
/// Quadratic length-bonus scale `(delta_matched / scale)^2`.
const PREFIX_LENGTH_SCALE_TOKENS: f64 = 4096.0;
/// Active KV utilization above which a worker is treated as saturated for admission filtering.
const SATURATED_KV_UTIL_THRESHOLD: f64 = 0.90;
/// Maximum bounded session-affinity map entries before LRU eviction.
const MAX_SESSION_AFFINITY_ENTRIES: usize = 4096;

/// Request routing policy across in-process DP workers (`--dp-routing`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DpRoutingPolicy {
    /// Exact radix prefix affinity + load/KV-pressure balance + idle-GPU spread guard.
    #[default]
    Prefix,
    /// Pure least-loaded (running + waiting + router-inflight + KV pressure).
    LeastLoaded,
    /// Round-robin across healthy workers.
    RoundRobin,
}

impl DpRoutingPolicy {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "prefix" | "cache" | "cache-aware" | "cache_aware" | "auto" => Ok(Self::Prefix),
            "least-loaded" | "least_loaded" | "load" => Ok(Self::LeastLoaded),
            "round-robin" | "round_robin" | "rr" => Ok(Self::RoundRobin),
            other => Err(format!(
                "invalid --dp-routing '{other}'; expected one of: prefix, least-loaded, round-robin"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::LeastLoaded => "least-loaded",
            Self::RoundRobin => "round-robin",
        }
    }
}

/// Computes chained 64-bit FNV-1a / SplitMix64 block hashes over already-salted token IDs
/// (as stored in [`crate::radix::MambaRadixCache`] nodes).
pub fn compute_chained_prefix_hashes_raw(tokens: &[i32]) -> Vec<u64> {
    let num_blocks = tokens.len() / DP_PREFIX_BLOCK_TOKENS;
    if num_blocks == 0 {
        return Vec::new();
    }
    let mut hashes = Vec::with_capacity(num_blocks);
    let mut state = mix64(0x9e37_79b9_7f4a_7c15u64);
    for block in tokens.as_chunks::<DP_PREFIX_BLOCK_TOKENS>().0 {
        let mut h = state;
        for &tok in block {
            h ^= (tok as u32) as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        state = mix64(h);
        hashes.push(state);
    }
    hashes
}

/// Computes chained 64-bit block hashes for raw (unsalted) prompt tokens and `lora_id`,
/// applying [`crate::lora::salt_radix_key_in_place`] to token 0 inline (zero allocation)
/// when `lora_id >= 0` so hashes match [`crate::radix::MambaRadixCache`] nodes identically
/// for both base model and LoRA adapters.
pub fn compute_chained_prefix_hashes(lora_id: i32, tokens: &[i32]) -> Vec<u64> {
    let num_blocks = tokens.len() / DP_PREFIX_BLOCK_TOKENS;
    if num_blocks == 0 {
        return Vec::new();
    }
    if lora_id < 0 {
        return compute_chained_prefix_hashes_raw(tokens);
    }
    let mut first_salted = [tokens[0]];
    crate::lora::salt_radix_key_in_place(&mut first_salted, lora_id);
    let mut hashes = Vec::with_capacity(num_blocks);
    let mut state = mix64(0x9e37_79b9_7f4a_7c15u64);
    for (block_idx, block) in tokens
        .as_chunks::<DP_PREFIX_BLOCK_TOKENS>()
        .0
        .iter()
        .enumerate()
    {
        let mut h = state;
        for (i, &raw_tok) in block.iter().enumerate() {
            let tok = if block_idx == 0 && i == 0 {
                first_salted[0]
            } else {
                raw_tok
            };
            h ^= (tok as u32) as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        state = mix64(h);
        hashes.push(state);
    }
    hashes
}

#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Number of independent lock shards in [`DpPrefixDirectory`] (must be a power of two).
const DP_PREFIX_SHARDS: usize = 64;

/// Uncached prompt tokens equivalent to 1 sequence slot of prefill work in [`load_cost`].
const PREFILL_TOKEN_LOAD_UNIT: f64 = 1024.0;

/// Exact in-process directory mapping chained 64-token prefix block hashes to a `u64` bitmask
/// of DP ranks that currently hold that block in their [`crate::radix::MambaRadixCache`].
///
/// Sharded across `DP_PREFIX_SHARDS` (64) independent `RwLock` buckets so concurrent worker
/// inserts/evictions and HTTP prefix lookups never contend on a single global lock.
#[derive(Debug)]
pub struct DpPrefixDirectory {
    /// `shards[hash & (DP_PREFIX_SHARDS - 1)]`: `block_hash -> bitmask` where bit `dp_rank`
    /// is set iff worker `dp_rank` currently holds that block.
    shards: Box<[RwLock<HashMap<u64, u64>>; DP_PREFIX_SHARDS]>,
}

impl Default for DpPrefixDirectory {
    fn default() -> Self {
        Self::new()
    }
}

impl DpPrefixDirectory {
    pub fn new() -> Self {
        let shards = std::array::from_fn(|_| RwLock::new(HashMap::new()));
        Self {
            shards: Box::new(shards),
        }
    }

    #[inline]
    fn shard_for(&self, hash: u64) -> &RwLock<HashMap<u64, u64>> {
        &self.shards[(hash as usize) & (DP_PREFIX_SHARDS - 1)]
    }

    #[inline]
    fn lookup_mask(&self, hash: u64) -> u64 {
        self.shard_for(hash)
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&hash)
            .copied()
            .unwrap_or(0)
    }

    #[inline]
    fn rank_bit(dp_rank: usize) -> Option<u64> {
        (dp_rank < MAX_DP_WORKERS).then_some(1u64 << dp_rank)
    }

    /// Marks every complete 64-token block in `tokens` (for `lora_id`) as resident on `dp_rank`.
    pub fn record_insert(&self, dp_rank: usize, lora_id: i32, tokens: &[i32]) {
        let hashes = compute_chained_prefix_hashes(lora_id, tokens);
        self.record_insert_hashes(dp_rank, &hashes);
    }

    /// Marks pre-computed chained block hashes as resident on `dp_rank`.
    pub fn record_insert_hashes(&self, dp_rank: usize, hashes: &[u64]) {
        let Some(bit) = Self::rank_bit(dp_rank) else {
            return;
        };
        if hashes.is_empty() {
            return;
        }
        for &h in hashes {
            let mut guard = self.shard_for(h).write().unwrap_or_else(|e| e.into_inner());
            *guard.entry(h).or_insert(0) |= bit;
        }
    }

    /// Clears `dp_rank`'s bit for the given evicted block hashes in reverse order (tail-first)
    /// so concurrent galloping/binary prefix readers never observe a non-monotonic hole.
    pub fn record_evict_hashes(&self, dp_rank: usize, hashes: &[u64]) {
        let Some(bit) = Self::rank_bit(dp_rank) else {
            return;
        };
        if hashes.is_empty() {
            return;
        }
        for &h in hashes.iter().rev() {
            let mut guard = self.shard_for(h).write().unwrap_or_else(|e| e.into_inner());
            if let Some(mask) = guard.get_mut(&h) {
                *mask &= !bit;
                if *mask == 0 {
                    guard.remove(&h);
                }
            }
        }
    }

    /// Clears all blocks belonging to `dp_rank` (called on `MambaRadixCache::reset`).
    pub fn clear_worker(&self, dp_rank: usize) {
        let Some(bit) = Self::rank_bit(dp_rank) else {
            return;
        };
        for shard in &*self.shards {
            let mut guard = shard.write().unwrap_or_else(|e| e.into_inner());
            guard.retain(|_, mask| {
                *mask &= !bit;
                *mask != 0
            });
        }
    }

    /// Returns the number of matched prefix tokens (multiple of [`DP_PREFIX_BLOCK_TOKENS`])
    /// on each worker `0..num_workers` using galloping + binary search across lock shards.
    pub fn match_prefix_tokens_per_worker(
        &self,
        num_workers: usize,
        lora_id: i32,
        tokens: &[i32],
    ) -> Vec<usize> {
        let mut matched = vec![0usize; num_workers];
        let hashes = compute_chained_prefix_hashes(lora_id, tokens);
        if hashes.is_empty() || num_workers == 0 {
            return matched;
        }
        let first_mask = self.lookup_mask(hashes[0]);
        if first_mask == 0 {
            return matched;
        }
        let n = hashes.len();
        for (rank, slot) in matched
            .iter_mut()
            .enumerate()
            .take(num_workers.min(MAX_DP_WORKERS))
        {
            let bit = 1u64 << rank;
            if (first_mask & bit) == 0 {
                continue;
            }
            // Galloping (exponential) search from index 1, then binary search.
            let mut lo = 1usize; // known: hashes[0..lo] are present
            let mut step = 1usize;
            let mut hi = n;
            while lo < n {
                let probe = (lo + step - 1).min(n - 1);
                let present = (self.lookup_mask(hashes[probe]) & bit) != 0;
                if present {
                    lo = probe + 1;
                    step = step.saturating_mul(2);
                } else {
                    hi = probe;
                    break;
                }
            }
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let present = (self.lookup_mask(hashes[mid]) & bit) != 0;
                if present {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            *slot = lo * DP_PREFIX_BLOCK_TOKENS;
        }
        matched
    }

    #[cfg(test)]
    pub fn tracked_block_count(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.read().unwrap_or_else(|e| e.into_inner()).len())
            .sum()
    }
}

/// Live per-worker load & health snapshot consumed by [`select_dp_worker`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerLoadSample {
    pub healthy: bool,
    pub active_requests: usize,
    pub waiting_requests: usize,
    pub router_inflight: usize,
    /// Uncached prompt tokens currently in flight on this worker (prevents short
    /// requests from piling up behind a heavy 16K–32K prefill on another GPU).
    pub router_inflight_prefill_tokens: usize,
    pub max_num_seqs: usize,
    /// Active (non-evictable) KV utilization in `[0.0, 1.0]`.
    pub active_kv_util: f64,
}

impl WorkerLoadSample {
    #[inline]
    pub fn effective_running(self) -> usize {
        self.active_requests.max(self.router_inflight)
    }

    #[inline]
    pub fn total_load(self) -> usize {
        self.active_requests
            .saturating_add(self.waiting_requests)
            .max(self.router_inflight)
    }

    #[inline]
    pub fn is_idle(self) -> bool {
        self.active_requests == 0 && self.waiting_requests == 0 && self.router_inflight == 0
    }

    #[inline]
    pub fn is_saturated(self) -> bool {
        self.active_kv_util >= SATURATED_KV_UTIL_THRESHOLD
            || (self.max_num_seqs > 0 && self.waiting_requests >= self.max_num_seqs)
    }
}

/// Request routing context passed to [`select_dp_worker`].
#[derive(Clone, Copy, Debug)]
pub struct DpRouteContext<'a> {
    pub policy: DpRoutingPolicy,
    pub prompt_tokens: usize,
    pub matched_tokens: &'a [usize],
    pub pinned_worker: Option<usize>,
    pub tie_break_cursor: usize,
}

/// Pure, deterministic worker selection for `DP = 1..=64`.
///
/// Returns `None` only when zero workers are healthy.
pub fn select_dp_worker(workers: &[WorkerLoadSample], ctx: &DpRouteContext<'_>) -> Option<usize> {
    let n = workers.len();
    if n == 0 {
        return None;
    }
    let healthy: Vec<usize> = (0..n).filter(|&i| workers[i].healthy).collect();
    if healthy.is_empty() {
        return None;
    }
    if healthy.len() == 1 {
        return Some(healthy[0]);
    }

    if ctx.policy == DpRoutingPolicy::RoundRobin {
        let start = ctx.tie_break_cursor % n;
        for offset in 0..n {
            let idx = (start + offset) % n;
            if workers[idx].healthy {
                return Some(idx);
            }
        }
        return Some(healthy[0]);
    }

    // Hard admission filter: exclude KV/queue-saturated workers if any non-saturated worker exists.
    let admitted: Vec<usize> = healthy
        .iter()
        .copied()
        .filter(|&i| !workers[i].is_saturated())
        .collect();
    let candidates = if admitted.is_empty() {
        &healthy
    } else {
        &admitted
    };

    if ctx.policy == DpRoutingPolicy::LeastLoaded {
        return pick_lowest_score(candidates, n, ctx.tie_break_cursor, |idx| {
            load_cost(workers[idx])
        });
    }

    // Hard session affinity (`previous_response_id` or `x-session-id`), respected unless
    // the pinned worker is unhealthy or excluded by admission saturation.
    if let Some(pinned) = ctx.pinned_worker
        && candidates.contains(&pinned)
    {
        return Some(pinned);
    }

    let matched = |idx: usize| ctx.matched_tokens.get(idx).copied().unwrap_or(0);
    let prompt_len = ctx.prompt_tokens.max(1);

    // Idle-GPU spread guard: if any admitted worker is completely idle, only stack onto a
    // lightly-loaded busy worker when the prefix match is a genuine multi-turn continuation
    // (either >= 4K tokens & >= 75% of the prompt, or > 256 unique tokens past an idle
    // worker's shared harness with >= 65% prompt coverage).
    let idle_candidates: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&i| workers[i].is_idle())
        .collect();
    if !idle_candidates.is_empty() {
        let best_idle_match = idle_candidates
            .iter()
            .map(|&i| matched(i))
            .max()
            .unwrap_or(0);
        let best_busy_continuation = candidates
            .iter()
            .copied()
            .filter(|&i| !workers[i].is_idle())
            .filter(|&i| {
                let w = workers[i];
                let m = matched(i);
                let ratio = m as f64 / prompt_len as f64;
                let light_load = w.router_inflight_prefill_tokens < 4096
                    && w.total_load() < (w.max_num_seqs / 2).max(1);
                let deep_continuation =
                    m >= IDLE_SPREAD_MIN_MATCHED_TOKENS && ratio >= IDLE_SPREAD_MIN_MATCH_RATIO;
                let unique_past_harness = best_idle_match > 0
                    && m >= best_idle_match.saturating_add(MIN_PREFIX_BONUS_DELTA_TOKENS)
                    && ratio >= IDLE_CONTINUATION_MIN_MATCH_RATIO;
                light_load && (deep_continuation || unique_past_harness)
            })
            .max_by_key(|&i| matched(i));

        if let Some(best_busy) = best_busy_continuation
            && matched(best_busy) > best_idle_match.saturating_add(DP_PREFIX_BLOCK_TOKENS)
        {
            return Some(best_busy);
        }

        // Among multiple idle workers, if the best match is only a shallow shared harness
        // (<= 2K tokens, < 85% of the prompt) and another idle worker is completely cold
        // (0 matched tokens), warm the cold idle worker so staggered agents spread evenly.
        if idle_candidates.len() > 1
            && best_idle_match > 0
            && best_idle_match <= SHALLOW_HARNESS_MAX_TOKENS
            && (best_idle_match as f64 / prompt_len as f64) < 0.85
        {
            let cold_idle: Vec<usize> = idle_candidates
                .iter()
                .copied()
                .filter(|&i| matched(i) == 0)
                .collect();
            if !cold_idle.is_empty() {
                return pick_lowest_score(&cold_idle, n, ctx.tie_break_cursor, |idx| {
                    workers[idx].active_kv_util
                });
            }
        }

        // Otherwise prefer idle workers with the deepest prefix match, rotating ties.
        let best_idle: Vec<usize> = idle_candidates
            .into_iter()
            .filter(|&i| matched(i) == best_idle_match)
            .collect();
        return pick_lowest_score(&best_idle, n, ctx.tie_break_cursor, |idx| {
            workers[idx].active_kv_util
        });
    }

    // All admitted workers are busy: balance load + KV pressure against relative prefix advantage.
    let min_matched = candidates.iter().map(|&i| matched(i)).min().unwrap_or(0);
    pick_lowest_score(candidates, n, ctx.tie_break_cursor, |idx| {
        let delta_matched = matched(idx).saturating_sub(min_matched);
        let prefix_bonus = if delta_matched >= MIN_PREFIX_BONUS_DELTA_TOKENS {
            let len_factor = (delta_matched as f64 / PREFIX_LENGTH_SCALE_TOKENS)
                .min(1.0)
                .powi(2);
            let ratio_factor = (delta_matched as f64 / prompt_len as f64).min(1.0);
            2.5 * len_factor + ratio_factor
        } else {
            0.0
        };
        load_cost(workers[idx]) - prefix_bonus
    })
}

#[inline]
fn load_cost(w: WorkerLoadSample) -> f64 {
    let base = w.total_load() as f64;
    let prefill_work = w.router_inflight_prefill_tokens as f64 / PREFILL_TOKEN_LOAD_UNIT;
    let kv_over = (w.active_kv_util - 0.5).max(0.0);
    let kv_pressure = (w.waiting_requests as f64) * 6.0 * kv_over + 0.5 * kv_over;
    base + prefill_work + kv_pressure
}

fn pick_lowest_score(
    candidates: &[usize],
    num_workers: usize,
    tie_break_cursor: usize,
    score_fn: impl Fn(usize) -> f64,
) -> Option<usize> {
    const EPS: f64 = 1e-6;
    let mut best_score = f64::INFINITY;
    let mut best_idx = None;
    let n = candidates.len();
    if n == 0 || num_workers == 0 {
        return None;
    }
    let target_worker = tie_break_cursor % num_workers;
    let start = candidates
        .iter()
        .position(|&idx| idx >= target_worker)
        .unwrap_or(0);
    for offset in 0..n {
        let idx = candidates[(start + offset) % n];
        let s = score_fn(idx);
        if s + EPS < best_score {
            best_score = s;
            best_idx = Some(idx);
        }
    }
    best_idx
}

/// Bounded LRU session-affinity map (`session_key -> dp_rank`).
#[derive(Debug, Default)]
pub struct SessionAffinityMap {
    map: HashMap<String, usize>,
    order: VecDeque<String>,
}

impl SessionAffinityMap {
    pub fn get(&mut self, key: &str) -> Option<usize> {
        let rank = self.map.get(key).copied()?;
        if let Some(pos) = self.order.iter().position(|k| k == key)
            && let Some(item) = self.order.remove(pos)
        {
            self.order.push_back(item);
        }
        Some(rank)
    }

    pub fn bind(&mut self, key: String, dp_rank: usize) {
        if self.map.insert(key.clone(), dp_rank).is_some() {
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
            }
            self.order.push_back(key);
        } else {
            self.order.push_back(key);
            while self.order.len() > MAX_SESSION_AFFINITY_ENTRIES {
                if let Some(evicted) = self.order.pop_front() {
                    self.map.remove(&evicted);
                }
            }
        }
    }
}

/// Shared router state tracking per-worker in-flight requests, session affinity,
/// tie-breaking cursor, and exact [`DpPrefixDirectory`].
#[derive(Debug)]
pub struct DpRouterCore {
    pub policy: DpRoutingPolicy,
    pub prefix_dir: Arc<DpPrefixDirectory>,
    inflight: Vec<AtomicUsize>,
    inflight_prefill_tokens: Vec<AtomicUsize>,
    /// Short critical section serializing load read -> `select_dp_worker` -> atomic increment
    /// (after prefix hashing) so concurrent requests never race on stale `inflight == 0`.
    select_lock: Mutex<()>,
    sessions: Mutex<SessionAffinityMap>,
    cursor: AtomicUsize,
}

/// Idempotent RAII guard for a request's uncached prefill-token load.
///
/// Released by the worker as soon as prefill completes (first generated token or terminal
/// event), and also on drop so cancelled/aborted requests never leak prefill weight.
#[derive(Debug)]
pub struct PrefillTokenGuard {
    counter: Arc<DpRouterCore>,
    dp_rank: usize,
    remaining: AtomicUsize,
}

impl PrefillTokenGuard {
    /// Releases any remaining prefill-token load immediately (idempotent).
    #[inline]
    pub fn release(&self) {
        let prev = self.remaining.swap(0, Ordering::AcqRel);
        if prev > 0
            && let Some(slot) = self.counter.inflight_prefill_tokens.get(self.dp_rank)
        {
            slot.fetch_sub(prev, Ordering::AcqRel);
        }
    }
}

impl Drop for PrefillTokenGuard {
    fn drop(&mut self) {
        self.release();
    }
}

/// RAII guard that decrements a worker's `router_inflight` counter when dropped and
/// holds a handle to the request's [`PrefillTokenGuard`].
pub struct InflightGuard {
    counter: Arc<DpRouterCore>,
    dp_rank: usize,
    prefill: Option<Arc<PrefillTokenGuard>>,
}

impl InflightGuard {
    /// Returns the shared [`PrefillTokenGuard`] so the worker can release prefill-token
    /// load as soon as prefill finishes (before decode completes).
    #[inline]
    pub fn prefill_guard(&self) -> Option<Arc<PrefillTokenGuard>> {
        self.prefill.clone()
    }

    /// Releases the prefill-token weight immediately (idempotent).
    #[inline]
    pub fn release_prefill(&self) {
        if let Some(p) = &self.prefill {
            p.release();
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.release_prefill();
        if let Some(slot) = self.counter.inflight.get(self.dp_rank) {
            slot.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl DpRouterCore {
    pub fn new(num_workers: usize, policy: DpRoutingPolicy) -> Arc<Self> {
        let inflight = (0..num_workers).map(|_| AtomicUsize::new(0)).collect();
        let inflight_prefill_tokens = (0..num_workers).map(|_| AtomicUsize::new(0)).collect();
        Arc::new(Self {
            policy,
            prefix_dir: Arc::new(DpPrefixDirectory::new()),
            inflight,
            inflight_prefill_tokens,
            select_lock: Mutex::new(()),
            sessions: Mutex::new(SessionAffinityMap::default()),
            cursor: AtomicUsize::new(0),
        })
    }

    #[inline]
    pub fn inflight_count(&self, dp_rank: usize) -> usize {
        self.inflight
            .get(dp_rank)
            .map_or(0, |a| a.load(Ordering::Acquire))
    }

    #[inline]
    pub fn inflight_prefill_tokens(&self, dp_rank: usize) -> usize {
        self.inflight_prefill_tokens
            .get(dp_rank)
            .map_or(0, |a| a.load(Ordering::Acquire))
    }

    pub fn lookup_session(&self, session_key: &str) -> Option<usize> {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session_key)
    }

    pub fn bind_session(&self, session_key: String, dp_rank: usize) {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bind(session_key, dp_rank);
    }

    /// Selects a worker and atomically acquires an [`InflightGuard`] on it.
    pub fn select_and_acquire(
        self: &Arc<Self>,
        mut samples: Vec<WorkerLoadSample>,
        lora_id: i32,
        prompt_tokens: &[i32],
        session_key: Option<&str>,
    ) -> Option<(usize, InflightGuard)> {
        // Compute prefix matches outside `select_lock` so heavy prompts never block
        // concurrent routing decisions while hashing.
        let matched = if self.policy == DpRoutingPolicy::Prefix && !prompt_tokens.is_empty() {
            self.prefix_dir
                .match_prefix_tokens_per_worker(samples.len(), lora_id, prompt_tokens)
        } else {
            Vec::new()
        };
        let pinned_worker = session_key.and_then(|k| self.lookup_session(k));

        let (chosen, uncached_prompt_tokens) = {
            let _guard = self.select_lock.lock().unwrap_or_else(|e| e.into_inner());
            for (i, sample) in samples.iter_mut().enumerate() {
                sample.router_inflight = self.inflight_count(i);
                sample.router_inflight_prefill_tokens = self.inflight_prefill_tokens(i);
            }
            let cursor = self.cursor.fetch_add(1, Ordering::Relaxed);
            let chosen = select_dp_worker(
                &samples,
                &DpRouteContext {
                    policy: self.policy,
                    prompt_tokens: prompt_tokens.len(),
                    matched_tokens: &matched,
                    pinned_worker,
                    tie_break_cursor: cursor,
                },
            )?;
            let matched_on_chosen = matched.get(chosen).copied().unwrap_or(0);
            let uncached_prompt_tokens = prompt_tokens.len().saturating_sub(matched_on_chosen);
            self.inflight[chosen].fetch_add(1, Ordering::AcqRel);
            if uncached_prompt_tokens > 0 {
                self.inflight_prefill_tokens[chosen]
                    .fetch_add(uncached_prompt_tokens, Ordering::AcqRel);
            }
            (chosen, uncached_prompt_tokens)
        };

        // Preserve an existing healthy session binding on transient saturation spillover;
        // only bind when the session is new or its previous worker is unhealthy.
        if let Some(k) = session_key {
            let keep_existing = pinned_worker
                .is_some_and(|p| p != chosen && samples.get(p).is_some_and(|s| s.healthy));
            if !keep_existing {
                self.bind_session(k.to_owned(), chosen);
            }
        }

        let prefill = (uncached_prompt_tokens > 0).then(|| {
            Arc::new(PrefillTokenGuard {
                counter: Arc::clone(self),
                dp_rank: chosen,
                remaining: AtomicUsize::new(uncached_prompt_tokens),
            })
        });
        Some((
            chosen,
            InflightGuard {
                counter: Arc::clone(self),
                dp_rank: chosen,
                prefill,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle_worker() -> WorkerLoadSample {
        WorkerLoadSample {
            healthy: true,
            active_requests: 0,
            waiting_requests: 0,
            router_inflight: 0,
            router_inflight_prefill_tokens: 0,
            max_num_seqs: 16,
            active_kv_util: 0.0,
        }
    }

    fn busy_worker(active: usize, waiting: usize, kv_util: f64) -> WorkerLoadSample {
        WorkerLoadSample {
            healthy: true,
            active_requests: active,
            waiting_requests: waiting,
            router_inflight: 0,
            router_inflight_prefill_tokens: 0,
            max_num_seqs: 16,
            active_kv_util: kv_util,
        }
    }

    #[test]
    fn prefix_directory_insert_evict_and_lora_isolation() {
        let base = crate::lora::BASE_MODEL_LORA_ID;
        let dir = DpPrefixDirectory::new();
        let seq: Vec<i32> = (0..1024).collect();
        dir.record_insert(0, base, &seq);
        dir.record_insert(2, base, &seq[..512]);
        dir.record_insert(1, 7, &seq); // different LoRA adapter

        // Base model (-1): worker 0 matches 1024, worker 1 matches 0, worker 2 matches 512, worker 3 matches 0.
        assert_eq!(
            dir.match_prefix_tokens_per_worker(4, base, &seq),
            vec![1024, 0, 512, 0]
        );
        // LoRA 7: only worker 1 matches.
        assert_eq!(
            dir.match_prefix_tokens_per_worker(4, 7, &seq),
            vec![0, 1024, 0, 0]
        );

        // Evict the tail 512 tokens from worker 0 -> drops to 512.
        let all_hashes = compute_chained_prefix_hashes(base, &seq);
        dir.record_evict_hashes(0, &all_hashes[8..]);
        assert_eq!(
            dir.match_prefix_tokens_per_worker(4, base, &seq),
            vec![512, 0, 512, 0]
        );

        // Clear worker 2 -> worker 2 drops to 0 while worker 0 stays 512.
        dir.clear_worker(2);
        assert_eq!(
            dir.match_prefix_tokens_per_worker(4, base, &seq),
            vec![512, 0, 0, 0]
        );
    }

    #[test]
    fn staggered_three_agents_shared_2k_harness_spreads_on_dp2_and_pins_turns() {
        let dir = DpPrefixDirectory::new();
        let harness: Vec<i32> = (100..2148).collect(); // 2,048 tokens (32 blocks)

        // Turn 1 of Agent A (3K prompt): both GPUs idle -> GPU 0.
        let mut agent_a = harness.clone();
        agent_a.extend(10_000..11_024);
        let matched = dir.match_prefix_tokens_per_worker(2, 0, &agent_a);
        let w_a1 = select_dp_worker(
            &[idle_worker(), idle_worker()],
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: agent_a.len(),
                matched_tokens: &matched,
                pinned_worker: None,
                tie_break_cursor: 0,
            },
        );
        assert_eq!(w_a1, Some(0));
        dir.record_insert(0, 0, &agent_a);

        // Turn 1 of Agent B (3K prompt, shares 2K harness): GPU 0 is busy, GPU 1 is idle.
        // Idle-GPU spread guard MUST send Agent B to GPU 1 instead of piling onto GPU 0!
        let mut agent_b = harness.clone();
        agent_b.extend(20_000..21_024);
        let matched = dir.match_prefix_tokens_per_worker(2, 0, &agent_b);
        assert_eq!(matched, vec![2048, 0]);
        let w_b1 = select_dp_worker(
            &[busy_worker(1, 0, 0.05), idle_worker()],
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: agent_b.len(),
                matched_tokens: &matched,
                pinned_worker: None,
                tie_break_cursor: 1,
            },
        );
        assert_eq!(w_b1, Some(1));
        dir.record_insert(1, 0, &agent_b);

        // Turn 1 of Agent C (3K prompt, shares 2K harness): both GPUs now match 2048 -> tied on prefix,
        // routes to whichever GPU has lower load/KV.
        let mut agent_c = harness.clone();
        agent_c.extend(30_000..31_024);
        let matched = dir.match_prefix_tokens_per_worker(2, 0, &agent_c);
        assert_eq!(matched, vec![2048, 2048]);
        let w_c1 = select_dp_worker(
            &[busy_worker(2, 0, 0.15), busy_worker(1, 0, 0.05)],
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: agent_c.len(),
                matched_tokens: &matched,
                pinned_worker: None,
                tie_break_cursor: 2,
            },
        );
        assert_eq!(w_c1, Some(1));

        // Turn 2 of Agent A (10K prompt: 9K cached on GPU 0, only 2K on GPU 1):
        // Even if GPU 0 has 1 active request and GPU 1 is idle, Agent A's 90% / 9K continuation
        // stays pinned to GPU 0!
        agent_a.extend(11_024..18_000);
        dir.record_insert(0, 0, &agent_a);
        let mut agent_a_turn2 = agent_a.clone();
        agent_a_turn2.extend(18_000..18_512);
        let matched = dir.match_prefix_tokens_per_worker(2, 0, &agent_a_turn2);
        assert!(matched[0] >= 9000 && matched[1] == 2048);
        let w_a2 = select_dp_worker(
            &[busy_worker(1, 0, 0.10), idle_worker()],
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: agent_a_turn2.len(),
                matched_tokens: &matched,
                pinned_worker: None,
                tie_break_cursor: 3,
            },
        );
        assert_eq!(w_a2, Some(0));
    }

    #[test]
    fn dp3_and_dp4_spread_and_spillover_on_kv_saturation() {
        // 3× GPUs: 3 staggered agents spread across GPUs 0, 1, 2.
        let mut workers = vec![idle_worker(); 3];
        let w0 = select_dp_worker(
            &workers,
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: 3000,
                matched_tokens: &[0, 0, 0],
                pinned_worker: None,
                tie_break_cursor: 0,
            },
        );
        assert_eq!(w0, Some(0));
        workers[0] = busy_worker(1, 0, 0.05);

        let w1 = select_dp_worker(
            &workers,
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: 3000,
                matched_tokens: &[2048, 0, 0],
                pinned_worker: None,
                tie_break_cursor: 1,
            },
        );
        assert_eq!(w1, Some(1));
        workers[1] = busy_worker(1, 0, 0.05);

        let w2 = select_dp_worker(
            &workers,
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: 3000,
                matched_tokens: &[2048, 2048, 0],
                pinned_worker: None,
                tie_break_cursor: 2,
            },
        );
        assert_eq!(w2, Some(2));

        // 4× GPUs: GPU 0 has a huge prefix match (32K) AND a session pin, but is KV-saturated (0.95)
        // -> hard admission guard spills to the least-loaded healthy GPU (GPU 3).
        let dp4_workers = vec![
            busy_worker(2, 1, 0.95), // saturated
            busy_worker(2, 0, 0.40),
            busy_worker(2, 0, 0.50),
            busy_worker(1, 0, 0.20), // freest
        ];
        let spilled = select_dp_worker(
            &dp4_workers,
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: 34_000,
                matched_tokens: &[32_768, 2048, 2048, 2048],
                pinned_worker: Some(0),
                tie_break_cursor: 0,
            },
        );
        assert_eq!(spilled, Some(3));
    }

    #[test]
    fn inflight_guard_spreads_simultaneous_burst_before_telemetry_ticks() {
        let router = DpRouterCore::new(4, DpRoutingPolicy::Prefix);
        let prompt: Vec<i32> = (0..256).collect();
        let samples = vec![idle_worker(); 4];

        let mut guards = Vec::new();
        let mut chosen = Vec::new();
        for _ in 0..4 {
            let (idx, guard) = router
                .select_and_acquire(samples.clone(), 0, &prompt, None)
                .expect("worker");
            chosen.push(idx);
            guards.push(guard);
        }
        assert_eq!(chosen, vec![0, 1, 2, 3]);
        assert_eq!(router.inflight_count(0), 1);
        drop(guards);
        assert_eq!(router.inflight_count(0), 0);
    }

    #[test]
    fn mamba_radix_cache_observer_syncs_insert_evict_and_reset() {
        let base = crate::lora::BASE_MODEL_LORA_ID;
        let dir = Arc::new(DpPrefixDirectory::new());
        let mut cache0 = crate::radix::MambaRadixCache::new(16, 64);
        let mut cache1 = crate::radix::MambaRadixCache::new(16, 64);
        cache0.set_dp_prefix_observer(0, Arc::clone(&dir));
        cache1.set_dp_prefix_observer(1, Arc::clone(&dir));

        let seq_a: Vec<i32> = (0..256).collect();
        let pages_a: Vec<i32> = (0..16).collect();
        cache0.insert(&seq_a, &pages_a, &[], 0, 1);

        let mut seq_b = seq_a[..128].to_vec();
        seq_b.extend(500..628);
        let pages_b: Vec<i32> = (100..116).collect();
        cache1.insert(&seq_b, &pages_b, &[], 0, 2);

        assert_eq!(
            dir.match_prefix_tokens_per_worker(2, base, &seq_a),
            vec![256, 128]
        );
        assert_eq!(
            dir.match_prefix_tokens_per_worker(2, base, &seq_b),
            vec![128, 256]
        );

        // Evict all pages from cache0 -> worker 0's entries vanish while worker 1's stay intact.
        let _ = cache0.evict_full_pages(16);
        assert_eq!(
            dir.match_prefix_tokens_per_worker(2, base, &seq_a),
            vec![0, 128]
        );
        assert_eq!(
            dir.match_prefix_tokens_per_worker(2, base, &seq_b),
            vec![0, 256]
        );

        // LoRA insert + evict also matches raw unsalted tokens queried with that lora_id.
        cache0.active_lora_id = 3;
        cache0.insert(&seq_a, &pages_a, &[], 0, 3);
        assert_eq!(
            dir.match_prefix_tokens_per_worker(2, 3, &seq_a),
            vec![256, 0]
        );
        assert_eq!(
            dir.match_prefix_tokens_per_worker(2, base, &seq_a),
            vec![0, 128]
        );
        let _ = cache0.evict_full_pages(16);
        assert_eq!(dir.match_prefix_tokens_per_worker(2, 3, &seq_a), vec![0, 0]);

        // Reset cache1 -> directory becomes completely empty.
        cache1.reset();
        assert_eq!(dir.tracked_block_count(), 0);
    }

    #[test]
    fn heterogeneous_heavy_prefill_deflects_short_requests_to_other_gpu() {
        let router = DpRouterCore::new(2, DpRoutingPolicy::Prefix);
        let heavy_prompt: Vec<i32> = (0..24_576).collect();
        let short_prompt: Vec<i32> = (50_000..50_064).collect();
        let samples = vec![idle_worker(); 2];

        // Heavy 24K prompt lands on worker 0 (adds ~12 sequence-slots of prefill work).
        let (w_heavy, g_heavy) = router
            .select_and_acquire(
                samples.clone(),
                crate::lora::BASE_MODEL_LORA_ID,
                &heavy_prompt,
                None,
            )
            .unwrap();
        assert_eq!(w_heavy, 0);
        assert_eq!(router.inflight_prefill_tokens(0), 24_576);

        // Next 8 short requests should ALL deflect to worker 1 rather than stalling behind
        // worker 0's 24K prefill!
        let mut short_guards = Vec::new();
        for _ in 0..8 {
            let (w, g) = router
                .select_and_acquire(
                    samples.clone(),
                    crate::lora::BASE_MODEL_LORA_ID,
                    &short_prompt,
                    None,
                )
                .unwrap();
            assert_eq!(
                w, 1,
                "short request must not stall behind 24K prefill on GPU 0"
            );
            short_guards.push(g);
        }

        // Dropping all guards resets both inflight counts and prefill token counters to 0.
        drop(short_guards);
        drop(g_heavy);
        assert_eq!(router.inflight_count(0), 0);
        assert_eq!(router.inflight_count(1), 0);
        assert_eq!(router.inflight_prefill_tokens(0), 0);
        assert_eq!(router.inflight_prefill_tokens(1), 0);
    }

    #[test]
    fn total_load_does_not_double_count_waiting_and_prefill_guard_releases_before_decode() {
        let w = WorkerLoadSample {
            healthy: true,
            active_requests: 12,
            waiting_requests: 4,
            router_inflight: 16,
            router_inflight_prefill_tokens: 0,
            max_num_seqs: 16,
            active_kv_util: 0.2,
        };
        assert_eq!(w.total_load(), 16);

        let router = DpRouterCore::new(2, DpRoutingPolicy::Prefix);
        let heavy: Vec<i32> = (0..16_384).collect();
        let (rank, guard) = router
            .select_and_acquire(
                vec![idle_worker(); 2],
                crate::lora::BASE_MODEL_LORA_ID,
                &heavy,
                None,
            )
            .unwrap();
        assert_eq!(rank, 0);
        assert_eq!(router.inflight_count(0), 1);
        assert_eq!(router.inflight_prefill_tokens(0), 16_384);

        // Releasing prefill guard clears prefill weight while request stays in decode.
        let pg = guard.prefill_guard().unwrap();
        pg.release();
        pg.release(); // idempotent
        assert_eq!(router.inflight_prefill_tokens(0), 0);
        assert_eq!(router.inflight_count(0), 1);

        drop(guard);
        assert_eq!(router.inflight_count(0), 0);
    }

    #[test]
    fn both_idle_shallow_harness_warms_cold_gpu_and_unique_continuation_stays_pinned() {
        // Both GPUs idle: GPU 0 has 2K harness cached, GPU 1 is cold (0).
        // New agent Turn 0 (3K prompt, only 2K harness match < 85%) warms cold GPU 1!
        let workers_both_idle = vec![idle_worker(), idle_worker()];
        let turn0_new_agent = select_dp_worker(
            &workers_both_idle,
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: 3000,
                matched_tokens: &[2048, 0],
                pinned_worker: None,
                tie_break_cursor: 0,
            },
        );
        assert_eq!(turn0_new_agent, Some(1));

        // Now both GPUs have the 2K harness; GPU 0 is lightly busy (1 decode, 0 prefill)
        // with Agent A's 2.8K Turn-2 history while GPU 1 is idle (2K harness only).
        // Agent A Turn 2 (3.2K prompt, 2.8K matched on GPU 0 = 87.5%) stays pinned on GPU 0!
        let workers_one_busy = vec![busy_worker(1, 0, 0.1), idle_worker()];
        let turn2_agent_a = select_dp_worker(
            &workers_one_busy,
            &DpRouteContext {
                policy: DpRoutingPolicy::Prefix,
                prompt_tokens: 3200,
                matched_tokens: &[2816, 2048],
                pinned_worker: None,
                tie_break_cursor: 1,
            },
        );
        assert_eq!(turn2_agent_a, Some(0));
    }
}
