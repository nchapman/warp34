//! Unified in-process L2 Host-RAM prefix pool shared across Data Parallel (`--dp`) replicas.
//!
//! Because all DP replicas in `warp34 serve --dp N` live in the same OS process with identical
//! `tp_size`, `KvLayerLayout`, and `GdnCheckpointLayout`, a prefix prefilled on GPU 0 (KV pages
//! and compressed WHT-128 Q8 GDN checkpoint) can be published once into host RAM and hydrated
//! directly into any other replica's GPU VRAM over PCIe H2D DMA — turning cross-GPU spillover
//! and cold-GPU harness warmup from a full GPU re-prefill into a ~2 ms DMA transfer.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Minimum prefix length (in tokens) eligible for cross-replica L2 publishing and hydration.
pub const DP_L2_MIN_PREFIX_TOKENS: usize = 256;

/// Shared harness threshold: prefixes up to this length are never superseded by longer
/// single-agent continuations so newly arriving agents always hit the shared system prompt.
const SHARED_HARNESS_KEEP_TOKENS: usize = 4096;

/// Default hard cap on host RAM consumed by [`SharedDpPrefixPool`] (`16 GiB`).
const DEFAULT_MAX_POOL_BYTES: usize = 16 * 1024 * 1024 * 1024;

/// Compressed GDN/Mamba checkpoint frame (`WHT128-Q8_0`) for one TP rank.
#[derive(Clone, Debug)]
pub struct SharedGdnCheckpoint {
    pub frame: Vec<u8>,
    pub ple_ngram_context: [i32; 2],
}

/// Immutable snapshot of a committed prefix (KV pages + optional GDN checkpoint) across all
/// TP ranks of a DP replica.
#[derive(Debug)]
pub struct SharedPrefixEntry {
    pub lora_id: i32,
    pub token_ids: Vec<i32>,
    pub num_pages: usize,
    pub kv_ranks: Vec<Vec<u8>>,
    pub ckpt_ranks: Vec<SharedGdnCheckpoint>,
    /// Optional speculative proposer prefix memo (`MTP` / `DFlash2` hidden features in host RAM).
    pub proposer_features: Option<Vec<u8>>,
    pub total_bytes: usize,
    pub source_dp_rank: usize,
}

impl SharedPrefixEntry {
    pub fn new(
        lora_id: i32,
        token_ids: Vec<i32>,
        num_pages: usize,
        kv_ranks: Vec<Vec<u8>>,
        ckpt_ranks: Vec<SharedGdnCheckpoint>,
        proposer_features: Option<Vec<u8>>,
        source_dp_rank: usize,
    ) -> Self {
        let kv_bytes: usize = kv_ranks.iter().map(Vec::len).sum();
        let ckpt_bytes: usize = ckpt_ranks.iter().map(|c| c.frame.len()).sum();
        let feat_bytes = proposer_features.as_ref().map_or(0, Vec::len);
        let token_bytes = token_ids.len() * std::mem::size_of::<i32>();
        Self {
            lora_id,
            token_ids,
            num_pages,
            kv_ranks,
            ckpt_ranks,
            proposer_features,
            total_bytes: kv_bytes + ckpt_bytes + feat_bytes + token_bytes,
            source_dp_rank,
        }
    }
}

/// Snapshot of [`SharedDpPrefixPool`] utilization and hit counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedDpPrefixPoolStats {
    pub entries: usize,
    pub used_bytes: usize,
    pub max_bytes: usize,
    pub publishes: u64,
    pub hits: u64,
    pub hydrated_tokens: u64,
    pub evictions: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PoolKey {
    lora_id: i32,
    len: usize,
    hash: u64,
}

#[derive(Debug)]
struct PoolState {
    entries: HashMap<PoolKey, Arc<SharedPrefixEntry>>,
    /// Distinct token lengths currently present in `entries` -> count of entries at that length.
    /// Allows `lookup_longest_match` to probe only lengths that actually exist in the pool.
    lengths: BTreeMap<usize, usize>,
    /// Approximate LRU eviction queue (front = oldest, back = most recently used/inserted).
    lru: VecDeque<PoolKey>,
    used_bytes: usize,
}

/// Thread-safe, memory-bounded in-process L2 prefix cache shared across all DP replicas.
#[derive(Debug)]
pub struct SharedDpPrefixPool {
    max_bytes: usize,
    page_size: usize,
    state: Mutex<PoolState>,
    entry_count: AtomicUsize,
    publishes: AtomicU64,
    hits: AtomicU64,
    hydrated_tokens: AtomicU64,
    evictions: AtomicU64,
}

/// Computes the cumulative page-aligned FNV-1a hash of `(lora_id, tokens)`.
#[inline]
pub fn hash_prefix_tokens(lora_id: i32, tokens: &[i32]) -> u64 {
    let mut h = seed_for_lora(lora_id);
    for &t in tokens {
        h = fnv_mix_i32(h, t);
    }
    h
}

#[inline]
fn seed_for_lora(lora_id: i32) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    fnv_mix_i32(FNV_OFFSET, lora_id)
}

#[inline]
fn fnv_mix_i32(mut h: u64, val: i32) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for b in val.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

impl SharedDpPrefixPool {
    /// Creates a new pool with explicit `max_bytes` capacity and `page_size`.
    pub fn new(max_bytes: usize, page_size: usize) -> Self {
        Self {
            max_bytes,
            page_size: page_size.max(1),
            state: Mutex::new(PoolState {
                entries: HashMap::new(),
                lengths: BTreeMap::new(),
                lru: VecDeque::new(),
                used_bytes: 0,
            }),
            entry_count: AtomicUsize::new(0),
            publishes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            hydrated_tokens: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Auto-sizes the pool from available host RAM (`min(avail / 4, 16 GiB)`), honoring
    /// `WARP34_DP_L2_POOL_MB` (`0` disables the pool and returns `None`).
    pub fn auto_sized(page_size: usize) -> Option<Arc<Self>> {
        if let Ok(raw) = std::env::var("WARP34_DP_L2_POOL_MB")
            && let Ok(mb) = raw.trim().parse::<usize>()
        {
            if mb == 0 {
                return None;
            }
            return Some(Arc::new(Self::new(
                mb.saturating_mul(1024 * 1024),
                page_size,
            )));
        }
        let avail = crate::cache::detect_available_host_ram_bytes().unwrap_or(0);
        let budget = (avail / 4).min(DEFAULT_MAX_POOL_BYTES);
        if budget < 64 * 1024 * 1024 {
            return None;
        }
        Some(Arc::new(Self::new(budget, page_size)))
    }

    #[inline]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entry_count.load(Ordering::Relaxed) == 0
    }

    /// Fast O(L) check whether `(lora_id, tokens)` is already stored in the pool (and, when
    /// `require_features` is `true`, carries speculative proposer features), touching its LRU
    /// position if found so callers can skip redundant D2H copies.
    pub fn contains_and_touch_with_features(
        &self,
        lora_id: i32,
        tokens: &[i32],
        require_features: bool,
    ) -> bool {
        if self.max_bytes == 0 || tokens.len() < DP_L2_MIN_PREFIX_TOKENS || self.is_empty() {
            return false;
        }
        let key = PoolKey {
            lora_id,
            len: tokens.len(),
            hash: hash_prefix_tokens(lora_id, tokens),
        };
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let found = st.entries.get(&key).is_some_and(|e| {
            e.token_ids.as_slice() == tokens && (!require_features || e.proposer_features.is_some())
        });
        if found {
            if let Some(pos) = st.lru.iter().position(|k| *k == key) {
                st.lru.remove(pos);
            }
            st.lru.push_back(key);
        }
        found
    }

    /// Fast O(L) check whether `(lora_id, tokens)` is already stored in the pool, touching
    /// its LRU position if found so callers can skip redundant D2H copies.
    pub fn contains_and_touch(&self, lora_id: i32, tokens: &[i32]) -> bool {
        self.contains_and_touch_with_features(lora_id, tokens, false)
    }

    /// Returns `true` if the pool already holds a prefix of `tokens` (under `lora_id`) whose
    /// length is within `max_delta` tokens of `tokens.len()`. Used to throttle redundant
    /// synchronous D2H publishes on every short turn of a pinned multi-turn agent.
    pub fn has_recent_prefix_within(&self, lora_id: i32, tokens: &[i32], max_delta: usize) -> bool {
        if self.max_bytes == 0 || tokens.len() <= max_delta || self.is_empty() {
            return false;
        }
        let min_len = tokens.len() - max_delta;
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let candidate_lens: Vec<usize> = st
            .lengths
            .range(min_len..=tokens.len())
            .map(|(&len, _)| len)
            .collect();
        if candidate_lens.is_empty() {
            return false;
        }
        let mut h = seed_for_lora(lora_id);
        let mut cursor = 0usize;
        for len in candidate_lens {
            for &t in &tokens[cursor..len] {
                h = fnv_mix_i32(h, t);
            }
            cursor = len;
            let key = PoolKey {
                lora_id,
                len,
                hash: h,
            };
            if let Some(entry) = st.entries.get(&key)
                && entry.token_ids.as_slice() == &tokens[..len]
            {
                return true;
            }
        }
        false
    }

    fn evict_lru_until_fits(&self, st: &mut PoolState, incoming_bytes: usize) {
        let mut evicted = 0u64;
        while st.used_bytes.saturating_add(incoming_bytes) > self.max_bytes {
            let Some(victim_key) = st.lru.pop_front() else {
                break;
            };
            if let Some(victim) = st.entries.remove(&victim_key) {
                st.used_bytes = st.used_bytes.saturating_sub(victim.total_bytes);
                if let Some(cnt) = st.lengths.get_mut(&victim_key.len) {
                    *cnt = cnt.saturating_sub(1);
                    if *cnt == 0 {
                        st.lengths.remove(&victim_key.len);
                    }
                }
                evicted += 1;
            }
        }
        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
        }
    }

    /// Inserts `entry` into the pool, evicting LRU entries if needed to stay within `max_bytes`.
    /// Entries larger than `max_bytes / 2` or shorter than [`DP_L2_MIN_PREFIX_TOKENS`] are
    /// ignored so a single oversized prompt cannot thrash the shared pool.
    pub fn insert(&self, entry: SharedPrefixEntry) -> bool {
        if self.max_bytes == 0
            || entry.token_ids.len() < DP_L2_MIN_PREFIX_TOKENS
            || !entry.token_ids.len().is_multiple_of(self.page_size)
            || entry.total_bytes == 0
            || entry.total_bytes > self.max_bytes / 2
        {
            return false;
        }
        let key = PoolKey {
            lora_id: entry.lora_id,
            len: entry.token_ids.len(),
            hash: hash_prefix_tokens(entry.lora_id, &entry.token_ids),
        };
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = st.entries.get(&key)
            && existing.token_ids == entry.token_ids
        {
            if existing.proposer_features.is_none() && entry.proposer_features.is_some() {
                // Upgrade a plain-mode entry with speculative proposer features in place.
                let old_bytes = existing.total_bytes;
                st.used_bytes = st.used_bytes.saturating_sub(old_bytes);
                if let Some(pos) = st.lru.iter().position(|k| *k == key) {
                    st.lru.remove(pos);
                }
                self.evict_lru_until_fits(&mut st, entry.total_bytes);
                st.used_bytes = st.used_bytes.saturating_add(entry.total_bytes);
                st.entries.insert(key, Arc::new(entry));
                st.lru.push_back(key);
                self.entry_count.store(st.entries.len(), Ordering::Relaxed);
                self.publishes.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            if let Some(pos) = st.lru.iter().position(|k| *k == key) {
                st.lru.remove(pos);
            }
            st.lru.push_back(key);
            return false;
        }

        // Supersede any strictly shorter trajectory-private checkpoint (> 4096 tokens) of the
        // same agent so multi-turn continuations keep only their latest L2 snapshot instead of
        // piling up stale intermediate turns, while preserving shared harnesses (<= 4096).
        if entry.token_ids.len() > SHARED_HARNESS_KEEP_TOKENS {
            let shorter_lens: Vec<usize> = st
                .lengths
                .range((SHARED_HARNESS_KEEP_TOKENS + 1)..entry.token_ids.len())
                .map(|(&len, _)| len)
                .collect();
            if !shorter_lens.is_empty() {
                let mut h = seed_for_lora(entry.lora_id);
                let mut cursor = 0usize;
                let mut stale_keys = Vec::new();
                for len in shorter_lens {
                    for &t in &entry.token_ids[cursor..len] {
                        h = fnv_mix_i32(h, t);
                    }
                    cursor = len;
                    let sk = PoolKey {
                        lora_id: entry.lora_id,
                        len,
                        hash: h,
                    };
                    if let Some(old) = st.entries.get(&sk)
                        && old.token_ids.as_slice() == &entry.token_ids[..len]
                    {
                        stale_keys.push(sk);
                    }
                }
                for sk in stale_keys {
                    if let Some(old) = st.entries.remove(&sk) {
                        st.used_bytes = st.used_bytes.saturating_sub(old.total_bytes);
                        if let Some(cnt) = st.lengths.get_mut(&sk.len) {
                            *cnt = cnt.saturating_sub(1);
                            if *cnt == 0 {
                                st.lengths.remove(&sk.len);
                            }
                        }
                        if let Some(pos) = st.lru.iter().position(|k| *k == sk) {
                            st.lru.remove(pos);
                        }
                    }
                }
            }
        }

        // Evict LRU entries until there is room for `entry.total_bytes`.
        self.evict_lru_until_fits(&mut st, entry.total_bytes);

        if let Some(prev) = st.entries.insert(key, Arc::new(entry)) {
            st.used_bytes = st.used_bytes.saturating_sub(prev.total_bytes);
            if let Some(pos) = st.lru.iter().position(|k| *k == key) {
                st.lru.remove(pos);
            }
        } else {
            *st.lengths.entry(key.len).or_insert(0) += 1;
        }
        let added = st.entries[&key].total_bytes;
        st.used_bytes += added;
        st.lru.push_back(key);
        self.entry_count.store(st.entries.len(), Ordering::Relaxed);
        self.publishes.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Finds the longest cached prefix in the pool strictly longer than `min_tokens` that
    /// matches `tokens` exactly under `lora_id`. Returns `None` without locking if the pool
    /// is empty or `tokens.len() <= min_tokens`.
    pub fn lookup_longest_match(
        &self,
        lora_id: i32,
        tokens: &[i32],
        min_tokens: usize,
    ) -> Option<Arc<SharedPrefixEntry>> {
        let floor = min_tokens.max(DP_L2_MIN_PREFIX_TOKENS.saturating_sub(1));
        if self.max_bytes == 0 || tokens.len() <= floor || self.is_empty() {
            return None;
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let max_candidate = tokens.len();
        let candidate_lens: Vec<usize> = st
            .lengths
            .range((floor + 1)..=max_candidate)
            .map(|(&len, _)| len)
            .collect();
        if candidate_lens.is_empty() {
            return None;
        }

        // Single rolling FNV-1a pass over `tokens[..max_probe_len]` collecting hashes at each
        // present length, then checking longest-first with exact slice equality.
        let mut h = seed_for_lora(lora_id);
        let mut cursor = 0usize;
        let mut hashed_candidates: Vec<(usize, u64)> = Vec::with_capacity(candidate_lens.len());
        for len in candidate_lens {
            for &t in &tokens[cursor..len] {
                h = fnv_mix_i32(h, t);
            }
            cursor = len;
            hashed_candidates.push((len, h));
        }

        for (len, hash) in hashed_candidates.into_iter().rev() {
            let key = PoolKey { lora_id, len, hash };
            if let Some(entry) = st.entries.get(&key)
                && entry.token_ids.as_slice() == &tokens[..len]
            {
                let cloned = Arc::clone(entry);
                if let Some(pos) = st.lru.iter().position(|k| *k == key) {
                    st.lru.remove(pos);
                }
                st.lru.push_back(key);
                return Some(cloned);
            }
        }
        None
    }

    /// Non-mutating probe returning `(matched_tokens, has_proposer_features)` for the longest
    /// cached prefix strictly longer than `min_tokens`, without touching LRU order or hit stats.
    pub fn peek_longest_match(
        &self,
        lora_id: i32,
        tokens: &[i32],
        min_tokens: usize,
    ) -> Option<(usize, bool)> {
        let floor = min_tokens.max(DP_L2_MIN_PREFIX_TOKENS.saturating_sub(1));
        if self.max_bytes == 0 || tokens.len() <= floor || self.is_empty() {
            return None;
        }
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let max_candidate = tokens.len();
        let candidate_lens: Vec<usize> = st
            .lengths
            .range((floor + 1)..=max_candidate)
            .map(|(&len, _)| len)
            .collect();
        if candidate_lens.is_empty() {
            return None;
        }

        let mut h = seed_for_lora(lora_id);
        let mut cursor = 0usize;
        let mut hashed_candidates: Vec<(usize, u64)> = Vec::with_capacity(candidate_lens.len());
        for len in candidate_lens {
            for &t in &tokens[cursor..len] {
                h = fnv_mix_i32(h, t);
            }
            cursor = len;
            hashed_candidates.push((len, h));
        }

        for (len, hash) in hashed_candidates.into_iter().rev() {
            let key = PoolKey { lora_id, len, hash };
            if let Some(entry) = st.entries.get(&key)
                && entry.token_ids.as_slice() == &tokens[..len]
            {
                return Some((len, entry.proposer_features.is_some()));
            }
        }
        None
    }

    /// Records a successful cross-replica hydration in pool metrics.
    pub fn record_hydration(&self, hydrated_tokens: usize) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.hydrated_tokens
            .fetch_add(hydrated_tokens as u64, Ordering::Relaxed);
    }

    pub fn stats(&self) -> SharedDpPrefixPoolStats {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        SharedDpPrefixPoolStats {
            entries: st.entries.len(),
            used_bytes: st.used_bytes,
            max_bytes: self.max_bytes,
            publishes: self.publishes.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            hydrated_tokens: self.hydrated_tokens.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(lora_id: i32, tokens: Vec<i32>, kv_bytes: usize) -> SharedPrefixEntry {
        let num_pages = tokens.len() / 16;
        SharedPrefixEntry::new(
            lora_id,
            tokens,
            num_pages,
            vec![vec![0xAB; kv_bytes]],
            vec![SharedGdnCheckpoint {
                frame: vec![0xCD; 64],
                ple_ngram_context: [1, 2],
            }],
            None,
            0,
        )
    }

    #[test]
    fn publish_lookup_longest_match_lora_isolation_and_lru_eviction() {
        // Capacity = 8 KiB -> max single entry = 4 KiB.
        let pool = SharedDpPrefixPool::new(8192, 16);
        let seq_256: Vec<i32> = (1..=256).collect();
        let seq_512: Vec<i32> = (1..=512).collect();

        assert!(pool.insert(make_entry(-1, seq_256.clone(), 1024)));
        assert!(pool.insert(make_entry(-1, seq_512.clone(), 1024)));
        assert!(pool.contains_and_touch(-1, &seq_256));

        // Querying a 640-token continuation matches the longest 512-token entry.
        let mut query = seq_512.clone();
        query.extend(1000..1128);
        let hit = pool
            .lookup_longest_match(-1, &query, 0)
            .expect("should match 512");
        assert_eq!(hit.token_ids.len(), 512);

        // If local cache already has 512 tokens, min_tokens=512 skips hydration cleanly.
        assert!(pool.lookup_longest_match(-1, &query, 512).is_none());

        // LoRA isolation: lora_id=7 must not match base (-1) entries.
        assert!(pool.lookup_longest_match(7, &query, 0).is_none());

        // LRU eviction: insert two 3 KiB entries -> evicts oldest (seq_256 first after seq_512 touch).
        let seq_a: Vec<i32> = (2000..2256).collect();
        let seq_b: Vec<i32> = (3000..3256).collect();
        assert!(pool.insert(make_entry(-1, seq_a.clone(), 3000)));
        assert!(pool.insert(make_entry(-1, seq_b.clone(), 3000)));
        let st = pool.stats();
        assert!(st.used_bytes <= 8192);
        assert!(st.evictions >= 1);
        assert!(pool.lookup_longest_match(-1, &seq_b, 0).is_some());
    }

    #[test]
    fn supersedes_stale_agent_turns_while_preserving_shared_harness_and_stride_check() {
        let pool = SharedDpPrefixPool::new(1024 * 1024, 16);
        let harness_4k: Vec<i32> = (1..=4096).collect();
        let mut turn1_6k = harness_4k.clone();
        turn1_6k.extend(10_000..12_048); // 6,144 tokens
        let mut turn2_8k = turn1_6k.clone();
        turn2_8k.extend(20_000..22_048); // 8,192 tokens
        let mut turn3_11k = turn2_8k.clone();
        turn3_11k.extend(30_000..33_072); // 11,264 tokens

        assert!(pool.insert(make_entry(-1, harness_4k.clone(), 1024)));
        assert!(pool.insert(make_entry(-1, turn1_6k.clone(), 1024)));
        assert_eq!(pool.stats().entries, 2);

        // turn2_8k is within 4096 tokens of turn1_6k -> stride throttle returns true.
        assert!(pool.has_recent_prefix_within(-1, &turn2_8k, 4096));
        // turn3_11k is > 4096 tokens past turn1_6k -> stride throttle returns false.
        assert!(!pool.has_recent_prefix_within(-1, &turn3_11k, 4096));

        // Publishing turn3_11k supersedes turn1_6k (> 4096) while keeping harness_4k (<= 4096).
        assert!(pool.insert(make_entry(-1, turn3_11k.clone(), 1024)));
        assert_eq!(pool.stats().entries, 2);
        assert!(pool.contains_and_touch(-1, &harness_4k));
        assert!(!pool.contains_and_touch(-1, &turn1_6k));
        assert!(pool.contains_and_touch(-1, &turn3_11k));

        // Upgrading a plain-mode entry (proposer_features: None) with proposer_features: Some(...)
        // replaces the entry in place and satisfies `contains_and_touch_with_features(..., true)`.
        assert!(!pool.contains_and_touch_with_features(-1, &harness_4k, true));
        let upgraded = SharedPrefixEntry::new(
            -1,
            harness_4k.clone(),
            harness_4k.len() / 16,
            vec![vec![0xAB; 1024]],
            vec![SharedGdnCheckpoint {
                frame: vec![0xCD; 64],
                ple_ngram_context: [1, 2],
            }],
            Some(vec![0xEF; 512]),
            0,
        );
        assert!(pool.insert(upgraded));
        assert_eq!(pool.stats().entries, 2);
        assert!(pool.contains_and_touch_with_features(-1, &harness_4k, true));
        let hit = pool
            .lookup_longest_match(-1, &harness_4k, 0)
            .expect("upgraded harness should match");
        assert_eq!(hit.proposer_features.as_deref(), Some(&[0xEF; 512][..]));
    }
}
