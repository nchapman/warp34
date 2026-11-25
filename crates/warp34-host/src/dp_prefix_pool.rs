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

    /// Fast O(L) check whether `(lora_id, tokens)` is already stored in the pool, touching
    /// its LRU position if found so callers can skip redundant D2H copies.
    pub fn contains_and_touch(&self, lora_id: i32, tokens: &[i32]) -> bool {
        if self.max_bytes == 0 || tokens.len() < DP_L2_MIN_PREFIX_TOKENS || self.is_empty() {
            return false;
        }
        let key = PoolKey {
            lora_id,
            len: tokens.len(),
            hash: hash_prefix_tokens(lora_id, tokens),
        };
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let found = st
            .entries
            .get(&key)
            .is_some_and(|e| e.token_ids.as_slice() == tokens);
        if found {
            if let Some(pos) = st.lru.iter().position(|k| *k == key) {
                st.lru.remove(pos);
            }
            st.lru.push_back(key);
        }
        found
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
            if let Some(pos) = st.lru.iter().position(|k| *k == key) {
                st.lru.remove(pos);
            }
            st.lru.push_back(key);
            return false;
        }

        // Evict LRU entries until there is room for `entry.total_bytes`.
        let mut evicted = 0u64;
        while st.used_bytes.saturating_add(entry.total_bytes) > self.max_bytes {
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
}
