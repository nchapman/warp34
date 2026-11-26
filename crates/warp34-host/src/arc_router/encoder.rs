//! Dedicated single-thread ARC-1.0 CUDA encoder worker sharing the host process.
//!
//! Runs `nchapman/ARC-1.0-W4A16-HGPTQ` (12 layers, 1024 hidden dim, W4A16-HGPTQ, no `lm_head`,
//! offloaded host `embed_tokens` + 96 MiB scratch arena) with:
//! - Per-block readout extraction + head+tail 32K context windowing
//! - Multi-session LRU KV+GDN prefix cache (16 resident sessions sharing a 4,096-page KV pool)
//!   so concurrent agent workers only prefill newly appended turns (~3–10 ms)
//! - Bounded per-block tokenization memo (`>= 256 B`, 32 MiB cap)

#[cfg(any(feature = "cuda", test))]
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

use super::input::ArcRoleBlock;

const ENCODER_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// 4,096 KV pages × 16 tokens/page = 65,536 total tokens across up to 16 concurrent sessions.
#[cfg(any(feature = "cuda", test))]
const ENCODER_KV_PAGES: usize = 4096;
/// Max tokens per individual session (leaves headroom for concurrent sessions in the pool).
#[cfg(feature = "cuda")]
const ENCODER_PER_SESSION_MAX_CTX: usize = 32_768;
/// Resident LRU GDN+KV sessions (2× headroom over typical 2-card 8-worker concurrency; ~24 MiB extra GDN state).
#[cfg(any(feature = "cuda", test))]
const ENCODER_MAX_SESSIONS: usize = 16;
/// Cap per-block tokenization memo at ~32 MiB of serialized text + token IDs.
#[cfg(feature = "cuda")]
const BLOCK_MEMO_MAX_BYTES: usize = 32 * 1024 * 1024;
#[cfg(feature = "cuda")]
const BLOCK_MEMO_MIN_TEXT_BYTES: usize = 256;

/// Encoder output for one request: one `1024`-dim `f32` readout per retained role block.
#[derive(Debug, Clone)]
pub struct EncodeResponse {
    pub block_readouts: Vec<Vec<f32>>,
    pub block_roles: Vec<String>,
    pub prompt_tokens: usize,
}

impl EncodeResponse {
    /// Returns the final block's readout vector (used by Stage 1 and PII heads).
    pub fn decision_readout(&self) -> &[f32] {
        self.block_readouts.last().map_or(&[], Vec::as_slice)
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct EncodeJob {
    blocks: Vec<ArcRoleBlock>,
    reply: tokio::sync::oneshot::Sender<Result<EncodeResponse, String>>,
}

/// Thread-safe handle to the background ARC-1.0 GPU encoder worker.
pub struct ArcEncoderWorker {
    tx: mpsc::SyncSender<EncodeJob>,
}

/// Splits literal reserved vision tokens inside block content (`<|vision_start|>` etc.) so
/// source code mentioning them tokenizes as ordinary text rather than special vision IDs.
#[cfg(feature = "cuda")]
fn tokenize_block_safe(
    engine: &crate::model::Qwen3_5Engine,
    block: &ArcRoleBlock,
) -> Result<Vec<i32>, String> {
    const VISION_LITERALS: [&str; 5] = [
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|video_pad|>",
    ];
    let serialized = block.serialized();
    if !VISION_LITERALS.iter().any(|pat| serialized.contains(pat)) {
        return engine
            .encode(&serialized)
            .map_err(|e| format!("ARC tokenize failed: {e}"));
    }
    // Cut right after `<|` whenever a vision literal appears so each half tokenizes as text.
    let mut out = Vec::new();
    let mut rest = serialized.as_str();
    while !rest.is_empty() {
        let next_hit = VISION_LITERALS
            .iter()
            .filter_map(|pat| rest.find(pat))
            .min();
        let Some(pos) = next_hit else {
            out.extend_from_slice(
                &engine
                    .encode(rest)
                    .map_err(|e| format!("ARC tokenize failed: {e}"))?,
            );
            break;
        };
        let split = pos + 2; // right after `<|`
        out.extend_from_slice(
            &engine
                .encode(&rest[..split])
                .map_err(|e| format!("ARC tokenize failed: {e}"))?,
        );
        rest = &rest[split..];
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
struct BlockTokenMemo {
    entries: HashMap<String, Vec<i32>>,
    approx_bytes: usize,
}

#[cfg(feature = "cuda")]
impl BlockTokenMemo {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            approx_bytes: 0,
        }
    }

    fn tokenize(
        &mut self,
        engine: &crate::model::Qwen3_5Engine,
        block: &ArcRoleBlock,
    ) -> Result<Vec<i32>, String> {
        let key = block.serialized();
        if key.len() < BLOCK_MEMO_MIN_TEXT_BYTES {
            return tokenize_block_safe(engine, block);
        }
        if let Some(cached) = self.entries.get(&key) {
            return Ok(cached.clone());
        }
        let toks = tokenize_block_safe(engine, block)?;
        let entry_bytes = key.len() + toks.len() * 4;
        if entry_bytes <= BLOCK_MEMO_MAX_BYTES / 4 {
            if self.approx_bytes + entry_bytes > BLOCK_MEMO_MAX_BYTES {
                self.entries.clear();
                self.approx_bytes = 0;
            }
            self.approx_bytes += entry_bytes;
            self.entries.insert(key, toks.clone());
        }
        Ok(toks)
    }
}

#[cfg(any(feature = "cuda", test))]
struct ResidentSession {
    state_slot: usize,
    kv_pages: Vec<i32>,
    tokens: Vec<i32>,
    readouts: HashMap<usize, Vec<f32>>,
    last_used: u64,
}

#[cfg(any(feature = "cuda", test))]
struct SessionPool {
    sessions: Vec<ResidentSession>,
    free_slots: Vec<usize>,
    free_pages: Vec<i32>,
    tick: u64,
}

#[cfg(any(feature = "cuda", test))]
impl SessionPool {
    fn new(total_pages: usize, max_sessions: usize) -> Self {
        Self {
            sessions: Vec::with_capacity(max_sessions),
            free_slots: (0..max_sessions).rev().collect(),
            free_pages: (0..total_pages as i32).rev().collect(),
            tick: 0,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn evict_lru(&mut self) -> bool {
        let victim = self
            .sessions
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.last_used)
            .map(|(i, _)| i);
        let Some(idx) = victim else {
            return false;
        };
        let evicted = self.sessions.swap_remove(idx);
        self.free_slots.push(evicted.state_slot);
        self.free_pages.extend(evicted.kv_pages);
        true
    }

    fn ensure_free_pages(&mut self, needed: usize) -> bool {
        while self.free_pages.len() < needed {
            if !self.evict_lru() {
                return false;
            }
        }
        true
    }

    /// Finds an exact-match session or the longest strict-prefix session whose cached readouts
    /// cover all offsets in `0..session.tokens.len()`.
    fn find_best_prefix(&self, flat_tokens: &[i32], offsets: &[usize]) -> Option<(usize, bool)> {
        let mut best: Option<(usize, usize, bool)> = None;
        for (idx, sess) in self.sessions.iter().enumerate() {
            let slen = sess.tokens.len();
            if slen == 0 || slen > flat_tokens.len() {
                continue;
            }
            if flat_tokens[..slen] != sess.tokens[..] {
                continue;
            }
            let all_prefix_offsets_cached = offsets
                .iter()
                .filter(|&&o| o < slen)
                .all(|o| sess.readouts.contains_key(o));
            if !all_prefix_offsets_cached {
                continue;
            }
            let is_exact = slen == flat_tokens.len();
            if best.is_none_or(|(_, best_len, _)| slen > best_len) {
                best = Some((idx, slen, is_exact));
            }
        }
        best.map(|(idx, _, is_exact)| (idx, is_exact))
    }

    fn allocate_pages(&mut self, count: usize) -> Vec<i32> {
        let split = self.free_pages.len().saturating_sub(count);
        self.free_pages.split_off(split)
    }
}

/// Applies head+tail block selection when tokenized blocks exceed `max_ctx`:
/// preserves leading system + first user task block (up to `max_ctx / 3`), then packs
/// as many most-recent blocks from the tail as fit.
#[cfg(feature = "cuda")]
fn select_head_tail_blocks(
    block_tokens: Vec<(String, Vec<i32>)>,
    max_ctx: usize,
) -> (Vec<i32>, Vec<usize>, Vec<String>) {
    let total: usize = block_tokens.iter().map(|(_, t)| t.len()).sum();
    if total <= max_ctx {
        let mut flat = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(block_tokens.len());
        let mut roles = Vec::with_capacity(block_tokens.len());
        for (role, toks) in block_tokens {
            if toks.is_empty() {
                continue;
            }
            flat.extend_from_slice(&toks);
            offsets.push(flat.len() - 1);
            roles.push(role);
        }
        return (flat, offsets, roles);
    }

    // Reserve up to 1/3 of max_ctx for the head (system + first user prompt).
    let head_budget = (max_ctx / 3).max(256);
    let mut head_count = 0usize;
    let mut head_used = 0usize;
    for (role, toks) in &block_tokens {
        if head_used + toks.len() > head_budget && head_count > 0 {
            break;
        }
        head_used += toks.len();
        head_count += 1;
        if role == "user" {
            break;
        }
    }
    if head_count == 0 && !block_tokens.is_empty() {
        head_count = 1;
    }

    let mut head_blocks: Vec<(String, Vec<i32>)> = Vec::new();
    let mut head_len = 0usize;
    for (role, mut toks) in block_tokens.iter().take(head_count).cloned() {
        let avail = head_budget.saturating_sub(head_len);
        if avail == 0 {
            break;
        }
        if toks.len() > avail {
            toks.truncate(avail);
        }
        head_len += toks.len();
        head_blocks.push((role, toks));
    }

    let tail_budget = max_ctx.saturating_sub(head_len);
    let tail_slice = &block_tokens[head_count..];
    // Drop from the middle in fixed 4,096-token quanta so long >32K agent trajectories keep a
    // stable `[head, tail_start..]` prefix across multiple turns instead of cold-prefilling 32K
    // on every single turn.
    const DROP_QUANTUM: usize = 4096;
    let tail_total: usize = tail_slice.iter().map(|(_, t)| t.len()).sum();
    let min_drop = tail_total.saturating_sub(tail_budget);
    let target_drop = if min_drop == 0 {
        0
    } else {
        let quantum_drop = min_drop.div_ceil(DROP_QUANTUM) * DROP_QUANTUM;
        quantum_drop.min(tail_total.saturating_sub(tail_budget / 2).max(min_drop))
    };
    let mut dropped = 0usize;
    let mut start_idx = 0usize;
    while start_idx + 1 < tail_slice.len() && dropped < target_drop {
        dropped += tail_slice[start_idx].1.len();
        start_idx += 1;
    }

    let mut tail_kept: Vec<(String, Vec<i32>)> = Vec::new();
    let mut tail_len = 0usize;
    for (role, mut toks) in tail_slice[start_idx..].iter().cloned() {
        if toks.is_empty() {
            continue;
        }
        let avail = tail_budget.saturating_sub(tail_len);
        if avail == 0 {
            break;
        }
        if toks.len() > avail {
            let cut = toks.len() - avail;
            toks = toks.split_off(cut);
        }
        tail_len += toks.len();
        tail_kept.push((role, toks));
    }

    let mut flat = Vec::with_capacity(head_len + tail_len);
    let mut offsets = Vec::with_capacity(head_blocks.len() + tail_kept.len());
    let mut roles = Vec::with_capacity(head_blocks.len() + tail_kept.len());
    for (role, toks) in head_blocks.into_iter().chain(tail_kept) {
        if toks.is_empty() {
            continue;
        }
        flat.extend_from_slice(&toks);
        offsets.push(flat.len() - 1);
        roles.push(role);
    }
    (flat, offsets, roles)
}

impl ArcEncoderWorker {
    /// Spawns a dedicated CUDA worker thread hosting the ARC-1.0 encoder on `device_id`.
    #[cfg(feature = "cuda")]
    pub fn spawn(model_dir: &std::path::Path, device_id: i32) -> Result<Self, String> {
        let model_dir = model_dir.to_path_buf();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
        let (tx, rx) = mpsc::sync_channel::<EncodeJob>(64);

        std::thread::Builder::new()
            .name("warp34-arc-encoder".to_string())
            .spawn(move || {
                let mut engine = match crate::model::Qwen3_5Engine::load_for_encoder_on_device(
                    &model_dir,
                    device_id,
                    ENCODER_KV_PAGES,
                    ENCODER_MAX_SESSIONS,
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("ARC encoder load failed: {e}")));
                        return;
                    }
                };
                let total_pages = engine.encoder_kv_pages();
                let max_ctx = (total_pages * 16).min(ENCODER_PER_SESSION_MAX_CTX);
                let _ = ready_tx.send(Ok(()));

                let mut memo = BlockTokenMemo::new();
                let mut pool = SessionPool::new(total_pages, ENCODER_MAX_SESSIONS);

                while let Ok(job) = rx.recv() {
                    if job.reply.is_closed() {
                        continue;
                    }
                    let result = (|| -> Result<EncodeResponse, String> {
                        let mut tokenized_blocks = Vec::with_capacity(job.blocks.len());
                        let mut total_prompt_tokens = 0usize;
                        for block in &job.blocks {
                            let toks = memo.tokenize(&engine, block)?;
                            total_prompt_tokens += toks.len();
                            if !toks.is_empty() {
                                tokenized_blocks.push((block.role.clone(), toks));
                            }
                        }
                        if tokenized_blocks.is_empty() {
                            return Err("ARC encoder received empty token sequence".to_string());
                        }
                        let (flat_tokens, offsets, roles) =
                            select_head_tail_blocks(tokenized_blocks, max_ctx);
                        if flat_tokens.is_empty() || offsets.is_empty() {
                            return Err("ARC encoder produced empty windowed tokens".to_string());
                        }

                        let pages_needed = flat_tokens.len().div_ceil(16);
                        let tick = pool.next_tick();

                        if let Some((sess_idx, is_exact)) =
                            pool.find_best_prefix(&flat_tokens, &offsets)
                        {
                            if is_exact {
                                let sess = &mut pool.sessions[sess_idx];
                                sess.last_used = tick;
                                let block_readouts =
                                    offsets.iter().map(|o| sess.readouts[o].clone()).collect();
                                return Ok(EncodeResponse {
                                    block_readouts,
                                    block_roles: roles,
                                    prompt_tokens: total_prompt_tokens,
                                });
                            }

                            // Strict prefix extension: take the session out of the pool so page
                            // evictions cannot invalidate its index, grow its KV pages, and prefill
                            // only the newly appended tail tokens.
                            let mut sess = pool.sessions.swap_remove(sess_idx);
                            let extra_pages = pages_needed.saturating_sub(sess.kv_pages.len());
                            if extra_pages > 0 {
                                if !pool.ensure_free_pages(extra_pages) {
                                    pool.free_slots.push(sess.state_slot);
                                    pool.free_pages.extend(sess.kv_pages);
                                    return Err("ARC encoder KV pool exhausted".to_string());
                                }
                                let added = pool.allocate_pages(extra_pages);
                                sess.kv_pages.extend(added);
                            }

                            let prev_len = sess.tokens.len();
                            let mut missing_indices = Vec::new();
                            let mut missing_offsets = Vec::new();
                            for (i, &off) in offsets.iter().enumerate() {
                                if off >= prev_len {
                                    missing_indices.push(i);
                                    missing_offsets.push(off);
                                }
                            }
                            let new_rows = match engine.encode_readouts_at_offsets_in_slot_f32(
                                &flat_tokens,
                                prev_len,
                                &missing_offsets,
                                sess.state_slot,
                                &sess.kv_pages,
                            ) {
                                Ok(rows) => rows,
                                Err(e) => {
                                    pool.free_slots.push(sess.state_slot);
                                    pool.free_pages.extend(sess.kv_pages);
                                    return Err(format!("ARC encode failed: {e}"));
                                }
                            };

                            let mut block_readouts: Vec<Vec<f32>> = vec![Vec::new(); offsets.len()];
                            for (i, &off) in offsets.iter().enumerate() {
                                if off < prev_len {
                                    block_readouts[i].clone_from(&sess.readouts[&off]);
                                }
                            }
                            for (idx, row) in missing_indices.into_iter().zip(new_rows) {
                                sess.readouts.insert(offsets[idx], row.clone());
                                block_readouts[idx] = row;
                            }
                            sess.tokens = flat_tokens;
                            sess.last_used = tick;
                            pool.sessions.push(sess);

                            return Ok(EncodeResponse {
                                block_readouts,
                                block_roles: roles,
                                prompt_tokens: total_prompt_tokens,
                            });
                        }

                        // Cold miss: allocate a fresh slot + KV pages (evicting LRU sessions as needed).
                        if pool.free_slots.is_empty() && !pool.evict_lru() {
                            return Err("ARC encoder has no free state slots".to_string());
                        }
                        if !pool.ensure_free_pages(pages_needed) {
                            return Err("ARC encoder KV pool exhausted".to_string());
                        }
                        let state_slot = pool
                            .free_slots
                            .pop()
                            .ok_or_else(|| "ARC encoder missing free state slot".to_string())?;
                        let kv_pages = pool.allocate_pages(pages_needed);

                        let rows = match engine.encode_readouts_at_offsets_in_slot_f32(
                            &flat_tokens,
                            0,
                            &offsets,
                            state_slot,
                            &kv_pages,
                        ) {
                            Ok(rows) => rows,
                            Err(e) => {
                                pool.free_slots.push(state_slot);
                                pool.free_pages.extend(kv_pages);
                                return Err(format!("ARC encode failed: {e}"));
                            }
                        };

                        let mut readouts = HashMap::with_capacity(offsets.len());
                        for (&off, row) in offsets.iter().zip(&rows) {
                            readouts.insert(off, row.clone());
                        }
                        pool.sessions.push(ResidentSession {
                            state_slot,
                            kv_pages,
                            tokens: flat_tokens,
                            readouts,
                            last_used: tick,
                        });

                        Ok(EncodeResponse {
                            block_readouts: rows,
                            block_roles: roles,
                            prompt_tokens: total_prompt_tokens,
                        })
                    })();
                    let _ = job.reply.send(result);
                }
            })
            .map_err(|e| format!("failed to spawn ARC encoder thread: {e}"))?;

        ready_rx
            .recv()
            .map_err(|_| "ARC encoder thread exited before ready".to_string())??;
        Ok(Self { tx })
    }

    /// CPU-only build stub.
    #[cfg(not(feature = "cuda"))]
    pub fn spawn(_model_dir: &std::path::Path, _device_id: i32) -> Result<Self, String> {
        Err("ARC-1.0 router requires a CUDA build (--features cuda)".to_string())
    }

    /// Runs the ARC-1.0 encoder asynchronously over canonical role blocks.
    pub async fn encode_blocks(&self, blocks: Vec<ArcRoleBlock>) -> Result<EncodeResponse, String> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .try_send(EncodeJob {
                blocks,
                reply: reply_tx,
            })
            .map_err(|e| format!("ARC encoder queue full/closed: {e}"))?;
        tokio::time::timeout(ENCODER_REQUEST_TIMEOUT, reply_rx)
            .await
            .map_err(|_| "ARC encoder timed out".to_string())?
            .map_err(|_| "ARC encoder dropped response".to_string())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_pool_handles_16_concurrent_interleaved_sessions_without_thrashing() {
        let mut pool = SessionPool::new(ENCODER_KV_PAGES, ENCODER_MAX_SESSIONS);
        assert_eq!(ENCODER_MAX_SESSIONS, 16);

        // Insert 16 distinct 128-token sessions (8 KV pages each = 128 pages total).
        for s in 0..16usize {
            let tokens: Vec<i32> = (0..128).map(|i| (s as i32) * 10_000 + i).collect();
            let pages_needed = tokens.len().div_ceil(16);
            assert!(pool.ensure_free_pages(pages_needed));
            let state_slot = pool.free_slots.pop().expect("slot available for <=16");
            let kv_pages = pool.allocate_pages(pages_needed);
            let tick = pool.next_tick();
            let mut readouts = HashMap::new();
            readouts.insert(127, vec![s as f32; 4]);
            pool.sessions.push(ResidentSession {
                state_slot,
                kv_pages,
                tokens,
                readouts,
                last_used: tick,
            });
        }
        assert_eq!(pool.sessions.len(), 16);
        assert!(pool.free_slots.is_empty());

        // Every one of the 16 sessions hits its prefix when extended with a new turn.
        for s in 0..16usize {
            let mut ext: Vec<i32> = (0..128).map(|i| (s as i32) * 10_000 + i).collect();
            ext.extend((0..32).map(|i| (s as i32) * 10_000 + 128 + i));
            let hit = pool.find_best_prefix(&ext, &[127, 159]);
            assert!(
                matches!(hit, Some((_, false))),
                "session {s} should hit strict prefix without eviction"
            );
        }

        // 17th session gracefully evicts LRU (session 0) and reclaims its slot + KV pages.
        assert!(pool.evict_lru());
        assert_eq!(pool.free_slots.len(), 1);
        assert_eq!(pool.sessions.len(), 15);
    }
}
