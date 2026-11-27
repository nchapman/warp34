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
    tx: std::sync::Mutex<Option<mpsc::SyncSender<EncodeJob>>>,
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// Splits literal reserved vision tokens inside block content (`<|vision_start|>` etc.) so
/// source code mentioning them tokenizes as ordinary text rather than special vision IDs.
#[cfg(feature = "cuda")]
fn tokenize_serialized_safe(
    engine: &crate::model::Qwen3_5Engine,
    serialized: &str,
) -> Result<Vec<i32>, String> {
    const VISION_LITERALS: [&str; 5] = [
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|video_pad|>",
    ];
    if !VISION_LITERALS.iter().any(|pat| serialized.contains(pat)) {
        return engine
            .encode(serialized)
            .map_err(|e| format!("ARC tokenize failed: {e}"));
    }
    // Cut right after `<|` whenever a vision literal appears so each half tokenizes as text.
    let mut out = Vec::new();
    let mut rest = serialized;
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
            return tokenize_serialized_safe(engine, &key);
        }
        if let Some(cached) = self.entries.get(&key) {
            return Ok(cached.clone());
        }
        let toks = tokenize_serialized_safe(engine, &key)?;
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
#[cfg(any(feature = "cuda", test))]
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
    let head_budget = (max_ctx / 3).max(256).min(max_ctx);
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
    let mut oversized_final_tail: Option<Vec<i32>> = None;
    for (idx, (role, mut toks)) in block_tokens.iter().take(head_count).cloned().enumerate() {
        let avail = head_budget.saturating_sub(head_len);
        if avail == 0 {
            break;
        }
        let is_final_overall = idx + 1 == block_tokens.len();
        if toks.len() > avail {
            if is_final_overall {
                // Keep the unconsumed suffix so oversized single-turn / final user blocks still
                // expose their ending instructions and any trailing PII to Stage 1 and PII heads.
                let tail_avail = max_ctx.saturating_sub(head_len + avail);
                let suffix_start = toks.len().saturating_sub(tail_avail).max(avail);
                if suffix_start < toks.len() {
                    oversized_final_tail = Some(toks[suffix_start..].to_vec());
                }
            }
            toks.truncate(avail);
        }
        head_len += toks.len();
        head_blocks.push((role, toks));
    }
    if let (Some(suffix), Some((_, last_toks))) = (oversized_final_tail, head_blocks.last_mut()) {
        head_len += suffix.len();
        last_toks.extend_from_slice(&suffix);
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

#[cfg(feature = "cuda")]
struct ClaimedEncodeJob {
    reply: tokio::sync::oneshot::Sender<Result<EncodeResponse, String>>,
    flat_tokens: Vec<i32>,
    offsets: Vec<usize>,
    roles: Vec<String>,
    total_prompt_tokens: usize,
    start_offset: usize,
    missing_indices: Vec<usize>,
    missing_offsets: Vec<usize>,
    sess: ResidentSession,
    tick: u64,
}

#[cfg(feature = "cuda")]
struct PreparedJobTokens {
    flat_tokens: Vec<i32>,
    offsets: Vec<usize>,
    roles: Vec<String>,
    total_prompt_tokens: usize,
}

#[cfg(feature = "cuda")]
fn prepare_job_tokens(
    engine: &crate::model::Qwen3_5Engine,
    memo: &mut BlockTokenMemo,
    blocks: &[ArcRoleBlock],
    max_ctx: usize,
) -> Result<PreparedJobTokens, String> {
    let mut tokenized_blocks = Vec::with_capacity(blocks.len());
    let mut total_prompt_tokens = 0usize;
    for block in blocks {
        let toks = memo.tokenize(engine, block)?;
        total_prompt_tokens += toks.len();
        if !toks.is_empty() {
            tokenized_blocks.push((block.role.clone(), toks));
        }
    }
    if tokenized_blocks.is_empty() {
        return Err("ARC encoder received empty token sequence".to_string());
    }
    let (flat_tokens, offsets, roles) = select_head_tail_blocks(tokenized_blocks, max_ctx);
    if flat_tokens.is_empty() || offsets.is_empty() {
        return Err("ARC encoder produced empty windowed tokens".to_string());
    }
    Ok(PreparedJobTokens {
        flat_tokens,
        offsets,
        roles,
        total_prompt_tokens,
    })
}

#[cfg(feature = "cuda")]
fn run_encoder_microbatch(
    engine: &mut crate::model::Qwen3_5Engine,
    memo: &mut BlockTokenMemo,
    pool: &mut SessionPool,
    max_ctx: usize,
    jobs: Vec<EncodeJob>,
) -> Vec<EncodeJob> {
    let mut claimed: Vec<ClaimedEncodeJob> = Vec::with_capacity(jobs.len());
    let mut deferred: Vec<EncodeJob> = Vec::new();

    for job in jobs {
        if job.reply.is_closed() {
            continue;
        }
        let PreparedJobTokens {
            flat_tokens,
            offsets,
            roles,
            total_prompt_tokens,
        } = match prepare_job_tokens(engine, memo, &job.blocks, max_ctx) {
            Ok(v) => v,
            Err(e) => {
                let _ = job.reply.send(Err(e));
                continue;
            }
        };

        let pages_needed = flat_tokens.len().div_ceil(16);
        let tick = pool.next_tick();

        if let Some((sess_idx, is_exact)) = pool.find_best_prefix(&flat_tokens, &offsets) {
            if is_exact {
                let sess = &mut pool.sessions[sess_idx];
                sess.last_used = tick;
                let block_readouts = offsets.iter().map(|o| sess.readouts[o].clone()).collect();
                let _ = job.reply.send(Ok(EncodeResponse {
                    block_readouts,
                    block_roles: roles,
                    prompt_tokens: total_prompt_tokens,
                }));
                continue;
            }

            let mut sess = pool.sessions.swap_remove(sess_idx);
            let extra_pages = pages_needed.saturating_sub(sess.kv_pages.len());
            if extra_pages > 0 {
                if !pool.ensure_free_pages(extra_pages) {
                    pool.sessions.push(sess);
                    if !claimed.is_empty() {
                        deferred.push(job);
                        continue;
                    }
                    let _ = job
                        .reply
                        .send(Err("ARC encoder KV pool exhausted".to_string()));
                    continue;
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
            claimed.push(ClaimedEncodeJob {
                reply: job.reply,
                flat_tokens,
                offsets,
                roles,
                total_prompt_tokens,
                start_offset: prev_len,
                missing_indices,
                missing_offsets,
                sess,
                tick,
            });
            continue;
        }

        // Cold miss: allocate a fresh slot + KV pages (defer if earlier items in this batch
        // are temporarily holding the remaining slots/pages).
        if pool.free_slots.is_empty() && !pool.evict_lru() {
            if !claimed.is_empty() {
                deferred.push(job);
            } else {
                let _ = job
                    .reply
                    .send(Err("ARC encoder has no free state slots".to_string()));
            }
            continue;
        }
        if !pool.ensure_free_pages(pages_needed) {
            if !claimed.is_empty() {
                deferred.push(job);
            } else {
                let _ = job
                    .reply
                    .send(Err("ARC encoder KV pool exhausted".to_string()));
            }
            continue;
        }
        let Some(state_slot) = pool.free_slots.pop() else {
            let _ = job
                .reply
                .send(Err("ARC encoder missing free state slot".to_string()));
            continue;
        };
        let kv_pages = pool.allocate_pages(pages_needed);
        let missing_indices = (0..offsets.len()).collect();
        let missing_offsets = offsets.clone();
        claimed.push(ClaimedEncodeJob {
            reply: job.reply,
            flat_tokens,
            offsets,
            roles,
            total_prompt_tokens,
            start_offset: 0,
            missing_indices,
            missing_offsets,
            sess: ResidentSession {
                state_slot,
                kv_pages,
                tokens: Vec::new(),
                readouts: HashMap::new(),
                last_used: tick,
            },
            tick,
        });
    }

    if claimed.is_empty() {
        return deferred;
    }

    let batch_items: Vec<crate::model::EncoderBatchItem<'_>> = claimed
        .iter()
        .map(|c| crate::model::EncoderBatchItem {
            tokens: &c.flat_tokens,
            start_offset: c.start_offset,
            offsets: &c.missing_offsets,
            state_slot: c.sess.state_slot,
            block_table: &c.sess.kv_pages,
        })
        .collect();

    match engine.encode_readouts_batch_in_slots_f32(&batch_items) {
        Ok(batch_rows) => {
            for (mut c, new_rows) in claimed.into_iter().zip(batch_rows) {
                let mut block_readouts: Vec<Vec<f32>> = vec![Vec::new(); c.offsets.len()];
                if c.start_offset > 0 {
                    for (i, &off) in c.offsets.iter().enumerate() {
                        if off < c.start_offset {
                            block_readouts[i].clone_from(&c.sess.readouts[&off]);
                        }
                    }
                }
                for (idx, row) in c.missing_indices.into_iter().zip(new_rows) {
                    c.sess.readouts.insert(c.offsets[idx], row.clone());
                    block_readouts[idx] = row;
                }
                c.sess.tokens = c.flat_tokens;
                c.sess.last_used = c.tick;
                pool.sessions.push(c.sess);
                let _ = c.reply.send(Ok(EncodeResponse {
                    block_readouts,
                    block_roles: c.roles,
                    prompt_tokens: c.total_prompt_tokens,
                }));
            }
        }
        Err(e) => {
            let msg = format!("ARC encode failed: {e}");
            for c in claimed {
                pool.free_slots.push(c.sess.state_slot);
                pool.free_pages.extend(c.sess.kv_pages);
                let _ = c.reply.send(Err(msg.clone()));
            }
        }
    }

    deferred
}

impl ArcEncoderWorker {
    /// Spawns a dedicated CUDA worker thread hosting the ARC-1.0 encoder on `device_id`.
    #[cfg(feature = "cuda")]
    pub fn spawn(model_dir: &std::path::Path, device_id: i32) -> Result<Self, String> {
        let model_dir = model_dir.to_path_buf();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
        let (tx, rx) = mpsc::sync_channel::<EncodeJob>(256);

        let handle = std::thread::Builder::new()
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
                let mut pending: Vec<EncodeJob> = Vec::new();

                loop {
                    if pending.is_empty() {
                        match rx.recv() {
                            Ok(job) => pending.push(job),
                            Err(_) => break,
                        }
                    }
                    // Opportunistically drain already-queued jobs into one ragged micro-batch
                    // without waiting (zero added latency when concurrency == 1).
                    while pending.len() < ENCODER_MAX_SESSIONS {
                        match rx.try_recv() {
                            Ok(job) => pending.push(job),
                            Err(_) => break,
                        }
                    }
                    let batch = std::mem::take(&mut pending);
                    pending =
                        run_encoder_microbatch(&mut engine, &mut memo, &mut pool, max_ctx, batch);
                }
            })
            .map_err(|e| format!("failed to spawn ARC encoder thread: {e}"))?;

        if let Err(e) = ready_rx
            .recv()
            .map_err(|_| "ARC encoder thread exited before ready".to_string())
            .and_then(|r| r)
        {
            drop(tx);
            let _ = handle.join();
            return Err(e);
        }
        Ok(Self {
            tx: std::sync::Mutex::new(Some(tx)),
            handle: std::sync::Mutex::new(Some(handle)),
        })
    }

    /// CPU-only build stub.
    #[cfg(not(feature = "cuda"))]
    pub fn spawn(_model_dir: &std::path::Path, _device_id: i32) -> Result<Self, String> {
        Err("ARC-1.0 router requires a CUDA build (--features cuda)".to_string())
    }

    /// Signals the encoder thread to exit and awaits CUDA engine teardown cleanly before
    /// process exit (preventing `atexit` `libcudart`/`libcublasLt` destruction races).
    pub async fn shutdown(&self) {
        let _ = self.tx.lock().unwrap_or_else(|e| e.into_inner()).take();
        let handle = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = handle {
            let _ = tokio::task::spawn_blocking(move || {
                let _ = h.join();
            })
            .await;
        }
    }

    /// Runs the ARC-1.0 encoder asynchronously over canonical role blocks.
    pub async fn encode_blocks(&self, blocks: Vec<ArcRoleBlock>) -> Result<EncodeResponse, String> {
        let sender = self
            .tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .cloned()
            .ok_or_else(|| "ARC encoder is shut down".to_string())?;
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        sender
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

impl Drop for ArcEncoderWorker {
    fn drop(&mut self) {
        let _ = self.tx.lock().unwrap_or_else(|e| e.into_inner()).take();
        if tokio::runtime::Handle::try_current().is_err()
            && let Some(h) = self.handle.lock().unwrap_or_else(|e| e.into_inner()).take()
        {
            let _ = h.join();
        }
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

    #[test]
    fn oversized_single_user_block_preserves_head_and_trailing_suffix() {
        let big_user: Vec<i32> = (0..40_000).collect();
        let blocks = vec![("user".to_string(), big_user)];
        let (toks, ends, roles) = select_head_tail_blocks(blocks, 32_768);
        assert_eq!(roles, vec!["user"]);
        assert_eq!(ends.len(), 1);
        assert_eq!(toks.len(), 32_768);
        // First token preserved (head) AND last token preserved (trailing suffix/PII).
        assert_eq!(toks[0], 0);
        assert_eq!(*toks.last().unwrap(), 39_999);

        // Even when `max_ctx < 256`, `head_budget` clamps to `max_ctx` without overflow.
        let small_blocks = vec![
            ("system".to_string(), (0..100).collect()),
            ("user".to_string(), (100..300).collect()),
        ];
        let (small_toks, small_ends, small_roles) = select_head_tail_blocks(small_blocks, 128);
        assert!(small_toks.len() <= 128);
        assert_eq!(small_ends.len(), small_roles.len());
    }
}
