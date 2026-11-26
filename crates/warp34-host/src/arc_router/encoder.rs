//! Dedicated single-thread ARC-1.0 CUDA encoder worker sharing the host process.
//!
//! Runs `nchapman/ARC-1.0-W4A16-HGPTQ` (12 layers, 1024 hidden dim, W4A16-HGPTQ, no `lm_head`)
//! with per-block readout extraction, head+tail context windowing up to 32,768 tokens, and
//! incremental prefix caching so consecutive agent turns only prefill the newly appended block(s).

#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

use super::input::ArcRoleBlock;

const ENCODER_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// 2,048 KV pages × 16 tokens/page = 32,768 tokens (~192 MiB KV pool for 3 full-attn layers).
#[cfg(feature = "cuda")]
const ENCODER_KV_PAGES: usize = 2048;

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

/// Applies head+tail block selection when tokenized blocks exceed `max_ctx`:
/// preserves leading system + first user task block (up to `max_ctx / 3`), then packs
/// as many most-recent blocks from the tail as fit. If a single block alone exceeds the
/// budget, keeps its head+tail tokens so the total never exceeds `max_ctx`.
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
    let mut seen_user = false;
    for (role, toks) in &block_tokens {
        if head_used + toks.len() > head_budget && head_count > 0 {
            break;
        }
        head_used += toks.len();
        head_count += 1;
        if role == "user" {
            seen_user = true;
            break;
        }
    }
    if head_count == 0 && !block_tokens.is_empty() {
        head_count = 1;
    }
    let _ = seen_user;

    // Pack tail blocks in reverse until remaining budget is full.
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
    let mut tail_rev: Vec<(String, Vec<i32>)> = Vec::new();
    let mut tail_len = 0usize;
    for (role, mut toks) in block_tokens.iter().skip(head_count).rev().cloned() {
        if toks.is_empty() {
            continue;
        }
        let avail = tail_budget.saturating_sub(tail_len);
        if avail == 0 {
            break;
        }
        if toks.len() > avail {
            if tail_rev.is_empty() {
                // Keep the very end of the latest block so we always include the latest turn.
                let start = toks.len() - avail;
                toks = toks.split_off(start);
                tail_len += toks.len();
                tail_rev.push((role, toks));
            }
            break;
        }
        tail_len += toks.len();
        tail_rev.push((role, toks));
    }
    tail_rev.reverse();

    let mut flat = Vec::with_capacity(head_len + tail_len);
    let mut offsets = Vec::with_capacity(head_blocks.len() + tail_rev.len());
    let mut roles = Vec::with_capacity(head_blocks.len() + tail_rev.len());
    for (role, toks) in head_blocks.into_iter().chain(tail_rev) {
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
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("ARC encoder load failed: {e}")));
                        return;
                    }
                };
                let max_ctx = engine.encoder_max_context();
                let _ = ready_tx.send(Ok(()));

                // Single-slot incremental prefix cache: when a multi-turn conversation strictly
                // appends new role blocks to the previous call, reuse the slot-0 KV/GDN state
                // and cached block readouts, prefilling only the newly appended tail tokens.
                let mut prev_tokens: Vec<i32> = Vec::new();
                let mut prev_readouts: HashMap<usize, Vec<f32>> = HashMap::new();

                while let Ok(job) = rx.recv() {
                    let result = (|| -> Result<EncodeResponse, String> {
                        let mut tokenized_blocks = Vec::with_capacity(job.blocks.len());
                        let mut total_prompt_tokens = 0usize;
                        for block in &job.blocks {
                            let toks = engine
                                .encode(&block.serialized())
                                .map_err(|e| format!("ARC tokenize failed: {e}"))?;
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

                        // Exact repeat of previous prompt: return cached readouts in ~0 ms.
                        if flat_tokens == prev_tokens
                            && offsets.iter().all(|o| prev_readouts.contains_key(o))
                        {
                            let block_readouts =
                                offsets.iter().map(|o| prev_readouts[o].clone()).collect();
                            return Ok(EncodeResponse {
                                block_readouts,
                                block_roles: roles,
                                prompt_tokens: total_prompt_tokens,
                            });
                        }

                        // Strict prefix extension: keep slot-0 KV/GDN state and prefill only tail.
                        let is_strict_extension = !prev_tokens.is_empty()
                            && flat_tokens.len() > prev_tokens.len()
                            && flat_tokens[..prev_tokens.len()] == prev_tokens[..]
                            && offsets
                                .iter()
                                .filter(|&&o| o < prev_tokens.len())
                                .all(|o| prev_readouts.contains_key(o));

                        let (start_offset, missing_indices, missing_offsets) =
                            if is_strict_extension {
                                let mut m_idx = Vec::new();
                                let mut m_off = Vec::new();
                                for (i, &off) in offsets.iter().enumerate() {
                                    if off >= prev_tokens.len() {
                                        m_idx.push(i);
                                        m_off.push(off);
                                    }
                                }
                                (prev_tokens.len(), m_idx, m_off)
                            } else {
                                (
                                    0usize,
                                    (0..offsets.len()).collect::<Vec<_>>(),
                                    offsets.clone(),
                                )
                            };

                        let new_rows = match engine.encode_readouts_at_offsets_f32(
                            &flat_tokens,
                            start_offset,
                            &missing_offsets,
                        ) {
                            Ok(rows) => rows,
                            Err(e) => {
                                // Clear incremental cache on any CUDA/state error so the next
                                // request starts clean from offset 0.
                                prev_tokens.clear();
                                prev_readouts.clear();
                                return Err(format!("ARC encode failed: {e}"));
                            }
                        };

                        let mut block_readouts: Vec<Vec<f32>> = vec![Vec::new(); offsets.len()];
                        if is_strict_extension {
                            for (i, &off) in offsets.iter().enumerate() {
                                if off < prev_tokens.len() {
                                    block_readouts[i].clone_from(&prev_readouts[&off]);
                                }
                            }
                        } else {
                            prev_readouts.clear();
                        }
                        for (idx, row) in missing_indices.into_iter().zip(new_rows) {
                            prev_readouts.insert(offsets[idx], row.clone());
                            block_readouts[idx] = row;
                        }
                        prev_tokens = flat_tokens;

                        Ok(EncodeResponse {
                            block_readouts,
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
