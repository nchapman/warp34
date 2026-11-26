//! Dedicated lightweight GPU encoder worker for ARC-1.0 (`~300 MiB` VRAM).
//!
//! Runs on a single background OS thread with a bounded channel so concurrent HTTP
//! requests never contend on CUDA contexts or block the main serving scheduler.

use std::path::Path;

pub struct EncodeResponse {
    pub readout: Vec<f32>,
    pub prompt_tokens: usize,
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::model::Qwen3_5Engine;
    use std::sync::mpsc;
    use tokio::sync::oneshot;

    struct EncodeJob {
        text: String,
        reply: oneshot::Sender<Result<EncodeResponse, String>>,
    }

    pub struct ArcEncoderWorker {
        tx: mpsc::SyncSender<EncodeJob>,
    }

    impl ArcEncoderWorker {
        /// Loads the 12-layer ARC-1.0 encoder onto `device_id` (`~300 MiB` VRAM total)
        /// and spawns its dedicated worker thread.
        pub fn spawn(model_dir: &Path, device_id: i32) -> Result<Self, String> {
            let model_dir = model_dir.to_path_buf();
            let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
            let (job_tx, job_rx) = mpsc::sync_channel::<EncodeJob>(64);

            std::thread::Builder::new()
                .name("warp34-arc-encoder".to_string())
                .spawn(move || {
                    // 256 KV pages = 4096 tokens context, 2 state slots, no CUDA graphs.
                    let init =
                        Qwen3_5Engine::load_for_encoder_on_device(&model_dir, device_id, 256);
                    let mut engine = match init {
                        Ok(e) => {
                            let _ = ready_tx.send(Ok(()));
                            e
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(format!("load ARC encoder: {e}")));
                            return;
                        }
                    };

                    while let Ok(job) = job_rx.recv() {
                        let res = (|| -> Result<EncodeResponse, String> {
                            let tokens = engine
                                .encode(&job.text)
                                .map_err(|e| format!("ARC tokenize failed: {e}"))?;
                            let prompt_tokens = tokens.len();
                            if tokens.is_empty() {
                                return Err("ARC tokenized input is empty".to_string());
                            }
                            let readout = engine
                                .encode_last_readout_f32(&tokens)
                                .map_err(|e| format!("ARC encode forward failed: {e}"))?;
                            Ok(EncodeResponse {
                                readout,
                                prompt_tokens,
                            })
                        })();
                        let _ = job.reply.send(res);
                    }
                })
                .map_err(|e| format!("spawn ARC encoder thread: {e}"))?;

            ready_rx
                .recv()
                .map_err(|_| "ARC encoder worker exited during init".to_string())??;
            Ok(Self { tx: job_tx })
        }

        pub async fn encode(&self, text: String) -> Result<EncodeResponse, String> {
            let (reply_tx, reply_rx) = oneshot::channel();
            self.tx
                .try_send(EncodeJob {
                    text,
                    reply: reply_tx,
                })
                .map_err(|_| "ARC encoder queue is full".to_string())?;
            tokio::time::timeout(std::time::Duration::from_secs(10), reply_rx)
                .await
                .map_err(|_| "ARC encoder timed out after 10s".to_string())?
                .map_err(|_| "ARC encoder worker closed".to_string())?
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod stub_impl {
    use super::*;

    pub struct ArcEncoderWorker;

    impl ArcEncoderWorker {
        pub fn spawn(_model_dir: &Path, _device_id: i32) -> Result<Self, String> {
            Err("ARC router encoder requires `--features cuda`".to_string())
        }

        pub async fn encode(&self, _text: String) -> Result<EncodeResponse, String> {
            Err("ARC router encoder requires `--features cuda`".to_string())
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::ArcEncoderWorker;
#[cfg(not(feature = "cuda"))]
pub use stub_impl::ArcEncoderWorker;
