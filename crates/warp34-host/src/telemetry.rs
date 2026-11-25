//! Shared engine telemetry. Snapshots copy CPU-owned state only; they never touch CUDA,
//! allocators, the scheduler, or network. Engine latencies start at ingress enqueue,
//! excluding HTTP parsing/tokenization/vision preprocessing. Generated tokens are committed
//! model output (including stop tokens), NOT tokens delivered to an HTTP client.
use serde::Serialize;
use std::{
    fmt::Write,
    sync::{Arc, Mutex, MutexGuard, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const BOUNDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1., 2.5, 5., 10., 30., 60., 120.,
];

#[derive(Clone, Debug, Serialize)]
pub struct Histogram {
    pub bounds: Vec<f64>,
    pub buckets: Vec<u64>,
    pub count: u64,
    pub sum: f64,
}
impl Default for Histogram {
    fn default() -> Self {
        Self {
            bounds: BOUNDS.to_vec(),
            buckets: vec![0; BOUNDS.len()],
            count: 0,
            sum: 0.,
        }
    }
}
impl Histogram {
    fn observe(&mut self, seconds: f64) {
        debug_assert!(seconds.is_finite() && seconds >= 0.);
        self.count += 1;
        self.sum += seconds;
        for (bound, bucket) in self.bounds.iter().zip(&mut self.buckets) {
            if seconds <= *bound {
                *bucket += 1;
            }
        }
    }

    fn merge_from(&mut self, other: &Self) {
        self.count = self.count.saturating_add(other.count);
        self.sum += other.sum;
        for (dst, src) in self.buckets.iter_mut().zip(&other.buckets) {
            *dst = dst.saturating_add(*src);
        }
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Requests {
    pub submitted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub running: u64,
    pub waiting: u64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Tokens {
    pub prompt: u64,
    pub computed: u64,
    pub cached: u64,
    pub generated: u64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Latency {
    pub ttft: Histogram,
    pub queue: Histogram,
    pub e2e: Histogram,
    pub time_per_output_token: Histogram,
}
/// Logical target KV pages / GDN slots, counted once across TP ranks. Cached means
/// evictable, active includes protected cache plus private/reserved allocations.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Pool {
    pub capacity: u64,
    pub active: Option<u64>,
    pub cached: u64,
    pub free: Option<u64>,
}
impl Pool {
    pub fn known(capacity: usize, free: usize, cached: usize) -> Self {
        Self {
            capacity: capacity as u64,
            // A bad allocator/radix invariant must not make telemetry panic the worker.
            active: capacity
                .checked_sub(free)
                .and_then(|n| n.checked_sub(cached))
                .map(|n| n as u64),
            cached: cached as u64,
            free: Some(free as u64),
        }
    }
}
/// Tiered `WHT128-Q8_0` GDN checkpoint occupancy (L1 VRAM and L2 pinned host RAM),
/// counted once across TP ranks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CheckpointPoolStats {
    pub vram_capacity: u64,
    pub vram_used: u64,
    pub host_capacity: u64,
    pub host_used: u64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Pools {
    pub kv: Pool,
    pub gdn: Pool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gdn_checkpoints: Option<CheckpointPoolStats>,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Speculation {
    pub rounds: u64,
    pub draft_tokens: Option<u64>,
    pub accepted_tokens: u64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct HttpResponses {
    pub success: u64,
    pub client_error: u64,
    pub server_error: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Config {
    pub tp_size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_size: Option<usize>,
    pub kv_format: String,
    pub speculation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lora_adapters: Vec<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub schema_version: u32,
    pub instance_id: String,
    pub uptime_seconds: f64,
    pub model: String,
    pub status: &'static str,
    pub config: Config,
    pub requests: Requests,
    pub tokens: Tokens,
    pub latency: Latency,
    pub pools: Pools,
    pub speculation: Speculation,
    pub http_responses: HttpResponses,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workers: Vec<Self>,
    #[serde(skip)]
    peak_running: u64,
    #[serde(skip)]
    decode_bucket_mask: u64,
}
struct Process {
    id: String,
    started: Instant,
}
fn process() -> &'static Process {
    static PROCESS: OnceLock<Process> = OnceLock::new();
    PROCESS.get_or_init(|| Process {
        id: format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ),
        started: Instant::now(),
    })
}
pub struct Telemetry {
    state: Mutex<Snapshot>,
}
impl Default for Telemetry {
    fn default() -> Self {
        Arc::try_unwrap(Self::new(Config {
            tp_size: 0,
            dp_size: None,
            kv_format: "unknown".into(),
            speculation: "none".into(),
            max_model_len: None,
            lora_adapters: Vec::new(),
        }))
        .ok()
        .unwrap()
    }
}
impl Telemetry {
    pub fn initialize(&self, config: Config, model: String) {
        let mut s = self.lock();
        s.config = config;
        s.model = model;
    }
    pub fn new(config: Config) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(Snapshot {
                schema_version: 1,
                instance_id: process().id.clone(),
                uptime_seconds: 0.,
                model: String::new(),
                status: "ready",
                config,
                requests: Requests::default(),
                tokens: Tokens::default(),
                latency: Latency::default(),
                pools: Pools::default(),
                speculation: Speculation {
                    draft_tokens: Some(0),
                    ..Speculation::default()
                },
                http_responses: HttpResponses::default(),
                workers: Vec::new(),
                peak_running: 0,
                decode_bucket_mask: 0,
            }),
        })
    }
    fn lock(&self) -> MutexGuard<'_, Snapshot> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn set_model(&self, model: String) {
        self.lock().model = model;
    }
    pub fn set_max_model_len(&self, max_model_len: Option<usize>) {
        self.lock().config.max_model_len = max_model_len;
    }
    pub fn set_status(&self, status: &'static str) {
        let mut s = self.lock();
        if s.status != "unhealthy" {
            s.status = status;
        }
    }
    pub fn snapshot(&self) -> Snapshot {
        let mut s = self.lock().clone();
        s.uptime_seconds = process().started.elapsed().as_secs_f64();
        s
    }
    pub fn publish_decode_bucket_mask(&self, mask: u64) {
        self.lock().decode_bucket_mask = mask;
    }
    pub fn publish_pools(&self, pools: Pools) {
        self.lock().pools = pools;
    }
    pub fn speculation(&self, rounds: u64, drafts: u64, accepted: u64) {
        let mut s = self.lock();
        s.speculation.rounds += rounds;
        *s.speculation.draft_tokens.get_or_insert(0) += drafts;
        s.speculation.accepted_tokens += accepted;
    }
    /// Status of inference HTTP responses, including pre-worker failures. SSE is
    /// counted at header creation; subsequent stream failures belong to lifecycle counters.
    pub fn http_response(&self, status: u16) {
        let mut s = self.lock();
        match status {
            200..=299 => s.http_responses.success += 1,
            400..=499 => s.http_responses.client_error += 1,
            500..=599 => s.http_responses.server_error += 1,
            _ => {}
        }
    }
    pub fn request(self: &Arc<Self>, prompt: usize) -> Lifecycle {
        let start = Instant::now();
        let mut s = self.lock();
        s.requests.submitted += 1;
        s.requests.waiting += 1;
        s.tokens.prompt += prompt as u64;
        Lifecycle {
            telemetry: self.clone(),
            state: Mutex::new(LifeState {
                start,
                scheduled: false,
                terminal: false,
                first: None,
                last: None,
                generated: 0,
            }),
        }
    }
}
struct LifeState {
    start: Instant,
    scheduled: bool,
    terminal: bool,
    first: Option<Instant>,
    last: Option<Instant>,
    generated: u64,
}
#[derive(Clone, Copy)]
pub enum Outcome {
    Completed,
    Failed,
    Cancelled,
}
/// Dropping an unfinished guard fails it, including locally owned requests lost
/// during admission/batched speculation unwind. Explicit terminal transitions win once.
pub struct Lifecycle {
    telemetry: Arc<Telemetry>,
    state: Mutex<LifeState>,
}
impl Lifecycle {
    pub fn scheduled(&self, cached: usize) {
        let mut l = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if l.scheduled || l.terminal {
            return;
        }
        l.scheduled = true;
        let mut s = self.telemetry.lock();
        s.requests.waiting -= 1;
        s.requests.running += 1;
        s.peak_running = s.peak_running.max(s.requests.running);
        s.tokens.cached += cached as u64;
        s.latency.queue.observe(l.start.elapsed().as_secs_f64());
    }
    pub fn computed(&self, tokens: usize) {
        self.telemetry.lock().tokens.computed += tokens as u64;
    }
    pub fn token(&self) {
        self.token_at(Instant::now());
    }
    fn token_at(&self, now: Instant) {
        let mut l = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if l.terminal {
            return;
        }
        let mut s = self.telemetry.lock();
        if l.first.is_none() {
            l.first = Some(now);
            s.latency
                .ttft
                .observe(now.duration_since(l.start).as_secs_f64());
        }
        l.last = Some(now);
        l.generated += 1;
        s.tokens.generated += 1;
    }
    pub fn finish(&self, outcome: Outcome) {
        self.finish_at(outcome, Instant::now());
    }
    fn finish_at(&self, outcome: Outcome, now: Instant) {
        let mut l = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if l.terminal {
            return;
        }
        l.terminal = true;
        let mut s = self.telemetry.lock();
        if l.scheduled {
            s.requests.running -= 1;
        } else {
            s.requests.waiting -= 1;
        }
        match outcome {
            Outcome::Completed => s.requests.completed += 1,
            Outcome::Failed => s.requests.failed += 1,
            Outcome::Cancelled => s.requests.cancelled += 1,
        }
        s.latency
            .e2e
            .observe(now.duration_since(l.start).as_secs_f64());
        // One observation per multi-token request, not fake ITL from draining
        // speculative output vectors or HTTP channels. Excludes first-token time.
        if l.generated > 1 {
            s.latency.time_per_output_token.observe(
                l.last
                    .unwrap()
                    .duration_since(l.first.unwrap())
                    .as_secs_f64()
                    / (l.generated - 1) as f64,
            );
        }
    }
}
impl Drop for Lifecycle {
    fn drop(&mut self) {
        self.finish(Outcome::Failed);
    }
}

fn metric(out: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
    )
    .unwrap();
}
fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}
impl Snapshot {
    pub fn prometheus(&self) -> String {
        let mut out = String::new();
        let r = &self.requests;
        let t = &self.tokens;
        for (name, help, value) in [
            (
                "warp34_requests_total",
                "Requests enqueued into the worker (all inference APIs).",
                r.submitted,
            ),
            (
                "warp34_completed_requests_total",
                "Worker requests completed successfully.",
                r.completed,
            ),
            (
                "warp34_failed_requests_total",
                "Worker requests failed, including panic and shutdown.",
                r.failed,
            ),
            (
                "warp34_cancelled_requests_total",
                "Worker requests cancelled, including output backpressure.",
                r.cancelled,
            ),
            (
                "warp34_prompt_tokens_total",
                "Prompt tokens submitted to the worker.",
                t.prompt,
            ),
            (
                "warp34_computed_prompt_tokens_total",
                "Prompt tokens in successfully executed engine prefill chunks.",
                t.computed,
            ),
            (
                "warp34_cached_prompt_tokens_total",
                "Prompt tokens restored at scheduler admission.",
                t.cached,
            ),
            (
                "warp34_tokens_generated_total",
                "Committed model output tokens, including stop tokens; not wire delivery.",
                t.generated,
            ),
            (
                "warp34_speculation_rounds_total",
                "Successful per-request speculative verification rounds.",
                self.speculation.rounds,
            ),
            (
                "warp34_speculation_accepted_tokens_total",
                "Accepted speculative draft tokens.",
                self.speculation.accepted_tokens,
            ),
        ] {
            metric(&mut out, name, "counter", help, value);
        }
        if let Some(n) = self.speculation.draft_tokens {
            metric(
                &mut out,
                "warp34_speculation_draft_tokens_total",
                "counter",
                "Draft tokens included in successful verification, after budget and constraint trimming.",
                n,
            );
        }
        metric(
            &mut out,
            "warp34_active_requests",
            "gauge",
            "Scheduled unfinished worker requests.",
            r.running,
        );
        metric(
            &mut out,
            "warp34_waiting_requests",
            "gauge",
            "Ingress plus scheduler waiting requests.",
            r.waiting,
        );
        metric(
            &mut out,
            "warp34_peak_active_requests",
            "gauge",
            "Peak scheduled unfinished worker requests.",
            self.peak_running,
        );
        metric(
            &mut out,
            "warp34_decode_bucket_mask",
            "gauge",
            "Diagnostic bit mask of captured decode request buckets, published by the worker.",
            self.decode_bucket_mask,
        );
        out.push_str("# HELP warp34_status Worker health state.\n# TYPE warp34_status gauge\n");
        for status in ["ready", "unhealthy", "stopping"] {
            writeln!(
                out,
                "warp34_status{{state=\"{status}\"}} {}",
                u8::from(self.status == status)
            )
            .unwrap();
        }
        metric(
            &mut out,
            "warp34_uptime_seconds",
            "gauge",
            "Process telemetry uptime in seconds.",
            self.uptime_seconds,
        );
        metric(
            &mut out,
            "warp34_radix_cache_hit_rate",
            "gauge",
            "Restored prompt tokens divided by submitted prompt tokens.",
            t.cached as f64 / t.prompt.max(1) as f64,
        );
        for (name, help, hist) in [
            (
                "warp34_queue_seconds",
                "Engine enqueue to first scheduler admission; excludes HTTP preprocessing.",
                &self.latency.queue,
            ),
            (
                "warp34_ttft_seconds",
                "Engine enqueue to first committed model token; excludes HTTP preprocessing and prefill-only requests.",
                &self.latency.ttft,
            ),
            (
                "warp34_e2e_seconds",
                "Engine enqueue to terminal, all outcomes; excludes HTTP preprocessing.",
                &self.latency.e2e,
            ),
            (
                "warp34_time_per_output_token_seconds",
                "Per-request (last-first engine token time)/(output tokens-1); multi-token requests only, not per-channel-drain ITL.",
                &self.latency.time_per_output_token,
            ),
        ] {
            writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram").unwrap();
            for (bound, count) in hist.bounds.iter().zip(&hist.buckets) {
                writeln!(out, "{name}_bucket{{le=\"{bound}\"}} {count}").unwrap();
            }
            writeln!(
                out,
                "{name}_bucket{{le=\"+Inf\"}} {}\n{name}_count {}\n{name}_sum {}",
                hist.count, hist.count, hist.sum
            )
            .unwrap();
        }
        out.push_str("# HELP warp34_pool_units Logical target pool units (KV pages or GDN slots); cached is evictable, active includes protected and reserved, TP counted once.\n# TYPE warp34_pool_units gauge\n");
        for (name, pool) in [("kv", &self.pools.kv), ("gdn", &self.pools.gdn)] {
            for (kind, value) in [
                ("capacity", Some(pool.capacity)),
                ("active", pool.active),
                ("cached", Some(pool.cached)),
                ("free", pool.free),
            ] {
                if let Some(n) = value {
                    writeln!(
                        out,
                        "warp34_pool_units{{pool=\"{name}\",state=\"{kind}\"}} {n}"
                    )
                    .unwrap();
                }
            }
        }
        if let Some(ckpt) = &self.pools.gdn_checkpoints {
            for (name, cap, used) in [
                ("gdn_ckpt_vram", ckpt.vram_capacity, ckpt.vram_used),
                ("gdn_ckpt_host", ckpt.host_capacity, ckpt.host_used),
            ] {
                writeln!(
                    out,
                    "warp34_pool_units{{pool=\"{name}\",state=\"capacity\"}} {cap}\nwarp34_pool_units{{pool=\"{name}\",state=\"used\"}} {used}"
                )
                .unwrap();
            }
        }
        writeln!(
            out,
            "# HELP warp34_lora_adapters_loaded Number of static LoRA adapters loaded on this engine.\n# TYPE warp34_lora_adapters_loaded gauge\nwarp34_lora_adapters_loaded {}",
            self.config.lora_adapters.len()
        )
        .unwrap();
        out.push_str("# HELP warp34_http_responses_total Inference HTTP response headers by bounded status class; excludes scrapes, status, token counts, and static routes.\n# TYPE warp34_http_responses_total counter\n");
        for (class, n) in [
            ("success", self.http_responses.success),
            ("client_error", self.http_responses.client_error),
            ("server_error", self.http_responses.server_error),
        ] {
            writeln!(out, "warp34_http_responses_total{{class=\"{class}\"}} {n}").unwrap();
        }
        out.push_str("# HELP warp34_model_info Static worker model configuration.\n# TYPE warp34_model_info gauge\n");
        let dp_size = self.config.dp_size.unwrap_or(1);
        writeln!(
            out,
            "warp34_model_info{{model=\"{}\",tp_size=\"{}\",dp_size=\"{}\",kv_format=\"{}\",speculation=\"{}\"}} 1",
            escape(&self.model),
            self.config.tp_size,
            dp_size,
            escape(&self.config.kv_format),
            escape(&self.config.speculation)
        )
        .unwrap();
        if self.workers.len() > 1 {
            out.push_str("# HELP warp34_dp_worker_requests In-flight requests per data-parallel worker.\n# TYPE warp34_dp_worker_requests gauge\n");
            for (rank, w) in self.workers.iter().enumerate() {
                writeln!(
                    out,
                    "warp34_dp_worker_requests{{dp_rank=\"{rank}\",state=\"running\"}} {}",
                    w.requests.running
                )
                .unwrap();
                writeln!(
                    out,
                    "warp34_dp_worker_requests{{dp_rank=\"{rank}\",state=\"waiting\"}} {}",
                    w.requests.waiting
                )
                .unwrap();
            }
            out.push_str("# HELP warp34_dp_worker_kv_utilization Active KV pool utilization ratio per data-parallel worker.\n# TYPE warp34_dp_worker_kv_utilization gauge\n");
            for (rank, w) in self.workers.iter().enumerate() {
                let util = if w.pools.kv.capacity > 0 {
                    w.pools.kv.active.unwrap_or(0) as f64 / w.pools.kv.capacity as f64
                } else {
                    0.0
                };
                writeln!(
                    out,
                    "warp34_dp_worker_kv_utilization{{dp_rank=\"{rank}\"}} {util:.6}"
                )
                .unwrap();
            }
        }
        out
    }

    /// Aggregates per-worker snapshots from an in-process DP pool into a single server-level
    /// snapshot while preserving per-worker details in `workers` for `/api/metrics` and `/metrics`.
    pub fn aggregate(snapshots: &[Self]) -> Self {
        assert!(
            !snapshots.is_empty(),
            "aggregate requires at least one snapshot"
        );
        if snapshots.len() == 1 {
            return snapshots[0].clone();
        }
        let mut agg = snapshots[0].clone();
        agg.config.dp_size = Some(snapshots.len());
        agg.status = if snapshots.iter().any(|s| s.status == "ready") {
            "ready"
        } else if snapshots.iter().all(|s| s.status == "stopping") {
            "stopping"
        } else {
            "error"
        };
        agg.workers = snapshots.to_vec();
        for w in &snapshots[1..] {
            agg.uptime_seconds = agg.uptime_seconds.max(w.uptime_seconds);
            agg.requests.submitted = agg.requests.submitted.saturating_add(w.requests.submitted);
            agg.requests.completed = agg.requests.completed.saturating_add(w.requests.completed);
            agg.requests.cancelled = agg.requests.cancelled.saturating_add(w.requests.cancelled);
            agg.requests.failed = agg.requests.failed.saturating_add(w.requests.failed);
            agg.requests.running = agg.requests.running.saturating_add(w.requests.running);
            agg.requests.waiting = agg.requests.waiting.saturating_add(w.requests.waiting);
            agg.tokens.prompt = agg.tokens.prompt.saturating_add(w.tokens.prompt);
            agg.tokens.computed = agg.tokens.computed.saturating_add(w.tokens.computed);
            agg.tokens.cached = agg.tokens.cached.saturating_add(w.tokens.cached);
            agg.tokens.generated = agg.tokens.generated.saturating_add(w.tokens.generated);
            agg.latency.ttft.merge_from(&w.latency.ttft);
            agg.latency.queue.merge_from(&w.latency.queue);
            agg.latency.e2e.merge_from(&w.latency.e2e);
            agg.latency
                .time_per_output_token
                .merge_from(&w.latency.time_per_output_token);
            merge_pool(&mut agg.pools.kv, &w.pools.kv);
            merge_pool(&mut agg.pools.gdn, &w.pools.gdn);
            match (&mut agg.pools.gdn_checkpoints, &w.pools.gdn_checkpoints) {
                (Some(dst), Some(src)) => {
                    dst.vram_capacity = dst.vram_capacity.saturating_add(src.vram_capacity);
                    dst.vram_used = dst.vram_used.saturating_add(src.vram_used);
                    dst.host_capacity = dst.host_capacity.saturating_add(src.host_capacity);
                    dst.host_used = dst.host_used.saturating_add(src.host_used);
                }
                (None, Some(src)) => agg.pools.gdn_checkpoints = Some(src.clone()),
                _ => {}
            }
            agg.speculation.rounds = agg.speculation.rounds.saturating_add(w.speculation.rounds);
            agg.speculation.accepted_tokens = agg
                .speculation
                .accepted_tokens
                .saturating_add(w.speculation.accepted_tokens);
            agg.speculation.draft_tokens =
                match (agg.speculation.draft_tokens, w.speculation.draft_tokens) {
                    (Some(a), Some(b)) => Some(a.saturating_add(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
            agg.http_responses.success = agg
                .http_responses
                .success
                .saturating_add(w.http_responses.success);
            agg.http_responses.client_error = agg
                .http_responses
                .client_error
                .saturating_add(w.http_responses.client_error);
            agg.http_responses.server_error = agg
                .http_responses
                .server_error
                .saturating_add(w.http_responses.server_error);
            agg.peak_running = agg.peak_running.max(w.peak_running);
            agg.decode_bucket_mask |= w.decode_bucket_mask;
        }
        agg.peak_running = agg.peak_running.max(agg.requests.running);
        agg
    }
}

fn merge_pool(dst: &mut Pool, src: &Pool) {
    dst.capacity = dst.capacity.saturating_add(src.capacity);
    dst.cached = dst.cached.saturating_add(src.cached);
    dst.active = match (dst.active, src.active) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    dst.free = match (dst.free, src.free) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    fn telemetry() -> Arc<Telemetry> {
        Telemetry::new(Config {
            tp_size: 1,
            dp_size: None,
            kv_format: "tq4".into(),
            speculation: "none".into(),
            max_model_len: None,
            lora_adapters: Vec::new(),
        })
    }
    #[test]
    fn lifecycle_includes_ingress_and_fails_once_on_unwind() {
        let t = telemetry();
        let result = std::panic::catch_unwind({
            let t = t.clone();
            move || {
                let _r = t.request(10);
                assert_eq!(t.snapshot().requests.waiting, 1);
                panic!("fixture");
            }
        });
        assert!(result.is_err());
        let r = t.request(20);
        r.scheduled(8);
        r.scheduled(8);
        r.computed(12);
        r.finish(Outcome::Cancelled);
        r.finish(Outcome::Completed);
        drop(r);
        let s = t.snapshot();
        assert_eq!(
            (
                s.requests.submitted,
                s.requests.failed,
                s.requests.cancelled,
                s.requests.completed,
                s.requests.waiting,
                s.requests.running
            ),
            (2, 1, 1, 0, 0, 0)
        );
        assert_eq!(
            (s.tokens.prompt, s.tokens.cached, s.tokens.computed),
            (30, 8, 12)
        );
        assert_eq!(s.latency.e2e.count, 2);
        assert_eq!(s.latency.queue.count, 1);
        assert_eq!(s.latency.ttft.count, 0);
    }
    #[test]
    fn engine_tokens_are_not_wire_drains_and_decisions_have_no_ttft() {
        let t = telemetry();
        let r = t.request(4);
        r.scheduled(0);
        let start = r.state.lock().unwrap().start;
        r.token_at(start + std::time::Duration::from_secs(1));
        r.token_at(start + std::time::Duration::from_secs(3));
        r.token_at(start + std::time::Duration::from_secs(3));
        r.finish_at(
            Outcome::Completed,
            start + std::time::Duration::from_secs(4),
        );
        let decision = t.request(5);
        decision.scheduled(0);
        decision.finish(Outcome::Completed);
        let s = t.snapshot();
        assert_eq!(s.tokens.generated, 3);
        assert_eq!(s.latency.ttft.sum, 1.);
        assert_eq!(s.latency.time_per_output_token.count, 1);
        assert_eq!(s.latency.time_per_output_token.sum, 1.);
    }
    #[test]
    fn snapshots_remain_balanced_under_concurrent_ingress_and_terminals() {
        let t = telemetry();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let t = t.clone();
                scope.spawn(move || {
                    for _ in 0..100 {
                        let r = t.request(3);
                        r.scheduled(1);
                        r.computed(2);
                        r.token();
                        r.finish(Outcome::Completed);
                    }
                });
            }
            for _ in 0..100 {
                let s = t.snapshot();
                let r = s.requests;
                assert_eq!(
                    r.submitted,
                    r.waiting + r.running + r.completed + r.failed + r.cancelled
                );
                assert_eq!(s.latency.e2e.count, r.completed + r.failed + r.cancelled);
            }
        });
        assert_eq!(t.snapshot().requests.completed, 400);
        t.set_status("unhealthy");
        t.set_status("stopping");
        assert_eq!(t.snapshot().status, "unhealthy");
    }

    #[test]
    fn histograms_serialization_and_prometheus_are_consistent() {
        let mut h = Histogram::default();
        h.observe(0.001);
        h.observe(0.1);
        h.observe(121.);
        assert_eq!(h.buckets[0], 1);
        assert_eq!(h.buckets[5], 2);
        assert_eq!(h.count, 3);
        let t = telemetry();
        t.set_model("a\"b\\c\nd".into());
        t.http_response(200);
        t.http_response(400);
        t.http_response(503);
        let mut s = t.snapshot();
        s.config.lora_adapters = vec!["sql-expert".into(), "code-review".into()];
        s.latency.ttft = h;
        s.pools.gdn_checkpoints = Some(CheckpointPoolStats {
            vram_capacity: 16,
            vram_used: 12,
            host_capacity: 99,
            host_used: 34,
        });
        let json = serde_json::to_value(&s).unwrap();
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json.as_object().unwrap().len(), 12);
        assert!(json.get("peak_running").is_none());
        assert_eq!(
            json["config"]["lora_adapters"],
            serde_json::json!(["sql-expert", "code-review"])
        );
        assert_eq!(json["latency"]["ttft"]["count"], 3);
        assert_eq!(json["speculation"]["draft_tokens"], 0);
        assert_eq!(json["pools"]["gdn_checkpoints"]["vram_capacity"], 16);
        assert_eq!(json["pools"]["gdn_checkpoints"]["vram_used"], 12);
        assert_eq!(json["pools"]["gdn_checkpoints"]["host_capacity"], 99);
        assert_eq!(json["pools"]["gdn_checkpoints"]["host_used"], 34);
        let p = s.prometheus();
        assert!(p.contains("warp34_ttft_seconds_bucket{le=\"+Inf\"} 3"));
        assert!(p.contains("model=\"a\\\"b\\\\c\\nd\""));
        assert!(p.contains("# TYPE warp34_requests_total counter"));
        assert!(p.contains("warp34_lora_adapters_loaded 2"));
        assert!(p.contains("warp34_pool_units{pool=\"gdn_ckpt_vram\",state=\"capacity\"} 16"));
        assert!(p.contains("warp34_pool_units{pool=\"gdn_ckpt_vram\",state=\"used\"} 12"));
        assert!(p.contains("warp34_pool_units{pool=\"gdn_ckpt_host\",state=\"capacity\"} 99"));
        assert!(p.contains("warp34_pool_units{pool=\"gdn_ckpt_host\",state=\"used\"} 34"));
        assert_eq!(Pool::known(100, 10, 70).active, Some(20));
        assert_eq!(t.snapshot().instance_id, telemetry().snapshot().instance_id);
    }

    #[test]
    fn aggregate_dp_snapshots_sums_counters_pools_and_emits_per_rank_prometheus() {
        let t0 = telemetry();
        let t1 = telemetry();
        let t2 = telemetry();
        t0.set_model("m".into());
        t1.set_model("m".into());
        t2.set_model("m".into());
        t0.publish_pools(Pools {
            kv: Pool::known(100, 60, 20),
            gdn: Pool::known(16, 12, 2),
            gdn_checkpoints: None,
        });
        t1.publish_pools(Pools {
            kv: Pool::known(100, 50, 10),
            gdn: Pool::known(16, 10, 4),
            gdn_checkpoints: None,
        });
        t2.publish_pools(Pools {
            kv: Pool::known(100, 90, 5),
            gdn: Pool::known(16, 15, 0),
            gdn_checkpoints: None,
        });
        let r0 = t0.request(100);
        r0.scheduled(40);
        r0.computed(60);
        r0.token();
        r0.finish(Outcome::Completed);
        let r1 = t1.request(200);
        r1.scheduled(150);

        let agg = Snapshot::aggregate(&[t0.snapshot(), t1.snapshot(), t2.snapshot()]);
        assert_eq!(agg.config.dp_size, Some(3));
        assert_eq!(agg.workers.len(), 3);
        assert_eq!(agg.requests.submitted, 2);
        assert_eq!(agg.requests.completed, 1);
        assert_eq!(agg.requests.running, 1);
        assert_eq!(agg.tokens.cached, 190);
        assert_eq!(agg.tokens.generated, 1);
        assert_eq!(agg.pools.kv.capacity, 300);
        assert_eq!(agg.pools.kv.active, Some(65));
        let prom = agg.prometheus();
        assert!(prom.contains("dp_size=\"3\""));
        assert!(prom.contains("warp34_dp_worker_requests{dp_rank=\"1\",state=\"running\"} 1"));
        assert!(prom.contains("warp34_dp_worker_kv_utilization{dp_rank=\"0\"} 0.200000"));
        assert!(prom.contains("warp34_dp_worker_kv_utilization{dp_rank=\"1\"} 0.400000"));
        assert!(prom.contains("warp34_dp_worker_kv_utilization{dp_rank=\"2\"} 0.050000"));
    }
}
