//! ARC-1.0 Stage-1, Stage-2, and `signals.pii.payload` heads + runtime catalog.
//!
//! Math follows the Apache-2.0 `rayline-ai/rayline-cli` (`rayline-arc-decide`) reference
//! over `runtime/heads.safetensors`, `model.safetensors`, `runtime/arc-runtime.json`,
//! `runtime/prices.json`, and `config.json`.

use crate::loader::SafetensorsArchive;
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

/// Billed token legs: `[uncached_input, cache_read, cache_write, output]`.
pub const LEGS: usize = 4;
const MIN_POOL: i64 = 20;
const FLOOR_TOKENS: f64 = 1.0;
const PLATT_EPS: f64 = 1e-6;
const ROLES: [&str; 4] = ["system", "user", "assistant", "tool"];
const USAGE_SCALE: f32 = 1_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageOnePrediction {
    pub p_pass: f64,
    pub usage: [f64; LEGS],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmScore {
    pub quality: f32,
    pub usage: [f32; LEGS],
}

#[derive(Debug, Clone)]
pub struct PiiSignalPrediction {
    pub probs: Vec<f64>,
    pub triggered_classes: Vec<String>,
    pub presence_prob: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArcModelEntry {
    pub name: String,
    pub wire: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArcActionEntry {
    pub model: usize,
    pub effort: String,
    #[serde(default)]
    pub level: Option<String>,
    #[serde(default)]
    pub arms: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArcStageTwoMeta {
    pub cost_weight: f64,
    pub support_columns: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArcRuntimeJson {
    pub models: Vec<ArcModelEntry>,
    pub actions: Vec<ArcActionEntry>,
    pub fallback_action: usize,
    pub stage_two: ArcStageTwoMeta,
}

#[derive(Debug, Clone, Deserialize)]
struct PricesFile {
    models: HashMap<String, [f64; LEGS]>,
}

#[derive(Debug, Clone)]
pub struct PiiLengthBucket {
    pub max_tokens: usize,
    pub threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ArcRuntimeCatalog {
    pub models: Vec<ArcModelEntry>,
    pub actions: Vec<ArcActionEntry>,
    pub fallback_action: usize,
    pub stage_two_cost_weight: f64,
    pub stage_two_support_columns: Vec<String>,
    /// Per-model 4-leg prices (USD per million tokens), indexed by `model_idx`.
    pub model_prices: Vec<Option<[f64; LEGS]>>,
    pub pii_classes: Vec<String>,
    pub pii_presence_threshold: f64,
    pub pii_type_threshold: f64,
    pub pii_length_buckets: Vec<PiiLengthBucket>,
}

impl ArcRuntimeCatalog {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let rt_bytes = std::fs::read(model_dir.join("runtime/arc-runtime.json"))
            .map_err(|e| format!("read runtime/arc-runtime.json: {e}"))?;
        let rt: ArcRuntimeJson = serde_json::from_slice(&rt_bytes)
            .map_err(|e| format!("parse runtime/arc-runtime.json: {e}"))?;

        let prices_bytes = std::fs::read(model_dir.join("runtime/prices.json"))
            .map_err(|e| format!("read runtime/prices.json: {e}"))?;
        let prices: PricesFile = serde_json::from_slice(&prices_bytes)
            .map_err(|e| format!("parse runtime/prices.json: {e}"))?;

        let model_prices = rt
            .models
            .iter()
            .map(|m| {
                m.wire
                    .as_deref()
                    .and_then(|w| prices.models.get(w).copied())
            })
            .collect();

        let cfg_bytes = std::fs::read(model_dir.join("config.json"))
            .map_err(|e| format!("read config.json: {e}"))?;
        let cfg: serde_json::Value =
            serde_json::from_slice(&cfg_bytes).map_err(|e| format!("parse config.json: {e}"))?;
        let pii = cfg
            .pointer("/arc/signals/heads/pii.payload")
            .ok_or_else(|| "config.json missing /arc/signals/heads/pii.payload".to_string())?;
        let pii_classes: Vec<String> = pii
            .get("classes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "pii.payload missing classes".to_string())?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        let pii_presence_threshold = pii
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.186684);
        let pii_type_threshold = pii
            .get("type_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let pii_length_buckets = pii
            .get("length_buckets")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|b| {
                        Some(PiiLengthBucket {
                            max_tokens: b.get("max_tokens")?.as_u64()? as usize,
                            threshold: b.get("threshold")?.as_f64()?,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            models: rt.models,
            actions: rt.actions,
            fallback_action: rt.fallback_action,
            stage_two_cost_weight: rt.stage_two.cost_weight,
            stage_two_support_columns: rt.stage_two.support_columns,
            model_prices,
            pii_classes,
            pii_presence_threshold,
            pii_type_threshold,
            pii_length_buckets,
        })
    }

    pub fn pii_threshold_for_tokens(&self, prompt_tokens: usize) -> f64 {
        for b in &self.pii_length_buckets {
            if prompt_tokens <= b.max_tokens {
                return b.threshold;
            }
        }
        self.pii_length_buckets
            .last()
            .map_or(self.pii_presence_threshold, |b| b.threshold)
    }
}

fn read_shape(archive: &SafetensorsArchive, name: &str) -> Result<Vec<usize>, String> {
    archive
        .tensor_info(name)
        .map(|m| m.shape.clone())
        .map_err(|e| format!("missing tensor `{name}`: {e}"))
}

fn read_bytes(
    archive: &SafetensorsArchive,
    name: &str,
    expected: &[usize],
    expected_dtype: Option<&str>,
    elem_bytes: usize,
) -> Result<Vec<u8>, String> {
    let info = archive
        .tensor_info(name)
        .map_err(|e| format!("missing tensor `{name}`: {e}"))?;
    if info.shape != expected {
        return Err(format!(
            "tensor `{name}` shape {:?} != expected {:?}",
            info.shape, expected
        ));
    }
    if let Some(dt) = expected_dtype
        && info.dtype != dt
    {
        return Err(format!("tensor `{name}` dtype {} != {dt}", info.dtype));
    }
    let expected_bytes = expected
        .iter()
        .try_fold(elem_bytes, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("tensor `{name}` byte size overflow"))?;
    if info.len_bytes as usize != expected_bytes {
        return Err(format!(
            "tensor `{name}` byte length {} != expected {expected_bytes}",
            info.len_bytes
        ));
    }
    let mut bytes = vec![0u8; expected_bytes];
    archive
        .read_exact_info(info, &mut bytes)
        .map_err(|e| format!("read `{name}`: {e}"))?;
    Ok(bytes)
}

fn read_f64(
    archive: &SafetensorsArchive,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f64>, String> {
    let bytes = read_bytes(archive, name, expected, Some("F64"), 8)?;
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .copied()
        .map(f64::from_le_bytes)
        .collect())
}

fn read_f32(
    archive: &SafetensorsArchive,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, String> {
    let bytes = read_bytes(archive, name, expected, Some("F32"), 4)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect())
}

fn read_i64(
    archive: &SafetensorsArchive,
    name: &str,
    expected: &[usize],
) -> Result<Vec<i64>, String> {
    let bytes = read_bytes(archive, name, expected, Some("I64"), 8)?;
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .copied()
        .map(i64::from_le_bytes)
        .collect())
}

fn read_flags(
    archive: &SafetensorsArchive,
    name: &str,
    expected: &[usize],
) -> Result<Vec<bool>, String> {
    let bytes = read_bytes(archive, name, expected, None, 1)?;
    Ok(bytes.iter().map(|&b| b != 0).collect())
}

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn upper_bound(sorted: &[f64], value: f64) -> usize {
    let (mut start, mut end) = (0, sorted.len());
    while start < end {
        let mid = start + ((end - start) >> 1);
        if sorted[mid] > value {
            end = mid;
        } else {
            start = mid + 1;
        }
    }
    start
}

fn linear_quantile(pool: &[f64], n: i64, q: f64) -> f64 {
    let last = (n - 1).max(0) as usize;
    let virtual_index = (n - 1) as f64 * q;
    let below = virtual_index.floor();
    let previous = (below.max(0.0) as usize).min(last);
    let following = (previous + 1).min(last);
    let (a, b) = (pool[previous], pool[following]);
    let gamma = virtual_index - below;
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

fn order_ratio(glm: f64, net: f64, pool_glm: &[f64], pool_net: &[f64], n: i64, alpha: f64) -> f64 {
    if !(glm.is_finite() && net.is_finite()) || n < MIN_POOL {
        return 1.0;
    }
    let count = n.max(1) as f64;
    let u = upper_bound(pool_net, net) as f64 / count;
    let v = upper_bound(pool_glm, glm) as f64 / count;
    let top = linear_quantile(pool_glm, n, u.clamp(0.0, 1.0)).max(FLOOR_TOKENS);
    let bottom = linear_quantile(pool_glm, n, v.clamp(0.0, 1.0)).max(FLOOR_TOKENS);
    (top / bottom).powf(alpha)
}

fn interp(x: f64, xp: &[f64], fp: &[f64], n: i64) -> f64 {
    let n = (n.max(0) as usize).min(xp.len()).min(fp.len());
    if n == 0 {
        return 0.0;
    }
    if n == 1 || x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let idx = upper_bound(&xp[..n], x).clamp(1, n - 1);
    let (x0, x1) = (xp[idx - 1], xp[idx]);
    let (y0, y1) = (fp[idx - 1], fp[idx]);
    let t = if (x1 - x0).abs() > 0.0 {
        (x - x0) / (x1 - x0)
    } else {
        0.0
    };
    y0 + t * (y1 - y0)
}

fn platt_f64(p: f64, on: bool, a: f64, b: f64) -> f64 {
    if !p.is_finite() || !on {
        return p;
    }
    let clipped = p.clamp(PLATT_EPS, 1.0 - PLATT_EPS);
    let logit = (clipped / (1.0 - clipped)).ln();
    sigmoid_f64((a * logit + b).clamp(-40.0, 40.0))
}

struct Orderer {
    axes_count: usize,
    hidden: usize,
    pool: usize,
    knots: usize,
    mean: Vec<f64>,
    axes: Vec<f64>,
    center: Vec<f64>,
    spread: Vec<f64>,
    w1: Vec<f64>,
    e1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    level: Vec<f64>,
    b: Vec<f64>,
    seen: Vec<bool>,
    pool_glm: Vec<f64>,
    pool_net: Vec<f64>,
    pool_n: Vec<i64>,
    alpha: f64,
    recal_x: Vec<f64>,
    recal_c: Vec<f64>,
    recal_n: Vec<i64>,
}

pub struct StageOneHead {
    members: usize,
    candidates: usize,
    dim: usize,
    rank: usize,
    hidden: usize,
    embed_dim: usize,
    regimes: usize,
    mean: Vec<f64>,
    scale: Vec<f64>,
    logit_bias: Vec<f64>,
    regime: Vec<f64>,
    difficulty: Vec<f64>,
    v: Vec<f64>,
    u: Vec<f64>,
    w1: Vec<f64>,
    c1: Vec<f64>,
    w2: Vec<f64>,
    c2: Vec<f64>,
    embed: Vec<f64>,
    p_ok: Vec<bool>,
    w_usage: Vec<f64>,
    b_usage: Vec<f64>,
    smear: Vec<f64>,
    low: Vec<f64>,
    high: Vec<f64>,
    usage_ok: Vec<bool>,
    orderer: Orderer,
    ens_point: Vec<usize>,
    cal_platt: Vec<bool>,
    cal_a: Vec<f64>,
    cal_b: Vec<f64>,
}

impl StageOneHead {
    #[allow(clippy::cognitive_complexity)]
    pub fn load(archive: &SafetensorsArchive) -> Result<Self, String> {
        let p = |k: &str| format!("stage_one.{k}");
        let w1_shape = read_shape(archive, &p("w1"))?;
        let [members, dim, hidden] = w1_shape.as_slice() else {
            return Err(format!("unexpected stage_one.w1 shape {w1_shape:?}"));
        };
        let embed_shape = read_shape(archive, &p("embed"))?;
        let [_, candidates, embed_dim] = embed_shape.as_slice() else {
            return Err(format!("unexpected stage_one.embed shape {embed_shape:?}"));
        };
        let v_shape = read_shape(archive, &p("v"))?;
        let [_, rank, _] = v_shape.as_slice() else {
            return Err(format!("unexpected stage_one.v shape {v_shape:?}"));
        };
        let reg_shape = read_shape(archive, &p("regime"))?;
        let [_, regimes] = reg_shape.as_slice() else {
            return Err(format!("unexpected stage_one.regime shape {reg_shape:?}"));
        };
        let ord_w1_shape = read_shape(archive, &p("ord_w1"))?;
        let [ord_legs, ord_axes, ord_hidden] = ord_w1_shape.as_slice() else {
            return Err(format!(
                "unexpected stage_one.ord_w1 shape {ord_w1_shape:?}"
            ));
        };
        let pool_shape = read_shape(archive, &p("ord_pool_glm"))?;
        let [_, _, pool] = pool_shape.as_slice() else {
            return Err(format!(
                "unexpected stage_one.ord_pool_glm shape {pool_shape:?}"
            ));
        };
        let recal_shape = read_shape(archive, &p("ord_recal_x"))?;
        let [_, _, knots] = recal_shape.as_slice() else {
            return Err(format!(
                "unexpected stage_one.ord_recal_x shape {recal_shape:?}"
            ));
        };
        if *ord_legs != LEGS {
            return Err(format!("stage_one ord_legs={ord_legs} != {LEGS}"));
        }
        let (m, c, d, r, h, e, reg) = (
            *members,
            *candidates,
            *dim,
            *rank,
            *hidden,
            *embed_dim,
            *regimes,
        );
        let (oa, oh, pl, kn) = (*ord_axes, *ord_hidden, *pool, *knots);
        let ens_shape = read_shape(archive, &p("ens_point"))?;
        let points = *ens_shape
            .first()
            .ok_or_else(|| "empty stage_one.ens_point".to_string())?;
        let ens_point: Vec<usize> = read_i64(archive, &p("ens_point"), &[points])?
            .into_iter()
            .map(|i| usize::try_from(i).map_err(|_| format!("invalid ens_point {i}")))
            .collect::<Result<_, _>>()?;

        let orderer = Orderer {
            axes_count: oa,
            hidden: oh,
            pool: pl,
            knots: kn,
            mean: read_f64(archive, &p("ord_mean"), &[d])?,
            axes: read_f64(archive, &p("ord_axes"), &[oa, d])?,
            center: read_f64(archive, &p("ord_center"), &[LEGS, oa])?,
            spread: read_f64(archive, &p("ord_spread"), &[LEGS, oa])?,
            w1: read_f64(archive, &p("ord_w1"), &[LEGS, oa, oh])?,
            e1: read_f64(archive, &p("ord_e1"), &[LEGS, c, oh])?,
            b1: read_f64(archive, &p("ord_b1"), &[LEGS, oh])?,
            w2: read_f64(archive, &p("ord_w2"), &[LEGS, oh])?,
            level: read_f64(archive, &p("ord_level"), &[LEGS, c])?,
            b: read_f64(archive, &p("ord_b"), &[LEGS])?,
            seen: read_flags(archive, &p("ord_seen"), &[LEGS, c])?,
            pool_glm: read_f64(archive, &p("ord_pool_glm"), &[c, LEGS, pl])?,
            pool_net: read_f64(archive, &p("ord_pool_net"), &[c, LEGS, pl])?,
            pool_n: read_i64(archive, &p("ord_pool_n"), &[c, LEGS])?,
            alpha: read_f64(archive, &p("ord_alpha"), &[1])?[0],
            recal_x: read_f64(archive, &p("ord_recal_x"), &[c, LEGS, kn])?,
            recal_c: read_f64(archive, &p("ord_recal_c"), &[c, LEGS, kn])?,
            recal_n: read_i64(archive, &p("ord_recal_n"), &[c, LEGS])?,
        };

        Ok(Self {
            members: m,
            candidates: c,
            dim: d,
            rank: r,
            hidden: h,
            embed_dim: e,
            regimes: reg,
            mean: read_f64(archive, &p("mean"), &[d])?,
            scale: read_f64(archive, &p("scale"), &[d])?,
            logit_bias: read_f64(archive, &p("logit_bias"), &[m, c])?,
            regime: read_f64(archive, &p("regime"), &[m, reg])?,
            difficulty: read_f64(archive, &p("difficulty"), &[m, d])?,
            v: read_f64(archive, &p("v"), &[m, r, d])?,
            u: read_f64(archive, &p("u"), &[m, c, r])?,
            w1: read_f64(archive, &p("w1"), &[m, d, h])?,
            c1: read_f64(archive, &p("c1"), &[m, h])?,
            w2: read_f64(archive, &p("w2"), &[m, h, e])?,
            c2: read_f64(archive, &p("c2"), &[m, e])?,
            embed: read_f64(archive, &p("embed"), &[m, c, e])?,
            p_ok: read_flags(archive, &p("p_ok"), &[m, c])?,
            w_usage: read_f64(archive, &p("w_usage"), &[m, c, LEGS, d])?,
            b_usage: read_f64(archive, &p("b_usage"), &[m, c, LEGS])?,
            smear: read_f64(archive, &p("smear"), &[m, c, LEGS])?,
            low: read_f64(archive, &p("low"), &[m, c, LEGS])?,
            high: read_f64(archive, &p("high"), &[m, c, LEGS])?,
            usage_ok: read_flags(archive, &p("usage_ok"), &[m, c, LEGS])?,
            orderer,
            ens_point,
            cal_platt: read_flags(archive, &p("cal_platt"), &[points, c])?,
            cal_a: read_f64(archive, &p("cal_a"), &[points, c])?,
            cal_b: read_f64(archive, &p("cal_b"), &[points, c])?,
        })
    }

    pub fn predict(
        &self,
        readout: &[f32],
        regime: usize,
    ) -> Result<Vec<StageOnePrediction>, String> {
        if readout.len() != self.dim || regime >= self.regimes {
            return Err(format!(
                "stage_one readout dim {} (expected {}) or regime {} out of range",
                readout.len(),
                self.dim,
                regime
            ));
        }
        let (m_count, c_count, d) = (self.members, self.candidates, self.dim);
        let z: Vec<f64> = readout
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((x, mean), scale)| (f64::from(*x) - mean) * scale)
            .collect();
        let net = self.orderer_net(&z);

        let mut p = vec![f64::NAN; m_count * c_count];
        let mut usage = vec![f64::NAN; m_count * c_count * LEGS];
        for m in 0..m_count {
            let difficulty = dot_f64(&z, &self.difficulty[m * d..(m + 1) * d]);
            let factors: Vec<f64> = (0..self.rank)
                .map(|r| {
                    dot_f64(
                        &z,
                        &self.v[(m * self.rank + r) * d..(m * self.rank + r + 1) * d],
                    )
                })
                .collect();
            let mut hidden = self.c1[m * self.hidden..(m + 1) * self.hidden].to_vec();
            for (k, zk) in z.iter().enumerate() {
                let row = &self.w1[(m * d + k) * self.hidden..(m * d + k + 1) * self.hidden];
                for (h, w) in hidden.iter_mut().zip(row) {
                    *h += zk * w;
                }
            }
            let mut projected = self.c2[m * self.embed_dim..(m + 1) * self.embed_dim].to_vec();
            for (h, &value) in hidden.iter().enumerate() {
                if value > 0.0 {
                    let row = &self.w2[(m * self.hidden + h) * self.embed_dim
                        ..(m * self.hidden + h + 1) * self.embed_dim];
                    for (e, w) in projected.iter_mut().zip(row) {
                        *e += value * w;
                    }
                }
            }
            for c in 0..c_count {
                let mc = m * c_count + c;
                if self.p_ok[mc] {
                    let interaction =
                        dot_f64(&factors, &self.u[mc * self.rank..(mc + 1) * self.rank]);
                    let embed = dot_f64(
                        &projected,
                        &self.embed[mc * self.embed_dim..(mc + 1) * self.embed_dim],
                    );
                    let eta = self.logit_bias[mc] + self.regime[m * self.regimes + regime]
                        - difficulty
                        + interaction
                        + embed;
                    p[mc] = sigmoid_f64(eta.clamp(-40.0, 40.0));
                }
                for leg in 0..LEGS {
                    let i = mc * LEGS + leg;
                    if !self.usage_ok[i] {
                        continue;
                    }
                    let exponent = dot_f64(&z, &self.w_usage[i * d..(i + 1) * d]) + self.b_usage[i];
                    let glm = (exponent.exp() * self.smear[i] - 1.0)
                        .max(self.low[i])
                        .min(self.high[i]);
                    usage[i] = self.ordered(glm, net[leg * c_count + c], c, leg);
                }
            }
        }

        let points = self.ens_point.len();
        Ok((0..c_count)
            .map(|c| {
                let (mut total, mut count) = (0.0, 0usize);
                let mut legs = [0.0; LEGS];
                for (i, &member) in self.ens_point.iter().enumerate() {
                    let mapped = platt_f64(
                        p[member * c_count + c],
                        self.cal_platt[i * c_count + c],
                        self.cal_a[i * c_count + c],
                        self.cal_b[i * c_count + c],
                    );
                    if mapped.is_finite() {
                        total += mapped;
                        count += 1;
                    }
                    for (leg, sum) in legs.iter_mut().enumerate() {
                        *sum += usage[(member * c_count + c) * LEGS + leg];
                    }
                }
                StageOnePrediction {
                    p_pass: if count > 0 {
                        total / count as f64
                    } else {
                        f64::NAN
                    },
                    usage: legs.map(|sum| sum / points as f64),
                }
            })
            .collect())
    }

    fn orderer_net(&self, z: &[f64]) -> Vec<f64> {
        let o = &self.orderer;
        let (c_count, d) = (self.candidates, self.dim);
        let centered: Vec<f64> = z.iter().zip(&o.mean).map(|(z, mean)| z - mean).collect();
        let projected: Vec<f64> = (0..o.axes_count)
            .map(|j| dot_f64(&centered, &o.axes[j * d..(j + 1) * d]))
            .collect();
        let mut net = vec![f64::NAN; LEGS * c_count];
        for leg in 0..LEGS {
            let mut shared = o.b1[leg * o.hidden..(leg + 1) * o.hidden].to_vec();
            for (j, &value) in projected.iter().enumerate() {
                let x =
                    (value - o.center[leg * o.axes_count + j]) / o.spread[leg * o.axes_count + j];
                let row = &o.w1
                    [(leg * o.axes_count + j) * o.hidden..(leg * o.axes_count + j + 1) * o.hidden];
                for (h, w) in shared.iter_mut().zip(row) {
                    *h += x * w;
                }
            }
            let w2 = &o.w2[leg * o.hidden..(leg + 1) * o.hidden];
            for c in 0..c_count {
                let lc = leg * c_count + c;
                if !o.seen[lc] {
                    continue;
                }
                let e1 = &o.e1[lc * o.hidden..(lc + 1) * o.hidden];
                let eta: f64 = shared
                    .iter()
                    .zip(e1)
                    .zip(w2)
                    .map(|((h, e), w)| (h + e).max(0.0) * w)
                    .sum::<f64>()
                    + o.level[lc]
                    + o.b[leg];
                net[lc] = eta.exp();
            }
        }
        net
    }

    fn ordered(&self, glm: f64, net: f64, candidate: usize, leg: usize) -> f64 {
        let o = &self.orderer;
        let cl = candidate * LEGS + leg;
        let out = glm
            * order_ratio(
                glm,
                net,
                &o.pool_glm[cl * o.pool..(cl + 1) * o.pool],
                &o.pool_net[cl * o.pool..(cl + 1) * o.pool],
                o.pool_n[cl],
                o.alpha,
            );
        if !(out.is_finite() && out > 0.0 && o.recal_n[cl] > 0) {
            return out;
        }
        let log = out.ln();
        let shift = interp(
            log,
            &o.recal_x[cl * o.knots..(cl + 1) * o.knots],
            &o.recal_c[cl * o.knots..(cl + 1) * o.knots],
            o.recal_n[cl],
        );
        (log + shift).exp()
    }
}

struct LinearF32 {
    rows: usize,
    columns: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearF32 {
    fn load(archive: &SafetensorsArchive, prefix: &str) -> Result<Self, String> {
        let w_name = format!("{prefix}.weight");
        let b_name = format!("{prefix}.bias");
        let shape = read_shape(archive, &w_name)?;
        let [rows, columns] = shape.as_slice() else {
            return Err(format!("`{w_name}` is not a 2D matrix: {shape:?}"));
        };
        Ok(Self {
            rows: *rows,
            columns: *columns,
            weight: read_f32(archive, &w_name, &[*rows, *columns])?,
            bias: read_f32(archive, &b_name, &[*rows])?,
        })
    }

    fn apply(&self, input: &[f32], offset: usize, out: &mut [f32], accumulate: bool) {
        for (row, value) in out.iter_mut().enumerate() {
            let weights = &self.weight[row * self.columns + offset..][..input.len()];
            let sum: f32 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
            *value = if accumulate {
                *value + sum
            } else {
                self.bias[row] + sum
            };
        }
    }
}

pub struct StageTwoHead {
    arms: usize,
    dim: usize,
    hidden: usize,
    arm_dim: usize,
    role_dim: usize,
    arm_vectors: Vec<f32>,
    roles: Vec<f32>,
    context: LinearF32,
    history: LinearF32,
    trunk_in: LinearF32,
    trunk_out: LinearF32,
    quality: LinearF32,
    usage: LinearF32,
    pub quality_support: Vec<bool>,
}

impl StageTwoHead {
    pub fn load(archive: &SafetensorsArchive, support_columns: &[String]) -> Result<Self, String> {
        let p = |k: &str| format!("stage_two.{k}");
        let idx_shape = read_shape(archive, &p("arms.model_index"))?;
        let [rows] = idx_shape.as_slice() else {
            return Err(format!("unexpected arms.model_index shape {idx_shape:?}"));
        };
        let arms = rows
            .checked_sub(1)
            .ok_or_else(|| "stage_two arm table is empty".to_string())?;
        let mut parts = Vec::new();
        let mut arm_dim = 0;
        for part in ["model", "effort", "configuration", "thinking"] {
            let w_name = p(&format!("arms.{part}_vectors.weight"));
            let shape = read_shape(archive, &w_name)?;
            let [count, width] = shape.as_slice() else {
                return Err(format!("`{w_name}` not 2D: {shape:?}"));
            };
            let index = read_i64(archive, &p(&format!("arms.{part}_index")), &[*rows])?;
            let table = read_f32(archive, &w_name, &[*count, *width])?;
            parts.push((index, table, *width));
            arm_dim += *width;
        }
        let mut arm_vectors = Vec::with_capacity(arms * arm_dim);
        for arm in 1..*rows {
            for (index, table, width) in &parts {
                let r = index[arm] as usize;
                arm_vectors.extend_from_slice(&table[r * width..(r + 1) * width]);
            }
        }
        let role_shape = read_shape(archive, &p("roles.weight"))?;
        let [role_count, role_dim] = role_shape.as_slice() else {
            return Err(format!("unexpected roles.weight shape {role_shape:?}"));
        };
        let context = LinearF32::load(archive, &p("context"))?;
        let history = LinearF32::load(archive, &p("history_projection"))?;
        let trunk_in = LinearF32::load(archive, &p("trunk.0"))?;
        let trunk_out = LinearF32::load(archive, &p("trunk.3"))?;
        let quality = LinearF32::load(archive, &p("quality"))?;
        let usage = LinearF32::load(archive, &p("log_usage"))?;
        let width = support_columns.len();
        let support = read_i64(archive, &p("meta.support"), &[arms, width])?;
        let q_col = support_columns
            .iter()
            .position(|c| c == "quality")
            .ok_or_else(|| "stage_two support_columns missing `quality`".to_string())?;
        Ok(Self {
            arms,
            dim: context.columns,
            hidden: context.rows,
            arm_dim,
            role_dim: *role_dim,
            arm_vectors,
            roles: read_f32(archive, &p("roles.weight"), &[*role_count, *role_dim])?,
            context,
            history,
            trunk_in,
            trunk_out,
            quality,
            usage,
            quality_support: (0..arms)
                .map(|arm| support[arm * width + q_col] > 0)
                .collect(),
        })
    }

    /// Score a subset of arms using multi-block pooled history (`rayline-arc-decide` parity).
    pub fn predict_arms(
        &self,
        block_readouts: &[Vec<f32>],
        block_roles: &[String],
        block_source_arms: &[Option<usize>],
        arm_indices: &[usize],
    ) -> Result<Vec<(usize, ArmScore)>, String> {
        let steps = block_readouts.len();
        if steps == 0 || block_roles.len() != steps {
            return Err(format!(
                "stage_two requires non-empty matching block_readouts ({steps}) and block_roles ({})",
                block_roles.len()
            ));
        }
        let mut pooled = vec![0f32; self.dim + self.arm_dim + self.role_dim];
        for (i, readout) in block_readouts.iter().enumerate() {
            if readout.len() != self.dim {
                return Err(format!(
                    "stage_two readout[{i}] dim {} != {}",
                    readout.len(),
                    self.dim
                ));
            }
            for (dst, &v) in pooled[..self.dim].iter_mut().zip(readout.iter()) {
                *dst += v;
            }
            if let Some(src_arm) = block_source_arms.get(i).copied().flatten()
                && src_arm < self.arms
            {
                let src_vec =
                    &self.arm_vectors[src_arm * self.arm_dim..(src_arm + 1) * self.arm_dim];
                for (dst, &v) in pooled[self.dim..self.dim + self.arm_dim]
                    .iter_mut()
                    .zip(src_vec.iter())
                {
                    *dst += v;
                }
            }
            let role_idx = ROLES
                .iter()
                .position(|&r| r == block_roles[i].as_str())
                .unwrap_or(1);
            let role_emb = &self.roles[role_idx * self.role_dim..(role_idx + 1) * self.role_dim];
            for (dst, &v) in pooled[self.dim + self.arm_dim..]
                .iter_mut()
                .zip(role_emb.iter())
            {
                *dst += v;
            }
        }
        let inv_steps = 1.0f32 / (steps as f32);
        for v in &mut pooled {
            *v *= inv_steps;
        }
        let latest_readout = &block_readouts[steps - 1];

        let hidden = self.hidden;
        let mut shared = vec![0f32; 2 * hidden];
        let (semantic, state) = shared.split_at_mut(hidden);
        self.context.apply(latest_readout, 0, semantic, false);
        self.history.apply(&pooled, 0, state, false);
        let mut base = vec![0f32; hidden];
        self.trunk_in.apply(&shared, 0, &mut base, false);
        let arm_offset = 3 * hidden;

        let mut first = vec![0f32; hidden];
        let mut latent = vec![0f32; hidden];
        let mut out = Vec::with_capacity(arm_indices.len());
        for &arm in arm_indices {
            if arm >= self.arms {
                continue;
            }
            first.copy_from_slice(&base);
            let vec = &self.arm_vectors[arm * self.arm_dim..(arm + 1) * self.arm_dim];
            self.trunk_in.apply(vec, arm_offset, &mut first, true);
            for v in &mut first {
                *v = v.max(0.0);
            }
            self.trunk_out.apply(&first, 0, &mut latent, false);
            for v in &mut latent {
                *v = v.max(0.0);
            }
            let mut q = [0f32; 1];
            self.quality.apply(&latent, 0, &mut q, false);
            let mut u = [0f32; LEGS];
            self.usage.apply(&latent, 0, &mut u, false);
            out.push((
                arm,
                ArmScore {
                    quality: sigmoid_f32(q[0]),
                    usage: u.map(|raw| softplus_f32(raw) * USAGE_SCALE),
                },
            ));
        }
        Ok(out)
    }
}

pub struct PiiPayloadHead {
    dim: usize,
    mean: Vec<f32>,
    scale: Vec<f32>,
    fc0: LinearF32,
    fc2: LinearF32,
    /// Platt `[class_count][2]` (`a`, `b`).
    platt: Vec<f64>,
    classes: Vec<String>,
}

impl PiiPayloadHead {
    pub fn load(archive: &SafetensorsArchive, classes: Vec<String>) -> Result<Self, String> {
        let p = |k: &str| format!("signals.pii.payload.{k}");
        let fc0 = LinearF32::load(archive, &p("net.0"))?;
        let fc2 = LinearF32::load(archive, &p("net.2"))?;
        let dim = fc0.columns;
        let num_classes = fc2.rows;
        if classes.len() != num_classes {
            return Err(format!(
                "pii.payload classes len {} != output dim {}",
                classes.len(),
                num_classes
            ));
        }
        Ok(Self {
            dim,
            mean: read_f32(archive, &p("mean"), &[dim])?,
            scale: read_f32(archive, &p("scale"), &[dim])?,
            fc0,
            fc2,
            platt: read_f64(archive, &p("platt"), &[num_classes, 2])?,
            classes,
        })
    }

    pub fn predict(
        &self,
        readout: &[f32],
        presence_threshold: f64,
        type_threshold: f64,
    ) -> Result<PiiSignalPrediction, String> {
        if readout.len() != self.dim {
            return Err(format!(
                "pii.payload readout dim {} != {}",
                readout.len(),
                self.dim
            ));
        }
        let z: Vec<f32> = readout
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((&x, &m), &s)| if s != 0.0 { (x - m) / s } else { 0.0 })
            .collect();
        let mut h = vec![0f32; self.fc0.rows];
        self.fc0.apply(&z, 0, &mut h, false);
        for v in &mut h {
            *v = v.max(0.0);
        }
        let mut logits = vec![0f32; self.fc2.rows];
        self.fc2.apply(&h, 0, &mut logits, false);

        let mut probs = Vec::with_capacity(self.classes.len());
        let mut triggered = Vec::new();
        let mut presence_prob = 0.0f64;
        for (i, &logit) in logits.iter().enumerate() {
            let a = self.platt[i * 2];
            let b = self.platt[i * 2 + 1];
            let prob = sigmoid_f64((a * f64::from(logit) + b).clamp(-40.0, 40.0));
            probs.push(prob);
            let cls = &self.classes[i];
            if cls == "any" {
                presence_prob = prob;
            } else if prob >= type_threshold {
                triggered.push(cls.clone());
            }
        }
        if presence_prob >= presence_threshold && !triggered.iter().any(|c| c == "any") {
            triggered.push("any".to_string());
        }
        Ok(PiiSignalPrediction {
            probs,
            triggered_classes: triggered,
            presence_prob,
        })
    }
}

pub struct ArcHeads {
    pub catalog: ArcRuntimeCatalog,
    pub stage_one: StageOneHead,
    pub stage_two: StageTwoHead,
    pub pii: PiiPayloadHead,
}

impl ArcHeads {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let catalog = ArcRuntimeCatalog::load(model_dir)?;
        let heads_path = model_dir.join("runtime/heads.safetensors");
        let model_path = model_dir.join("model.safetensors");
        let heads_archive = SafetensorsArchive::open(&heads_path)
            .map_err(|e| format!("open {}: {e}", heads_path.display()))?;
        let model_archive = SafetensorsArchive::open(&model_path)
            .map_err(|e| format!("open {}: {e}", model_path.display()))?;
        let stage_one = StageOneHead::load(&heads_archive)?;
        let stage_two = StageTwoHead::load(&heads_archive, &catalog.stage_two_support_columns)?;
        let pii = PiiPayloadHead::load(&model_archive, catalog.pii_classes.clone())?;
        Ok(Self {
            catalog,
            stage_one,
            stage_two,
            pii,
        })
    }
}
