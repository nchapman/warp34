//! Strict HGPTQ metadata. Keep patterns as strings so configuration equality is semantic.
use super::*;
use fancy_regex::{Regex, RegexBuilder};

pub const HADAMARD_BLOCK_16: usize = 16;
pub const HADAMARD_BLOCK_64: usize = 64;
pub const HADAMARD_BLOCK_128: usize = 128;
pub const HADAMARD_BLOCK_512: usize = 512;
pub const HADAMARD_BLOCK_1024: usize = 1024;
pub const HADAMARD_BLOCK_4096: usize = 4096;
pub const SUPPORTED_HADAMARD_BLOCKS: [usize; 6] = [
    HADAMARD_BLOCK_16,
    HADAMARD_BLOCK_64,
    HADAMARD_BLOCK_128,
    HADAMARD_BLOCK_512,
    HADAMARD_BLOCK_1024,
    HADAMARD_BLOCK_4096,
];
pub const SUPPORTED_MARLIN_U2_GROUP_SIZES: [usize; 4] = [0, 32, 64, 128];
pub const SUPPORTED_MARLIN_U4_GROUP_SIZES: [usize; 4] = [0, 32, 64, 128];
pub const SUPPORTED_MARLIN_U8_GROUP_SIZES: [usize; 4] = [0, 32, 64, 128];

/// Returns `true` when `block` is either disabled (`0`) or a supported Hadamard block size
/// (`16`, `64`, `128`, `512`, `1024`, or `4096`) that evenly divides non-zero `dim`.
#[inline]
pub const fn is_valid_hadamard_block_for_dim(block: usize, dim: usize) -> bool {
    match block {
        0 => true,
        HADAMARD_BLOCK_16 | HADAMARD_BLOCK_64 | HADAMARD_BLOCK_128 | HADAMARD_BLOCK_512
        | HADAMARD_BLOCK_1024 | HADAMARD_BLOCK_4096 => dim != 0 && dim.is_multiple_of(block),
        _ => false,
    }
}

const MAX_REGEX_PATTERN_BYTES: usize = 8 * 1024;
const REGEX_BACKTRACK_LIMIT: usize = 100_000;
const MAX_COMPRESSED_TENSORS_PATTERNS: usize = 1024;
const MAX_COMPRESSED_TENSORS_GROUPS: usize = 32;
const MAX_TENSOR_NAME_BYTES: usize = 16 * 1024;
const COMPRESSED_TENSORS_U2_BITS: usize = 2;
const COMPRESSED_TENSORS_U4_BITS: usize = 4;
const COMPRESSED_TENSORS_U8_BITS: usize = 8;

type MoeExpertQuantInfo = (usize, usize, bool, bool);
pub type InputHadamardSpec = (usize, Option<std::sync::Arc<[i8]>>);

/// Post-dequantization transform metadata for quantized `embed_tokens` (`warp_quant_embed_w2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbedQuantSpec {
    pub group_size: usize,
    pub hadamard_block_size: usize,
    pub signs: std::sync::Arc<[i8]>,
    pub tensor_suffix: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CompressedTensorRules {
    ignore: Vec<CompiledPattern>,
    groups: Vec<QuantRule>,
    transforms: Vec<TransformRule>,
    embed_quant: Option<EmbedQuantSpec>,
    moe_expert_layer_cache: std::sync::OnceLock<Vec<Option<MoeExpertQuantInfo>>>,
}

impl PartialEq for CompressedTensorRules {
    fn eq(&self, other: &Self) -> bool {
        self.ignore == other.ignore
            && self.groups == other.groups
            && self.transforms == other.transforms
            && self.embed_quant == other.embed_quant
    }
}

impl Eq for CompressedTensorRules {}

#[derive(Clone, Debug)]
enum CompiledPattern {
    Literal(String),
    Regex { raw: String, re: Regex },
}

impl PartialEq for CompiledPattern {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Literal(a), Self::Literal(b)) => a == b,
            (Self::Regex { raw: a, .. }, Self::Regex { raw: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl Eq for CompiledPattern {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct QuantRule {
    targets: Vec<CompiledPattern>,
    bits: usize,
    group_size: usize,
    symmetric: bool,
    code_offset_half: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TransformRule {
    targets: Vec<CompiledPattern>,
    ignore: Vec<CompiledPattern>,
    head_dim: usize,
    offline: bool,
    input_signs: Option<std::sync::Arc<[i8]>>,
}

fn parse_sign_array(v: &Json, block_size: usize, label: &str) -> io::Result<std::sync::Arc<[i8]>> {
    let arr = v
        .as_array()
        .ok_or_else(|| invalid(format!("{label} must be an array of -1/+1")))?;
    if arr.is_empty() || !arr.len().is_multiple_of(block_size) {
        return Err(invalid(format!(
            "{label} length ({}) must be a positive multiple of block_size {block_size}",
            arr.len()
        )));
    }
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        match item.as_i64() {
            Some(1) => out.push(1i8),
            Some(-1) => out.push(-1i8),
            _ => {
                return Err(invalid(format!("{label} entries must be -1 or 1")));
            }
        }
    }
    Ok(std::sync::Arc::from(out))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn regex(pattern: &str) -> io::Result<Regex> {
    if pattern.len() > MAX_REGEX_PATTERN_BYTES {
        return Err(invalid("HGPTQ regex exceeds size limit"));
    }
    RegexBuilder::new(&format!(r"\A(?:{pattern})"))
        .backtrack_limit(REGEX_BACKTRACK_LIMIT)
        .build()
        .map_err(|e| invalid(format!("invalid HGPTQ regex: {e}")))
}

fn compile_pattern(s: &str) -> io::Result<CompiledPattern> {
    if s.is_empty() || s.len() > MAX_REGEX_PATTERN_BYTES {
        return Err(invalid("HGPTQ pattern must be a bounded nonempty string"));
    }
    if let Some(p) = s.strip_prefix("re:") {
        Ok(CompiledPattern::Regex {
            raw: s.to_owned(),
            re: regex(p)?,
        })
    } else {
        Ok(CompiledPattern::Literal(s.to_owned()))
    }
}

fn patterns(value: Option<&Json>, required: bool) -> io::Result<Vec<CompiledPattern>> {
    let Some(value) = value else {
        return if required {
            Err(invalid("missing HGPTQ targets"))
        } else {
            Ok(vec![])
        };
    };
    let array = value
        .as_array()
        .ok_or_else(|| invalid("HGPTQ targets/ignore must be arrays"))?;
    if required && array.is_empty() {
        return Err(invalid("empty HGPTQ targets"));
    }
    if array.len() > MAX_COMPRESSED_TENSORS_PATTERNS {
        return Err(invalid("HGPTQ targets/ignore exceeds pattern count limit"));
    }
    array
        .iter()
        .map(|v| {
            let s = v
                .as_str()
                .ok_or_else(|| invalid("HGPTQ pattern must be a nonempty string"))?;
            compile_pattern(s)
        })
        .collect()
}

fn matches(patterns: &[CompiledPattern], name: &str) -> io::Result<bool> {
    if name.len() > MAX_TENSOR_NAME_BYTES {
        return Err(invalid("HGPTQ tensor name exceeds size limit"));
    }
    // Python re.match semantics: expressions are anchored at the start, not searched.
    for pattern in patterns {
        let matched = match pattern {
            CompiledPattern::Regex { re, .. } => re
                .find(name)
                .map_err(|e| invalid(format!("HGPTQ regex matching failed: {e}")))?
                .is_some_and(|m| m.start() == 0),
            CompiledPattern::Literal(lit) => {
                lit == "Linear"
                    || name == lit
                    || name
                        .strip_suffix(lit.as_str())
                        .is_some_and(|prefix| prefix.ends_with('.'))
            }
        };
        if matched {
            return Ok(true);
        }
    }
    Ok(false)
}

impl CompressedTensorRules {
    pub(super) fn parse(q: &Json) -> io::Result<Option<Self>> {
        let embed_quant = match q.get("warp_quant_embed_w2").filter(|v| !v.is_null()) {
            Some(v) => {
                let obj = v
                    .as_object()
                    .ok_or_else(|| invalid("warp_quant_embed_w2 must be an object"))?;
                let group_size = obj
                    .get("group_size")
                    .and_then(Json::as_u64)
                    .ok_or_else(|| invalid("warp_quant_embed_w2 missing group_size"))?
                    as usize;
                let hadamard_block_size = obj
                    .get("hadamard_block_size")
                    .and_then(Json::as_u64)
                    .ok_or_else(|| invalid("warp_quant_embed_w2 missing hadamard_block_size"))?
                    as usize;
                if !SUPPORTED_HADAMARD_BLOCKS.contains(&hadamard_block_size) {
                    return Err(invalid(
                        "unsupported warp_quant_embed_w2 hadamard_block_size",
                    ));
                }
                let signs = parse_sign_array(
                    obj.get("signs")
                        .ok_or_else(|| invalid("warp_quant_embed_w2 missing signs"))?,
                    hadamard_block_size,
                    "warp_quant_embed_w2.signs",
                )?;
                let tensor_suffix = obj
                    .get("tensor_suffix")
                    .and_then(Json::as_str)
                    .map(str::to_owned);
                Some(EmbedQuantSpec {
                    group_size,
                    hadamard_block_size,
                    signs,
                    tensor_suffix,
                })
            }
            None => None,
        };

        let mut transforms = vec![];
        if let Some(config) = q.get("transform_config").filter(|v| !v.is_null()) {
            let obj = config
                .as_object()
                .ok_or_else(|| invalid("transform_config must be an object"))?;
            if !obj.is_empty() {
                if obj.keys().any(|k| k != "config_groups") {
                    return Err(invalid("unsupported transform_config metadata"));
                }
                let groups = config
                    .get("config_groups")
                    .and_then(Json::as_object)
                    .ok_or_else(|| invalid("missing transform config_groups"))?;
                if groups.len() > MAX_COMPRESSED_TENSORS_GROUPS {
                    return Err(invalid("transform config_groups exceeds group count limit"));
                }
                let input_signs_map = match q.get("warp_quant_input_signs").filter(|v| !v.is_null())
                {
                    Some(v) => Some(
                        v.as_object()
                            .ok_or_else(|| invalid("warp_quant_input_signs must be an object"))?,
                    ),
                    None => None,
                };
                for (group_name, group) in groups {
                    let obj = group
                        .as_object()
                        .ok_or_else(|| invalid("invalid transform group"))?;
                    if obj.keys().any(|k| {
                        !matches!(
                            k.as_str(),
                            "type"
                                | "head_dim"
                                | "randomize"
                                | "requires_grad"
                                | "precision"
                                | "apply"
                        )
                    }) || group.get("type").and_then(Json::as_str) != Some("hadamard")
                        || group.get("randomize").and_then(Json::as_bool) != Some(false)
                        || group.get("requires_grad").is_some_and(|v| v != false)
                        || group
                            .get("precision")
                            .is_some_and(|v| v != "torch.float32" && v != "float32")
                    {
                        return Err(invalid(
                            "unsupported non-deterministic HGPTQ transform metadata",
                        ));
                    }
                    let head_dim = group
                        .get("head_dim")
                        .and_then(Json::as_u64)
                        .ok_or_else(|| invalid("missing Hadamard head_dim"))?
                        as usize;
                    if !SUPPORTED_HADAMARD_BLOCKS.contains(&head_dim) {
                        return Err(invalid("unsupported Hadamard head_dim"));
                    }
                    let group_signs = match input_signs_map.and_then(|m| m.get(group_name)) {
                        Some(v) => Some(parse_sign_array(
                            v,
                            head_dim,
                            &format!("warp_quant_input_signs.{group_name}"),
                        )?),
                        None => None,
                    };
                    let apply = group
                        .get("apply")
                        .and_then(Json::as_array)
                        .filter(|a| !a.is_empty())
                        .ok_or_else(|| invalid("missing transform apply rules"))?;
                    for rule in apply {
                        let obj = rule
                            .as_object()
                            .ok_or_else(|| invalid("invalid transform apply rule"))?;
                        if obj.keys().any(|k| {
                            !matches!(k.as_str(), "targets" | "ignore" | "location" | "inverse")
                        }) {
                            return Err(invalid("unsupported transform apply metadata"));
                        }
                        let offline = match (
                            rule.get("location").and_then(Json::as_str),
                            rule.get("inverse").and_then(Json::as_bool),
                        ) {
                            (Some("input"), Some(false)) => false,
                            (Some("weight_input"), Some(true)) => true,
                            _ => {
                                return Err(invalid(
                                    "unsupported transform location/inverse (only input and offline inverse weight_input supported)",
                                ));
                            }
                        };
                        transforms.push(TransformRule {
                            targets: patterns(rule.get("targets"), true)?,
                            ignore: patterns(rule.get("ignore"), false)?,
                            head_dim,
                            offline,
                            input_signs: if offline { None } else { group_signs.clone() },
                        });
                    }
                }
            }
        }
        if transforms.is_empty() && embed_quant.is_none() {
            return Ok(None);
        }
        let ignore = patterns(q.get("ignore"), false)?;
        let mut quant_groups = vec![];
        let quant_map = q
            .get("config_groups")
            .and_then(Json::as_object)
            .ok_or_else(|| invalid("missing quantization groups"))?;
        if quant_map.len() > MAX_COMPRESSED_TENSORS_GROUPS {
            return Err(invalid(
                "quantization config_groups exceeds group count limit",
            ));
        }
        for (key, group) in quant_map {
            let (bits, group_size, symmetric, code_offset_half, is_embedding_only) =
                ModelConfig::parse_compressed_tensors_group(key, group)?;
            if is_embedding_only {
                continue;
            }
            let w = &group["weights"];
            let valid_bits_and_gs = (bits == COMPRESSED_TENSORS_U2_BITS
                && SUPPORTED_MARLIN_U2_GROUP_SIZES.contains(&group_size))
                || (bits == COMPRESSED_TENSORS_U4_BITS
                    && SUPPORTED_MARLIN_U4_GROUP_SIZES.contains(&group_size))
                || (bits == COMPRESSED_TENSORS_U8_BITS
                    && SUPPORTED_MARLIN_U8_GROUP_SIZES.contains(&group_size));
            if !valid_bits_and_gs
                || !symmetric
                || w.get("dynamic").is_some_and(|v| v != false)
                || (group_size == 0
                    && (w.get("strategy").and_then(Json::as_str) != Some("channel")
                        || w.get("group_size").is_some_and(|v| !v.is_null())))
            {
                return Err(invalid(
                    "HGPTQ requires static symmetric W2/W4/W8 (g32/g64/g128 or channel) weights",
                ));
            }
            quant_groups.push(QuantRule {
                targets: patterns(group.get("targets"), true)?,
                bits,
                group_size,
                symmetric,
                code_offset_half,
            });
        }
        Ok(Some(Self {
            ignore,
            groups: quant_groups,
            transforms,
            embed_quant,
            moe_expert_layer_cache: std::sync::OnceLock::new(),
        }))
    }

    pub(super) fn resolve_moe_layer_expert(
        &self,
        layer_id: usize,
    ) -> Option<(usize, usize, bool, bool)> {
        const MAX_CACHED_MOE_LAYERS: usize = 128;
        if layer_id < MAX_CACHED_MOE_LAYERS {
            let cache = self.moe_expert_layer_cache.get_or_init(|| {
                (0..MAX_CACHED_MOE_LAYERS)
                    .map(|l| self.resolve_moe_layer_expert_uncached(l))
                    .collect()
            });
            return cache[layer_id];
        }
        self.resolve_moe_layer_expert_uncached(layer_id)
    }

    fn resolve_moe_layer_expert_uncached(
        &self,
        layer_id: usize,
    ) -> Option<(usize, usize, bool, bool)> {
        for candidate in [
            format!("model.layers.{layer_id}.mlp.experts.0.gate_proj"),
            format!("model.language_model.layers.{layer_id}.mlp.experts.0.gate_proj"),
            format!("layers.{layer_id}.mlp.experts.0.gate_proj"),
        ] {
            if let Ok((Some(g), _)) = self.resolve(&candidate) {
                return Some((g.bits, g.group_size, g.symmetric, g.code_offset_half));
            }
        }
        None
    }

    fn resolve(&self, name: &str) -> io::Result<(Option<&QuantRule>, Option<InputHadamardSpec>)> {
        if matches(&self.ignore, name)? {
            return Ok((None, None));
        }
        let mut selected: Option<&QuantRule> = None;
        for group in &self.groups {
            if matches(&group.targets, name)? {
                if selected.is_some_and(|old| {
                    (
                        old.bits,
                        old.group_size,
                        old.symmetric,
                        old.code_offset_half,
                    ) != (
                        group.bits,
                        group.group_size,
                        group.symmetric,
                        group.code_offset_half,
                    )
                }) {
                    return Err(invalid(format!(
                        "conflicting quantization rules for {name}"
                    )));
                }
                selected = Some(group);
            }
        }
        let mut input: Option<InputHadamardSpec> = None;
        let mut offline: Option<usize> = None;
        for rule in &self.transforms {
            if matches(&rule.targets, name)? && !matches(&rule.ignore, name)? {
                let Some(sel) = selected else {
                    return Err(invalid(format!(
                        "transform targets unquantized linear {name}"
                    )));
                };
                if !matches!(
                    sel.bits,
                    COMPRESSED_TENSORS_U2_BITS | COMPRESSED_TENSORS_U4_BITS
                ) {
                    return Err(invalid(format!(
                        "transform targets non-W2/W4 linear {name}"
                    )));
                }
                if rule.offline {
                    if offline.is_some() {
                        return Err(invalid(format!("overlapping transform rules for {name}")));
                    }
                    offline = Some(rule.head_dim);
                } else {
                    if input.is_some() {
                        return Err(invalid(format!("overlapping transform rules for {name}")));
                    }
                    input = Some((rule.head_dim, rule.input_signs.clone()));
                }
            }
        }
        if offline != input.as_ref().map(|(hd, _)| *hd) {
            return Err(invalid(format!(
                "inconsistent offline/input transform for {name}"
            )));
        }
        Ok((selected, input))
    }
}

impl QuantScheme {
    /// Resolve against the actual checkpoint tensor base name, without removing prefixes.
    pub fn for_linear(&self, name: &str) -> io::Result<Self> {
        if let Self::CompressedTensors {
            rules: Some(rules),
            ignore,
            ..
        } = self
        {
            let (group, _) = rules.resolve(name)?;
            return Ok(match group {
                Some(g) => Self::CompressedTensors {
                    num_bits: g.bits,
                    group_size: g.group_size,
                    symmetric: g.symmetric,
                    code_offset_half: g.code_offset_half,
                    ignore: ignore.clone(),
                    rules: Some(rules.clone()),
                },
                None => Self::Unquantized,
            });
        }
        Ok(self.clone())
    }

    /// Online input transform only; offline weight_input transforms must never run again.
    pub fn input_hadamard(&self, name: &str) -> io::Result<Option<usize>> {
        Ok(self.input_hadamard_with_signs(name)?.map(|(hd, _)| hd))
    }

    /// Online input transform plus optional per-channel pre-signs (`warp_quant_input_signs`).
    pub fn input_hadamard_with_signs(&self, name: &str) -> io::Result<Option<InputHadamardSpec>> {
        match self {
            Self::CompressedTensors {
                rules: Some(rules), ..
            } => Ok(rules.resolve(name)?.1),
            _ => Ok(None),
        }
    }

    /// Optional quantized `embed_tokens` post-dequant transform (`warp_quant_embed_w2`).
    pub fn embed_quant(&self) -> Option<&EmbedQuantSpec> {
        match self {
            Self::CompressedTensors {
                rules: Some(rules), ..
            } => rules.embed_quant.as_ref(),
            _ => None,
        }
    }

    /// Whether this scheme carries any online/offline transform rules.
    #[inline]
    pub fn has_transforms(&self) -> bool {
        matches!(
            self,
            Self::CompressedTensors {
                rules: Some(rules),
                ..
            } if !rules.transforms.is_empty()
        )
    }
}

#[cfg(test)]
#[path = "config_rules_tests.rs"]
mod tests;
