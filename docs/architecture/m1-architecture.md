# M1 Static Analyzer — Architecture

> v0.4.0 (2026-05-03). 765 tests passing.

## Pipeline

```
adapter.safetensors
        │
        ▼
┌──────────────────────┐
│   parsers/           │  load_adapter()          → raw tensors + metadata
│  safetensors.py      │  _group_lora_layers()    → {layer: {A, B}}
│  metadata.py         │  parse_adapter_metadata()→ AdapterMetadata
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│   features/  (per LoRA layer pair A, B)              │
│                                                      │
│  tensor_stats.py   compute_tensor_stats(A,B,fast)   │
│                    compute_svd_stats(A, fast)        │
│  delta_norm.py     compute_norm_features(A,B,fast)  │
│  distribution.py   compute_distribution_features(   │
│                      A,B,fast)                       │
│  entropy_compression.py                             │
│                    compute_entropy_compression_      │
│                      features(A,B)   ← O(n), M1-ANAL-02  │
│  inter_layer_similarity.py                          │
│                    compute_inter_layer_similarity(  │
│                      pairs, fast)   ← adapter-level, M1-ANAL-03 │
│  layer_stats.py    detect_layer_anomalies()         │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│   detectors/  (per layer + adapter level)            │
│                                                      │
│  entropy.py        compute_entropy() + flags         │
│  outlier.py        zscore + IsolationForest (20t)    │
│  wasserstein.py    W1 distance vs reference distrib  │
│  cross_layer.py    flag concentration anomaly        │
│  init_detector.py  INIT_ONLY / PARTIALLY_TRAINED     │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│   scoring/                                           │
│                                                      │
│  risk_scorer.py    RiskScorer.score_flags()          │
│                    → overall_risk 0–100              │
│  ensemble.py       EnsembleDetector.score_families() │
│                    → ensemble_score 0–100            │
│  score_breakdown.py compute_score_breakdown(report)  │
│                    → ScoreBreakdown (7 sub-scores)   │
│  confidence.py     compute_quality_score(report)     │
│                    compute_confidence_score(report)  │
│                    → ConfidenceScore + quality axes  │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│   reporting/                                         │
│                                                      │
│  per_layer.py      rank_layer_findings(report)       │
│                    → list[PerLayerFinding] top-10    │
│  human_summary.py  render_human_summary(report,      │
│                      verbose, no_color)              │
│                    → fixed-block CLI output          │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│   schemas/        AdapterReport (v1.0.0 stable)      │
│                   Finding, TensorRecord, ScanError   │
│                   NormFeatures, DistributionFeatures │
│                   EntropyCompressionFeatures         │
│                   InterLayerSimilarityFeatures       │
│                   ScoreBreakdown, ConfidenceScore    │
│                   PerLayerFinding                    │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│   reporters/      text → render_human_summary()      │
│                   json → AdapterReport JSON          │
│                   sarif → SARIF 2.1.0                │
└──────────────────────────────────────────────────────┘
```

## Key design decisions

- **bfloat16 support**: Parser detects bfloat16 tensors via header JSON inspection and converts to float32 using the bfloat16 bit-layout identity (`uint16 << 16 → view as float32`). `safetensors.numpy` does not natively support bfloat16. (v0.4.0)
- **Read-only**: M1 never loads a base model, never executes adapter weights.
- **No inference**: All signals are derived from weight tensor statistics only.
- **ΔW = B @ A**: feature families operate on the effective weight update, not on A or B alone. Exception: A/B per-tensor stats are supplementary signals only (M1-ANAL-01).
- **Dual scoring**: Rule-based additive score (stable, interpretable) + ensemble score (accurate). Both exposed in AdapterReport.
- **ScoreBreakdown**: 7 per-family sub-scores (parse, metadata, norm, distribution, entropy, similarity, training_pattern) each with raw_score, normalized_score, weight, top_reasons.
- **Circular-logic guard**: ConfidenceScore and AnalysisQualityScore are derived only from data-quality and coverage signals — never from anomaly features on the risk path.
- **Init suppression**: LoRA zero-init artifacts (B=0, A=random) are classified as INIT_ONLY and suppressed to avoid alert fatigue.
- **Stable schema**: AdapterReport is versioned from day one for M2-M4 integration.

## Feature families (M1-ANAL-01/02/03)

### NormFeatures (delta_norm.py)
Frobenius norm, max/mean abs, delta_norm_ratio of ΔW = B @ A.
- Fast mode: Cholesky path (no ΔW materialization for large tensors)
- Full mode: float32 matmul, in-place abs, einsum fro

### DistributionFeatures (distribution.py)
kurtosis, skewness, mean, std, median, p01, p99, iqr, zero_ratio, entropy of ΔW.
Per-tensor A/B supplementary stats also computed.
- Fast mode: proxy path (lora_A rows), kurtosis/skewness only
- Full mode: float32 matmul, stride-sample BEFORE kurtosis (50K), sort-heavy on 50K

### EntropyCompressionFeatures (entropy_compression.py) — M1-ANAL-02
value_repeat_ratio, unique_value_ratio, approx_compression_ratio (zlib), byte_entropy,
sign_entropy, sign_balance, quantization_suspect_score. O(n), runs in both modes.
Size caps: 64KB zlib, 50K unique sample.

### InterLayerSimilarityFeatures (inter_layer_similarity.py) — M1-ANAL-03
Pairwise cosine + Pearson between ΔW matrices across all layers.
- Fast mode: lora_A rows as proxy, capped to 10K elements
- Full mode: stride-sampled ΔW, capped to 10K elements
- Groups by (A_shape, B_shape), capped at 100 layers per group
- Top-5 suspicious non-adjacent pairs (cosine > 0.85)

## Scoring output (M1-SCORE-01/02/03)

`compute_score_breakdown(report)` — decompose ensemble risk into 7 sub-scores.
Family weights: distribution=0.30, similarity=0.20, parse=metadata=norm=entropy=training_pattern=0.10 each.
ScoringPolicy supports per-family cap/floor and escalation rules with score_bump.

`compute_confidence_score(report, quality)` — ConfidenceScore:
- sample_size_factor, analysis_quality, inter_family_agreement, scan_mode_factor
- verdict_certainty: high ≥0.75, medium ≥0.45, low <0.45
- Never derived from anomaly features (circular-logic guard)

## CLI output (M1-RPT-01/02)

`render_human_summary(report, *, verbose, no_color)` — M1-RPT-02:
- Compact default: VERDICT block + TOP SIGNALS + FINDINGS
- Verbose (--verbose): + SCORE BREAKDOWN + TOP SUSPICIOUS LAYERS + ANALYSIS QUALITY
- ANSI colour: risk level, confidence, recommended action

`rank_layer_findings(report)` — M1-RPT-01:
- Top-10 suspicious layers by severity_score
- RULE_CATALOG stable wording — machine-parseable
- remediation_hint per layer
