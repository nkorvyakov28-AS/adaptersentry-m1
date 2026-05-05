# AdapterSentry — Architecture Overview

> v1.0.2 (2026-05-05). Entry point to the architecture documentation subdirectory.

AdapterSentry is a security scanner for LoRA adapters distributed as `.safetensors`
files. It inspects adapter weight tensors statically — without loading a base model or
executing any inference — to surface structural anomalies consistent with backdoor
injection, safety-alignment suppression, or targeted layer manipulation.

---

## Open-Core Model

AdapterSentry follows an open-core model. M1 is the OSS static analysis engine
(Apache 2.0). Additional commercial capabilities are planned for future releases
and are not part of this package.

---

## Repository Structure

This repository (`adaptersentry-m1`) is the OSS M1 package (Apache 2.0).
Commercial capabilities are developed separately and are not part of this repository.

See [repo-layout.md](repo-layout.md) for the full directory layout of the OSS package.

---

## Pipeline at a Glance

```
adapter.safetensors
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  parsers/                                                   │
│    has_lora_pairs()      key-only pre-check; no tensor load │
│    load_adapter()        safetensors → raw numpy tensors    │
│    _group_lora_layers()  {layer_name: {lora_A, lora_B}}     │
│    parse_adapter_metadata() → AdapterMetadata               │
│    bfloat16 → float32 auto-conversion (header-level detect) │
└──────────────────────────┬──────────────────────────────────┘
                           │ per LoRA pair (A, B)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  features/   (FeatureExtractor.extract_layer)               │
│                                                             │
│  tensor_stats.py         compute_tensor_stats(A,B,fast)     │
│                          compute_svd_stats(A, fast)         │
│  delta_norm.py           compute_norm_features(A,B,fast)    │
│                          ΔW Frobenius norm via Cholesky/matmul│
│  distribution.py         compute_distribution_features(     │
│                            A,B,fast) — kurtosis, skew,      │
│                            median, p01/p99, iqr, zero_ratio  │
│  entropy_compression.py  compute_entropy_compression_       │
│                            features(A,B) — byte_entropy,    │
│                            zlib ratio, sign_entropy, O(n)   │
│  inter_layer_similarity.py  adapter-level pairwise cosine + │
│                            Pearson across all LoRA layers   │
│  layer_stats.py          detect_layer_anomalies()           │
└──────────────────────────┬──────────────────────────────────┘
                           │ FeatureFamilyResult list per layer
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  detectors/                                                 │
│    entropy.py            Shannon entropy flags              │
│    outlier.py            Z-score + IsolationForest (20t)    │
│    wasserstein.py        W1 distance lora_A vs lora_B       │
│    cross_layer.py        concentration anomaly across layers│
│    init_detector.py      INIT_ONLY / PARTIALLY_TRAINED      │
└──────────────────────────┬──────────────────────────────────┘
                           │ typed detector output
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  scoring/                                                   │
│    ensemble.py           EnsembleDetector.score_families()  │
│                          → EnsembleSignal [0–100]           │
│    risk_scorer.py        RiskScorer.score_flags()           │
│                          → overall_risk [0–100]             │
│    score_breakdown.py    compute_score_breakdown(report)    │
│                          → ScoreBreakdown (7 sub-scores)    │
│    confidence.py         compute_quality_score(report)      │
│                          compute_confidence_score(report)   │
│                          → ConfidenceScore + verdict_certainty│
└──────────────────────────┬──────────────────────────────────┘
                           │ AdapterReport
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  reporting/                                                 │
│    per_layer.py          rank_layer_findings(report)        │
│                          → list[PerLayerFinding] top-10     │
│    human_summary.py      render_human_summary(report,       │
│                            verbose, no_color) → str        │
└──────────────────────────┬──────────────────────────────────┘
                           │ AdapterReport + findings
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  schemas/   AdapterReport v1.0.0 (stable public contract)   │
│             ScanResult v1.0.0 (engine output contract)      │
│             ScoreBreakdown, ConfidenceScore, PerLayerFinding│
│             NormFeatures, DistributionFeatures, ...         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  reporters/                                                 │
│    text    → render_human_summary()  (ANSI colour)          │
│    summary-json  → ScanResult v1.0.0 (CI/machine contract)  │
│    debug-json    → ScanResult + tensor_records (local debug)│
│    sarif   → SARIF 2.1.0 (GitHub code scanning)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### Parsers

`src/adaptersentry/parsers/` is the only place that touches raw file bytes.

- `has_lora_pairs()` performs a key-only header scan — no tensor allocation — to
  reject non-LoRA formats before any data is loaded.
- `load_adapter()` reads the file, converts bfloat16 tensors to float32 via the
  bfloat16 bit-layout identity (`uint16 << 16 → view as float32`), and groups
  weight matrices into `{layer_name: {"lora_A": ndarray, "lora_B": ndarray}}`.
- Parser errors are never uncaught exceptions — per-tensor errors surface in
  `TensorRecord.parse_error`; file-level failures produce `ParseStatus.FAILED`.

### Feature Extraction

`src/adaptersentry/features/` computes per-layer statistics. All families operate on
`ΔW = B @ A` — the effective weight update — rather than on `A` or `B` alone.
Per-tensor A/B supplementary stats are supplementary signals only.

`FeatureExtractor.extract_layer()` (in `engine/feature_extractor.py`) coordinates
feature family computation and returns a typed `(TensorRecord, list[FeatureFamilyResult],
list[ScanError])` tuple for each LoRA pair.

See [m1-architecture.md](m1-architecture.md) for per-family implementation details,
sampling strategies, and fast-mode proxy paths.

### Detectors

`src/adaptersentry/detectors/` applies anomaly detection logic on top of the feature
output. Detectors produce typed flags (e.g. `HIGH_KURTOSIS`, `ENERGY_CONCENTRATION`,
`PARTIALLY_TRAINED`) that feed the scoring stage.

Key detectors and their ensemble weights:

| Detector | Weight | Signal |
|----------|--------|--------|
| Kurtosis | 34.0% | Heavy-tailed weights — sparse injection |
| Energy concentration | 26.5% | Dominant singular value — rank-1 trigger |
| Wasserstein distance | 13.5% | A vs B distribution asymmetry |
| Cross-layer consistency | 11.3% | Anomaly concentration in specific layers |
| Shannon entropy | 6.7% | Near-zero (sparse) or near-uniform (noise) |
| Z-score outlier rate | 5.3% | Weight fraction beyond ±3σ |
| IsolationForest | 2.6% | Non-Gaussian structure |

### Scoring

`src/adaptersentry/scoring/` produces the final risk signal.

- `EnsembleDetector.score_families()` — the current typed path; consumes
  `list[FeatureFamilyResult]` and returns a weighted ensemble score [0–100].
- `compute_score_breakdown()` — decomposes the ensemble score into 7 per-family
  sub-scores (parse, metadata, norm, distribution, entropy, similarity,
  training_pattern) each with a raw score, normalized score, weight, and top reasons.
- `compute_confidence_score()` — orthogonal to risk; derived only from analysis
  coverage and data-quality signals (never from anomaly features). Reports
  `verdict_certainty: high / medium / low`. Circular-logic guard enforced by design.

Risk levels map ensemble scores as follows:

| Level | Score range | Meaning |
|-------|-------------|---------|
| LOW | 0–6 | No anomalies detected |
| MEDIUM | 7–13 | Elevated signal; likely benign, warrants review |
| HIGH | 14–35 | Multiple independent detectors agree; manual inspection required |
| CRITICAL | 36–100 | Strong multi-signal evidence; do not load without thorough review |

### Schemas

`src/adaptersentry/schemas/` contains all Pydantic models (`extra="ignore"` on engine
schemas for forward compatibility). The two stable public contracts are:

- `AdapterReport` v1.0.0 — returned by `adaptersentry.scan()` and `--format json`
- `ScanResult` v1.0.0 — returned by the batch engine and `--format summary-json`

`scan_id` in `ScanResult` is deterministic: `sha256(content_hash + ':' + config_hash + ':' + schema_version)`.
Same file with the same configuration always produces the same `scan_id`.

### Reporting

`src/adaptersentry/reporting/` sits between the schema layer and the output formatters.

- `rank_layer_findings()` — ranks the top-10 most suspicious LoRA layers by severity
  score, with triggered families, stable `RULE_CATALOG` wording, and a
  `remediation_hint`.
- `render_human_summary()` — fixed-block CLI output: compact default (VERDICT, TOP
  SIGNALS, FINDINGS) plus an optional verbose block (SCORE BREAKDOWN, TOP SUSPICIOUS
  LAYERS, ANALYSIS QUALITY).

### Reporters

`src/adaptersentry/reporters/` formats the output for each consumer:

| Format | Flag | Contract stability |
|--------|------|--------------------|
| Text (ANSI) | `--format text` (default) | Not a machine contract |
| Summary JSON | `--format summary-json` | Stable — `ScanResult` v1.0.0 |
| Debug JSON | `--format debug-json` | Unstable — local debugging only |
| SARIF 2.1.0 | `--format sarif` | Stable — GitHub code scanning |

---

## Batch Scan Engine

For large-scale scanning, the CLI command `adaptersentry batch` wraps the M1 analyzer
in a production-grade pipeline:

```
adaptersentry batch --input-dir ./adapters --workers 8 --mode fast
        │
        ▼
┌───────────────┐
│  cli/batch.py │  argument parsing, run_id generation
└───────┬───────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  engine/orchestrator.py (or orchestrator_ray.py│
│  when --backend ray)                           │
│    build_manifest() — resolve paths, dedup     │
│    run_batch()      — worker pool management   │
│    resume_after_failure() — crash recovery     │
└───────┬────────────────────────────────────────┘
        │ per-adapter ScanRequest
        ▼
┌────────────────────────────────────────────────┐
│  engine/worker.py   worker_main()              │
│    Phase 1 — Identity:  BLAKE3/SHA256 hash     │
│    Phase 2 — Cache:     CacheStore.lookup()    │
│    Phase 3 — Analysis:  analyzer.scan()        │
│    Phase 4 — Assemble:  ScanResult + DebugReport│
└───────┬────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  engine/result_sink.py   atomic write          │
│  engine/cache.py         content-addressed store│
│  engine/manifest.py      SQLite job state      │
└────────────────────────────────────────────────┘
```

Key engine properties:

- **Content-addressed cache** — `CacheStore` uses `(content_hash, config_hash)` as the
  cache key. A config or mode change automatically invalidates stale entries.
- **Resumable execution** — `ManifestDB` (SQLite WAL) tracks `pending → queued →
  leased → persisted` state. `--resume` resets non-terminal jobs without re-processing
  completed ones.
- **Atomic writes** — `ResultSink` writes to a `.tmp` file, fsyncs, then renames
  (atomic on POSIX). A process kill never produces a partial result.
- **Two backends** — `--backend mp` uses `multiprocessing.Pool(spawn)` (default);
  `--backend ray` uses a persistent Ray actor pool with `max_restarts=3` crash
  isolation and optional multi-node scaling.

See [scan-engine.md](scan-engine.md) for the full engine architecture including the
Ray actor pool, Rust hot-path extensions, and crash recovery details.

---

## Scan Modes

Both `adaptersentry scan` and `adaptersentry batch` accept `--mode full|fast`:

| | `--mode full` | `--mode fast` |
|--|--------------|--------------|
| Use for | Security audits, final verification | Corpus screening, CI pre-filter |
| ΔW norm | float32 B@A materialized | Cholesky path, no ΔW materialization |
| ΔW distribution | B@A + 50K stride-sample | lora_A rows as proxy |
| IsolationForest | Always (20 trees, 2K samples) | Skipped |
| Entropy/compression | O(n), always | O(n), always |
| Inter-layer similarity | ΔW stride-sampled to 10K | lora_A rows to 10K |
| Typical single adapter | ~40s (168-layer adapter) | ~4.5s (168-layer adapter) |

The recommended workflow for large corpora:

```
Large corpus → adaptersentry batch --mode fast
                    │
                    ├── LOW / MEDIUM → allow or manual review
                    └── HIGH / CRITICAL → adaptersentry scan --mode full
```

Fast and full scans produce different `scan_id` values because `scan_mode` is
included in `config_hash`. A fast result never serves as a cache hit for a full
scan request.

See [scan-modes.md](scan-modes.md) for detection equivalence analysis and per-signal
sensitivity differences between modes.

---

## Performance at a Glance (8-CPU VPS, v1.0.0)

| Mode | Backend | Workers | Throughput | Notes |
|------|---------|---------|-----------|-------|
| `fast` | mp | 8 | 203/min | 2.5 min for 500 adapters |
| `fast` | ray | 8 | 211/min | 2.4 min |
| `full` | mp | 4 | 22/min | 22.5 min |
| `full` | ray | 8 | 38/min | 13.3 min |
| `full` | ray + rust | 8 | **69/min** | 7.2 min — Rust hot-path (OPT-04) |

The Rust extension (`adaptersentry-rs/`, built with PyO3 + maturin) accelerates
`isolation_score_1d` (334×), `tensor_stats_f32` (2.4×), `byte_entropy` (4.5×), and
`sign_stats` (2×). All functions have Python fallbacks — the scanner is fully
functional without Rust.

---

---

## Architecture Documents

| Document | Description |
|----------|-------------|
| [m1-architecture.md](m1-architecture.md) | Full parser → features → detectors → scoring → report pipeline with per-family implementation detail |
| [scan-engine.md](scan-engine.md) | Batch scan engine: worker pool, cache, manifest, Ray actor pool, Rust extensions, crash recovery |
| [scan-modes.md](scan-modes.md) | `fast` vs `full`: what changes, detection equivalence, recommended workflow |
| [open-core-boundary.md](open-core-boundary.md) | OSS / commercial boundary; public API contract |
| [repo-layout.md](repo-layout.md) | Directory layout of the installable OSS package |
