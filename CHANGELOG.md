# Changelog

All notable changes to AdapterSentry are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [1.0.1] — 2026-05-04

### Fixed

**BUG-02: feature_completeness always 0% in fast mode**
`compute_quality_score()` checks that all three typed feature schemas
(`norm_features`, `distribution_features`, `entropy_compression_features`) are
non-None to count a layer as "fully analyzed". In fast mode,
`entropy_compression_features` was unconditionally set to `None` in
`FeatureExtractor.extract_layer()` — making `feature_completeness=0%` for any
adapter scanned with `--mode fast`, regardless of parse coverage.

Root cause: `entropy_compression` was gated to full mode only
(`if fast: layer_dict["entropy_compression_features"] = None`). This
contradicts the M1 spec ("O(n) — runs in both fast and full") and became
especially wrong after OPT-04 Rust reduced `byte_entropy` to <0.1 ms/layer.

Fix: removed the fast-mode skip. `entropy_compression` now runs in both modes,
consistent with the M1 spec and the per-layer timing budget.

**BUG-01: rule score 100/100 misleading when ensemble is LOW**
The rule scorer (`RiskScorer.score_flags()`) is additive — each anomaly flag
adds a fixed weight, capped at 100. On adapters with many layers (100+), moderate
flags from multiple layers accumulate and hit the 100 cap even when the ensemble
score (weighted, normalized per adapter) is LOW or MEDIUM.

Example: 126-layer adapter with `cross_layer_consistency=0.000` → multiple
cross-layer flags → rule score saturates at 100 → VERDICT shows
`rule 100/100` next to `ensemble 14.9/100`, which is visually alarming.

Fix: `render_human_summary()` now appends a dim explanatory note in the VERDICT
block when `overall_risk ≥ 75 AND ensemble_score < 25`:
> `↳ Note: rule score (100/100) is inflated by additive flagging across N layers.`
> `Ensemble (14.9/100) is the calibrated risk metric.`

The ensemble score is and remains the authoritative verdict signal.

### Changed
- `engine/feature_extractor.py` — entropy_compression runs in both fast and full modes
- `reporting/human_summary.py` — discrepancy note in VERDICT when rule >> ensemble
- `.gitignore` — publish-safe: sensitive identifiers moved to `.git/info/exclude`

---

## [1.0.0] — 2026-05-04

M1 Static Analyzer — first stable release. All M1 detection families, analytics,
scoring, reporting, batch engine, and performance optimisations are complete and
validated against 498 real HuggingFace LoRA adapters.

### Added

**OPT-04: Rust hot-path extensions (`adaptersentry-rs`)**

`adaptersentry-rs/` Rust crate (PyO3 + maturin) with graceful Python fallback:

- `isolation_score_1d` — exact ECDF-based 1D anomaly score: closed-form solution of
  IsolationForest for 1D data with infinite trees. O(n log n) vs O(n × 20 × 15).
  **334× faster than sklearn IsolationForest.** Score convention: sklearn-compatible.
- `tensor_stats_f32` — single-pass kurtosis, skewness, + one sort for all percentiles.
  **2.4× faster than numpy** (eliminates float64 cast and 4 intermediate allocations).
- `byte_entropy` — single-pass 256-bin histogram. **4.5× faster than numpy bincount.**
- `sign_stats` — single-pass sign_balance + sign_entropy.
- `percentiles_f32` — sort once, compute all quantiles.

Integration: `try: from adaptersentry_rs import ...; except ImportError: use_numpy()`.
Install: `pip install "adaptersentry[rust]"` (adds maturin build step).

**Real-world validation (498 HuggingFace adapters, 8-CPU VPS):**

| Mode | Workers | Backend | Throughput | Wall time | vs baseline |
|------|---------|---------|-----------|-----------|-------------|
| fast | 8 | mp | 203/min | 2.5 min | 168× |
| fast | 8 | ray | 211/min | 2.4 min | 176× |
| full | 4 | mp | 22/min | 22.5 min | 18× |
| full | 8 | ray | 38/min | 13.3 min | 31× |
| full | 8 | ray + rust | **69/min** | **7.2 min** | **57×** |

AlgoCore single adapter (168 layers, full mode): **5.9s** (was 40s, −85%).

### Changed
- `detectors/outlier.py` — `isolation_forest_score()` uses Rust when available; sklearn
  IsolationForest moved to fallback path; interface (mean_score, anomalous_fraction) unchanged
- `features/tensor_stats.py` — `compute_tensor_stats()` uses `tensor_stats_f32` (Rust)
  when available; float32 internally instead of float64 (precision sufficient for thresholds)
- `features/entropy_compression.py` — `byte_entropy` uses Rust single-pass histogram
- All three fallback to numpy/sklearn transparently if `adaptersentry_rs` is not installed

### Docs
- `docs/architecture/scan-engine.md` — OPT-04 section, final performance table
- `docs/internal/performance-analysis.md` — OPT-04 results, full optimisation history

---

## [0.4.1] — 2026-05-04

### Added

**Ray actor pool backend (OPT-03)**
- `engine/orchestrator_ray.py` — `run_batch_ray()` drop-in for `run_batch()`:
  - `ScanWorkerActor` (`@ray.remote`, `max_restarts=3`) — persistent stateful actor; heavy modules pre-imported once in `__init__` (equivalent to OPT-01 `_pool_initializer`)
  - Manual `ray.wait()` loop with `future → actor` tracking — OOM-killed actors reported as failed and replaced without stalling the batch
  - `ray_address` parameter for remote clusters (multi-node horizontal scaling)
- `adaptersentry batch --backend ray|mp` — selects backend (default: `mp`)
- `adaptersentry batch --ray-address ADDRESS` — Ray cluster address (local if omitted)
- `pyproject.toml` optional dep: `pip install adaptersentry[ray]` (`ray[default]>=2.9.0`)
- 8 new Ray-specific tests (`tests/engine/test_orchestrator_ray.py`)

**Real-world benchmark results (Ray, 498 HuggingFace adapters):**

| Mode | Workers | Backend | Throughput | Wall time | p95 | Peak/worker |
|------|---------|---------|-----------|-----------|-----|-------------|
| fast | 8 | mp (v0.4.0) | 203.2/min | 2.5 min | 6.1s | 455 MB |
| fast | 8 | **ray** | **211.1/min** | **2.4 min** | **4.4s** | 721 MB |
| full | 4 | mp (v0.4.0) | 22.1/min | 22.5 min | 23.0s | 473 MB |
| full | 8 | **ray** | **37.6/min** | **13.3 min** | **24.0s** | 710 MB |

Ray full mode on 8 workers: +70% throughput vs mp on 4 workers; no OOM kills.

### Fixed

**BLAS over-subscription (permanent fix)**
`_pool_initializer()` in `engine/orchestrator.py` now sets `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` via `os.environ.setdefault()` before importing numpy. Previously, each of N workers spawned its own OMP thread pool (load avg 47 on 8 cores without the env vars). The Ray `ScanWorkerActor.__init__` applies the same fix. No caller-side environment variables required.

**scan_id RESULT CONFLICT false positive**
`ResultSink.write()` compared the full `ScanResult` JSON for idempotency, including `started_at`, `completed_at`, and `wall_time_ms` — fields that change on every scan of the same adapter (wall-clock time varies). This caused false RESULT CONFLICT warnings on `--resume` and `--force-rescan` even when analysis output was identical. Fix: `_stable_content_hash()` excludes timing-volatile fields from comparison; true analysis conflicts (different verdict/findings/errors) still log RESULT CONFLICT.

### Changed
- `benchmarks/harness.py` — `_run_scenario()` gains `backend: str = "mp"` parameter
- `benchmarks/run_real.py` — `--backend mp|ray` flag

### Docs
- `docs/architecture/scan-engine.md` — Ray backend section, updated performance table, OPT-03 marked done

---

## [0.4.0] — 2026-05-03

### Added

**M1 Analytics Expansion**
- `DistributionFeatures` extended (M1-ANAL-01): `delta_median`, `delta_p01`, `delta_p99`, `delta_iqr`, `delta_zero_ratio`, `delta_entropy` on ΔW = B @ A; per-tensor `stats_A` / `stats_B` supplementary fields in `TensorRecord`
- `EntropyCompressionFeatures` family (M1-ANAL-02): `value_repeat_ratio`, `unique_value_ratio`, `approx_compression_ratio` (zlib), `byte_entropy`, `sign_entropy`, `sign_balance`, `quantization_suspect_score` — O(n), runs in both fast and full mode
- `InterLayerSimilarityFeatures` family (M1-ANAL-03): pairwise cosine + Pearson between ΔW matrices across all adapter layers, grouped by module type; top-5 suspicious non-adjacent pairs (cosine > 0.85); fast mode uses lora_A proxy, full mode uses stride-sampled ΔW
- `ScoreBreakdown` schema + `scoring/score_breakdown.py` (M1-SCORE-01): 7 per-family sub-scores (parse, metadata, norm, distribution, entropy, similarity, training_pattern) each with `raw_score`, `normalized_score`, `weight`, `top_reasons`, cap/floor flags
- `ScoringPolicy` schema (M1-SCORE-02): versioned weights config, per-family cap/floor/escalation rules with `score_bump`, reason tracing — configurable without code changes
- `ConfidenceScore` + `AnalysisQualityScore` (M1-SCORE-03): orthogonal to risk score; derived only from data-quality/coverage signals; circular-logic guard enforced; `verdict_certainty: high / medium / low`
- `PerLayerFinding` schema + `reporting/per_layer.py` (M1-RPT-01): top-10 suspicious layers ranked by `severity_score`, triggered families, stable `RULE_CATALOG` wording, `remediation_hint`
- `render_human_summary()` (M1-RPT-02): fixed-block CLI output; compact default + `--verbose` mode (score breakdown, top-layer findings, analysis quality block); ANSI colour; `--no-color` flag
- `adaptersentry scan --verbose` — enables verbose text output (M1-RPT-02 full blocks)
- `adaptersentry scan --no-color` — disables ANSI colour output

**Parser**
- bfloat16 adapter support: parser reads safetensors header JSON to detect `BF16` tensors before calling `get_tensor()`, then converts raw bytes to float32 via the bfloat16 bit-layout identity (`uint16 << 16 → view as float32`). Previously 48/498 HuggingFace adapters (9.6%) failed with `INVALID_SAFETENSORS`; error rate now ~0%

### Fixed

**OOM in full-mode batch (RCA-08)**
`inter_layer_similarity._make_vector()` returned a stride VIEW (`delta.flatten()[::k][:N]`) that kept the full 83.9 MB float64 ΔW buffer alive for the entire batch. For a 216-layer adapter with 72 layers at ΔW = 10.5M elements: 72 × 83.9 MB = 6 GB accumulated in `shape_groups`. Fix: `.copy()` on all stride returns in `_make_vector`. Peak RSS 7680 MB → **524 MB** on the largest real corpus adapters (-93%).

**Secondary RSS accumulation (RCA-09)**
`distribution.py` and `delta_norm.py` materialised ΔW for layers with `out × in ≤ 16M`. Lowered `_MAX_DELTA_NUMEL_FULL` 16M → 4M in both files: layers above the threshold now use lora_A proxy (distribution) or Cholesky path (delta_norm), reducing per-layer peak from ~41 MB to ~2 MB for typical LLM attention layers.

### Performance

Per-layer bottlenecks eliminated (168-layer real adapter):

| Step | Before | After |
|------|--------|-------|
| FULL end-to-end | ∞ (hung) | **40s** |
| FAST end-to-end | 227s | **4.5s** |
| `distrib_full` / layer | 217ms | 73ms |
| `norm_full` / layer | 69ms | 29ms |
| `outlier_IF` / layer | 118ms | 40ms |
| `wasserstein` ×168 | 5.2s | 0.2s |
| `inter_layer_sim` | 535s | <1s |

Root causes eliminated: kurtosis computed before stride-sample (RCA-01); float64 matmuls (RCA-02/04); rng.choice at high sampling rates (RCA-03/07); IsolationForest 100→20 trees (RCA-05); inter-layer vectors 4.3M→10K elements (RCA-06).

**Real corpus benchmarks (498 adapters, HuggingFace Hub):**

| Mode | Workers | Backend | Throughput | Wall time |
|------|---------|---------|-----------|-----------|
| fast | 8 | mp | 203.2/min | 2.5 min |
| full | 4 | mp | 22.1/min | 22.5 min |

### Changed
- `adaptersentry scan --format text` now uses `render_human_summary()` (M1-RPT-02); replaces old text renderer
- `EnsembleDetector.score_families(list[FeatureFamilyResult])` — new typed path; `score(dict)` kept as legacy
- `_MAX_DELTA_NUMEL_FULL = 4_000_000` in `distribution.py` and `delta_norm.py` (was 16M)
- `inter_layer_similarity._make_vector()` returns `.copy()` of stride slices (memory safety)
- 9 new tests for OOM memory-guard behavior; 5 new tests for bfloat16 parser support

### Docs
- `docs/architecture/m1-architecture.md` — updated with M1-ANAL/SCORE/RPT families
- `docs/architecture/scan-modes.md` — updated ΔW materialization threshold (16M → 4M)
- `docs/cli/usage.md` — `--verbose`, `--no-color`, score breakdown table
- `docs/internal/performance-analysis.md` — RCA-01 through RCA-09 full analysis
- `docs/internal/scan-engine-architecture.md` — Memory Guards section (view retention hazard)

---

## [0.3.0] — 2026-05-01

### Added

**Batch scan engine**
- `adaptersentry batch` CLI — scan directories of adapters with a resumable worker pool, content-addressed cache, atomic result writes
- `engine/identity.py` — BLAKE3/SHA256 content hash; deterministic `scan_id`
- `engine/manifest.py` — SQLite WAL manifest; `resume_after_failure()`
- `engine/cache.py` — content-addressed object store; BLAKE3 integrity on read
- `engine/orchestrator.py` — `multiprocessing.Pool(spawn)`; imap_unordered; heartbeat/lease
- `engine/worker.py` — per-adapter scan pipeline; poison-job isolation
- `engine/result_sink.py` — atomic rename+fsync; JSONL append; idempotency
- `engine/feature_extractor.py` — typed `FeatureFamilyResult` per layer; `families_from_record()` migration bridge
- `EnsembleDetector.score_families()` — typed path, no raw dict string key access

**Scan modes**
- `--mode fast|full` on both `scan` and `batch` commands
- `fast` — truncated SVD (top-50, randomised) for matrices ≥ 512×512; 50K-element deterministic sample for tensors > 100K; IsolationForest skipped for tensors > 5M elements; ~5× faster on 7B-model adapters
- `full` (default) — all detectors at full depth; for security audits

**Output contracts**
- `--format summary-json` → `ScanResult` v1.0.0 (stable; embeds `ScanIdentity` + `AdapterArtifactIdentity`)
- `--format debug-json` → `DebugReport` (adds `tensor_records`, `feature_family_results`; not stable)
- Engine schemas: `ScanResult`, `DebugReport`, `ScanIdentity`, `AdapterArtifactIdentity`, `EnsembleSignal`, `RiskVerdict`, `FeatureSignal`, `FeatureFamilyResult`, `CombinedReport` (M2 bridge placeholder)

**Error taxonomy**
- `ScanPhase` enum (parse / metadata / feature / scoring / reporting) on `ScanError`
- `ErrorSeverity` enum (fatal / degraded / warning) — auto-inferred from error code

**Migration tests**
- `tests/fixtures/scan_result_v1.0.0.json` — committed baseline fixture
- `tests/schemas/test_migration.py` — round-trip + forward-compat tests
- `scripts/snapshot_schema.py` — generate fixtures for future version bumps

**Benchmark harness**
- `benchmarks/corpus.py` — deterministic synthetic LoRA corpus generator
- `benchmarks/harness.py` — cold/warm scenarios, p50/p95/p99, peak RSS tracking
- `benchmarks/check_regression.py` — CI regression gate (p95 > 2× baseline → fail)
- `benchmarks/run.sh` — single entrypoint for CI and local use
- `benchmarks/results/baseline.json` — committed reference baseline

**Docs**
- `docs/architecture/scan-engine.md` — batch engine architecture
- `docs/architecture/scan-modes.md` — fast vs full mode reference
- `docs/output-schema/scan-result.md` — ScanResult v1.0.0 full field reference
- Updated `docs/architecture/m1-architecture.md`, `docs/cli/usage.md`, `docs/output-schema/error-taxonomy.md`

### Fixed
- **Kurtosis / skewness non-determinism** — `RuntimeWarning: Precision loss (catastrophic cancellation)` on near-constant tensors (zero-init B matrices) now returns `0.0` instead of `nan`; eliminates `RESULT CONFLICT` warnings in batch runs
- **Non-LoRA format handling** — `has_lora_pairs()` reads key names only before loading tensor data; Whisper, IA³, prefix-tuning adapters rejected immediately without loading their weights

### Changed
- `compute_tensor_stats(tensor, *, fast=False)` — new `fast` keyword argument (backward-compatible)
- `compute_svd_stats(tensor, *, fast=False)` — new `fast` keyword; randomised SVD in fast mode
- `detect_outlier_anomalies(..., *, fast=False)` — new `fast` keyword with size guard
- `adaptersentry scan --format json` is now a legacy alias for `--format summary-json`
- `Development Status` classifier updated to `4 - Beta`
- `psutil>=5.9.0` added to dependencies

### Security
- Non-LoRA adapters rejected at parser level before tensor data is loaded
- IsolationForest size guard prevents memory exhaustion on large tensors in fast mode

---

## [0.2.0] — 2026-04-29

### Added
- `src/adaptersentry/` package with src-layout
- CLI: `adaptersentry scan ADAPTER [--format text|json|sarif] [--output FILE] [--fail-on SEVERITY]`
- Pydantic schemas: `AdapterReport` v1.0.0, `TensorRecord`, `AdapterMetadata`, `Finding`, `ScanError`
- Error taxonomy: MALFORMED / UNSUPPORTED / DEGRADED
- Reporters: text, json, sarif (SARIF 2.1.0 with GitHub `security-severity`)
- New tests: schemas, reporters, CLI, backward-compatibility (298 total, +79 from v0.1.0)
- Docs: architecture, output schema, CLI usage, benchmark methodology

### Changed
- Package renamed: `m1_static` → `adaptersentry` (new canonical import)
- CLI entry point standardized to `adaptersentry`
- Legacy `analyze()` dict format frozen as backward-compatibility contract

### Deprecated
- `adaptersentry-m1` CLI alias (kept for backward compatibility, scheduled for removal in a future major version)

### Migration

```python
# Old (v0.1.0)
from m1_static.analyzer import analyze
result = analyze("adapter.safetensors")

# New (v0.2.0)
from adaptersentry import scan
report = scan("adapter.safetensors")  # AdapterReport
```

---

## [0.1.0] — 2026-04-28

### Added

**M1 Static Analyzer (initial public release)**

- `adaptersentry scan` CLI with `--format text|json|sarif`, `--fail-on`, `--output`, `--rank`
- `adaptersentry.analyzer.scan()` — new typed API returning `AdapterReport`
- `adaptersentry.analyzer.analyze()` — legacy dict API (stable, backward-compatible)
- `python -m adaptersentry` entry point
- Seven-detector weighted ensemble (kurtosis, energy concentration, Wasserstein distance,
  cross-layer consistency, Shannon entropy, Z-score outlier rate, IsolationForest)
- Rule-based additive flag scoring with `RiskScorer`
- Init-only adapter detection with systematic false-positive suppression
- PARTIALLY_TRAINED adapter classification
- Pydantic schemas: `AdapterReport`, `Finding`, `TensorRecord`, `AdapterMetadata`,
  `ScanError` with error taxonomy (malformed / unsupported / degraded)
- SARIF 2.1.0 reporter with `properties.security-severity` for GitHub code scanning
- JSON reporter via `AdapterReport.model_dump_json()`
- Text reporter with ANSI colour, severity-coded findings, degraded-state warnings
- GitHub Actions integration helper (`src/adaptersentry/integrations/github_action.py`)
- HuggingFace Hub benchmark pipeline (`benchmarks/`)
- 500-adapter HuggingFace Hub benchmark: LOW 64.2% · MEDIUM 29.3% · HIGH 5.3% · CRITICAL 1.1%
- `docs/` covering architecture, output schema, CLI usage, and benchmark methodology
- `src/` layout: `adaptersentry` package replaces `m1_static`

### Security

- M1 operates in parse-only mode on `.safetensors` files; no model code or pickle executed
- Tensor bomb protection: rejects tensors exceeding 1B elements
- Path traversal mitigation via `pathlib.Path.resolve()` throughout
- Metadata depth check (> 5 levels) flagged as anomaly signal
- All custom weights validated on construction to prevent score manipulation

[1.0.1]: https://github.com/nkorvyakov28-AS/adaptersentry/releases/tag/v1.0.1
[1.0.0]: https://github.com/nkorvyakov28-AS/adaptersentry/releases/tag/v1.0.0
[0.4.1]: https://github.com/nkorvyakov28-AS/adaptersentry/releases/tag/v0.4.1
[0.4.0]: https://github.com/nkorvyakov28-AS/adaptersentry/releases/tag/v0.4.0
[0.3.0]: https://github.com/nkorvyakov28-AS/adaptersentry/releases/tag/v0.3.0
[0.2.0]: https://github.com/nkorvyakov28-AS/adaptersentry/releases/tag/v0.2.0
[0.1.0]: https://github.com/nkorvyakov28-AS/adaptersentry/releases/tag/v0.1.0
