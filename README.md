# AdapterSentry

![Tests](https://img.shields.io/badge/tests-773%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.2-blue)

AdapterSentry is a static security scanner for LoRA adapters distributed as `.safetensors`
files. Anyone can publish an adapter to HuggingFace Hub; a malicious adapter can inject
backdoors, suppress safety alignment, or redirect model behaviour — all without touching the
base model weights. AdapterSentry inspects the adapter weight tensors directly, before the
adapter is loaded into any model.

**v1.0.2** promotes `BehavioralResult` / `ProbeResult` to the v1.0.0 public wire contract
and adds `scan_to_result(adapter_path)` as a stable public API returning `ScanResult`
directly (required by downstream sandbox runners). Also adds `ScanPhase.BEHAVIORAL`.
**v1.0.1** fixes two bugs: `feature_completeness` always 0% in fast mode, and a
misleading `rule 100/100` display when ensemble is LOW.
**v1.0.0** — M1 Static Analyzer complete: 69 adapters/min (Ray + Rust), 57× faster than baseline.
See [docs/architecture/open-core-boundary.md](docs/architecture/open-core-boundary.md).

---

## Why this matters

LoRA adapters are tiny files — typically 10–200 MB — that modify a base model's behaviour
by adding a low-rank weight delta at every targeted layer. The supply-chain attack surface is
real: a user who downloads an adapter from Hub applies that delta to their model automatically,
with no code review and often no sandboxing. Structural anomalies in the weight tensors —
abnormal kurtosis, near-rank-1 energy concentration, selective layer targeting — are detectable
without running the model. M1 surfaces these signals and lets you make an informed decision
before loading.

---

## Quick Start

### Install

```bash
pip install adaptersentry

# With Ray backend (optional, recommended for large corpora)
pip install "adaptersentry[ray]"

# With Rust hot-path extensions (optional, requires Rust toolchain — 57× full-mode throughput)
pip install maturin
cd adaptersentry-rs && VIRTUAL_ENV=$(python -c "import sys; print(sys.prefix)") maturin develop --release

# Development install from source
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
pip install -e ".[dev]"
```

### Scan a single adapter

```bash
# Default: text output with verdict + top signals
adaptersentry scan adapter.safetensors

# Full breakdown: score decomposition, per-layer findings, analysis quality
adaptersentry scan adapter.safetensors --verbose

# Stable JSON for CI gate
adaptersentry scan adapter.safetensors --format summary-json --output report.json

# Fast screening mode (~9× faster, equivalent detection)
adaptersentry scan adapter.safetensors --mode fast

# SARIF for GitHub code scanning
adaptersentry scan adapter.safetensors --format sarif --output results.sarif

# Fail CI on HIGH or CRITICAL findings
adaptersentry scan adapter.safetensors --fail-on HIGH

# Per-layer debug detail
adaptersentry scan adapter.safetensors --format debug-json
```

### Scan a directory (batch)

```bash
# Fast screening — multiprocessing (default)
adaptersentry batch --input-dir ./adapters --mode fast --workers 8

# Fast screening — Ray backend (better crash isolation, same interface)
adaptersentry batch --input-dir ./adapters --mode fast --workers 8 --backend ray

# Full audit — Ray, 8 workers (vs 4 max with mp before OOM fix)
adaptersentry batch --input-dir ./flagged --mode full --workers 8 --backend ray

# Resume after crash
adaptersentry batch --input-dir ./adapters --run-id my-run --resume
```

### Python API

```python
from pathlib import Path
from adaptersentry import scan, scan_to_result
from adaptersentry.scoring.score_breakdown import compute_score_breakdown
from adaptersentry.scoring.confidence import compute_confidence_score, compute_quality_score

# Full analysis (default) — returns AdapterReport
report = scan(Path("adapter.safetensors"))
print(report.risk_summary.risk_level)          # LOW / MEDIUM / HIGH / CRITICAL

# Engine-level ScanResult — returns ScanResult with .identity, .verdict, .artifact
result = scan_to_result(Path("adapter.safetensors"))
print(result.verdict.overall_level)            # LOW / MEDIUM / HIGH / CRITICAL

# Score breakdown across 7 feature families
breakdown = compute_score_breakdown(report)
for sub in breakdown.sub_scores:
    print(f"{sub.family}: {sub.normalized_score:.2f}  {sub.top_reasons}")

# Confidence in the result
quality = compute_quality_score(report)
conf = compute_confidence_score(report, quality)
print(conf.verdict_certainty)                  # high / medium / low

# Fast mode for throughput screening
report = scan(Path("adapter.safetensors"), fast=True)
```

---

## What's New in v1.0.2

### BehavioralResult / ProbeResult v1.0.0 wire contract

`BehavioralResult` is promoted from a 5-field placeholder to the full public v1.0.0 schema.
New / promoted fields: `behavioral_verdict`, `trigger_confirmed`, `behavioral_score`,
`semantic_drift_score`, `base_model_used`, `base_model_sha`, `probe_set_version`,
`n_probes_run`, `n_probes_confirmed`, `n_probes_skipped`, `skip_reason` (enum),
`probe_results: list[ProbeResult]`, `targeted_layers`. Legacy `sandbox_verdict` and
`raw` fields are kept for backwards compatibility.

`ProbeResult` fields: `probe_id`, `probe_set_version`, `trigger_type`,
`verdict` (confirmed / cleared / inconclusive / skipped / error), `trigger_confirmed`,
`semantic_drift`, `kl_drift`, `string_match`, `refusal_bypass`, `severity_weight`,
`base_output_hash`, `patched_output_hash`, `elapsed_ms`, `error`.

Both schemas use `extra='ignore'` and `frozen=True` for forward compatibility. This is
the public wire contract; downstream M2 implementations populate it.

### `scan_to_result()` public API

New top-level `scan_to_result(adapter_path)` returns the engine-level `ScanResult`
(with `.identity`, `.verdict`, `.artifact`) directly, bypassing the `AdapterReport`
intermediary. Required by `CombinedReport` and downstream sandbox runners; previously
only accessible via the private `cli._build_scan_result`.

### `ScanPhase.BEHAVIORAL`

New `ScanPhase.BEHAVIORAL` enum member marks the M2 pipeline phase in `ScanError`
records, completing the phase taxonomy: `parse / metadata / feature / scoring /
reporting / behavioral`.

---

## Output Formats

### `--format text` (default)

Human-readable terminal output with risk level, ensemble score, confidence, and findings.
ANSI colour enabled by default (`--no-color` to disable). Add `--verbose` for full score
breakdown, per-layer findings, and analysis quality block.

### `--format summary-json`

Emits a versioned `ScanResult` JSON document (`schema_version: "1.0.0"`) — the stable
public contract for CI gates and machine consumers. Embeds `ScanIdentity` (deterministic
`scan_id`) and `AdapterArtifactIdentity` (content hash).
See [docs/output-schema/scan-result.md](docs/output-schema/scan-result.md).

### `--format debug-json`

Extends `ScanResult` with per-layer `tensor_records` and `feature_family_results`.
Not a stable contract — for local debugging only.

### `--format sarif`

Emits [SARIF 2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/) for direct ingestion by
GitHub code scanning. Findings include `properties.security-severity` (0–10 CVSS-like scale).

```yaml
# .github/workflows/adapter-scan.yml
- name: Scan LoRA adapter
  run: adaptersentry scan adapter.safetensors --format sarif --output results.sarif

- name: Upload to GitHub code scanning
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
  if: always()
```

See [docs/cli/usage.md](docs/cli/usage.md) for full flag reference and exit codes.

---

## Scan Modes

| Mode | SVD | Stats | IsolationForest | Use for |
|------|-----|-------|-----------------|---------|
| `--mode full` (default) | Full spectrum | Full tensor | Always | Security audits, final verification |
| `--mode fast` | Top-50, randomised | 50K-element sample | Skipped >5M elements | Corpus screening, CI pre-filter |

Fast mode preserves detection quality for typical backdoor patterns.
See [docs/architecture/scan-modes.md](docs/architecture/scan-modes.md) for details.

---

## How It Works

AdapterSentry inspects `.safetensors` files in read-only mode without executing any model code.

### M1 pipeline

```
adapter.safetensors
        │
  parsers/          has_lora_pairs() pre-check → load_adapter → _group_lora_layers
                    bfloat16 tensors auto-converted to float32 (v0.4.0)
        │
  engine/           FeatureExtractor.extract_layer() per LoRA pair
  features/         spectral · norm · distribution · entropy · outlier
                    entropy_compression · inter_layer_similarity
        │
  detectors/        wasserstein · cross_layer · init_detector
        │
  scoring/          EnsembleDetector.score_families() → EnsembleSignal [0–100]
                    compute_score_breakdown() → ScoreBreakdown (7 families)
                    compute_confidence_score() → ConfidenceScore
                    RiskVerdict: allow / review / block
        │
  reporting/        rank_layer_findings() → list[PerLayerFinding] top-10
                    render_human_summary() → fixed-block CLI output
        │
  schemas/          ScanResult v1.0.0  →  reporters/text · summary-json · debug-json · sarif
```

See [docs/architecture/m1-architecture.md](docs/architecture/m1-architecture.md) for detail.

---

## M1 Detection Methods

### Detectors

| Detector | Ensemble weight | Signal |
|----------|-----------------|--------|
| **Kurtosis** | 0.340 | Excess kurtosis > 10× — heavy-tailed weights consistent with sparse injection |
| **Energy concentration** | 0.265 | `σ₁² / Σσᵢ² > 0.95` (SVD) — single dominant direction; consistent with rank-1 trigger |
| **Wasserstein distance** | 0.135 | W1 distance between lora_A and lora_B distributions — large asymmetry signals different populations |
| **Cross-layer consistency** | 0.113 | Low score = anomaly concentration in specific layers; targeted modification pattern |
| **Shannon entropy** | 0.067 | Near-zero (sparse) or near-unity (uniform noise) both flagged |
| **Z-score outlier rate** | 0.053 | Fraction of weights beyond ±3σ; Gaussian adapters have < 0.3% |
| **Isolation Forest** | 0.026 | Unsupervised anomaly score; catches non-Gaussian structure Z-score misses |

### Extended feature families (v0.4.0)

| Family | Signals |
|--------|---------|
| **DistributionFeatures** | kurtosis, skewness, mean, std, median, p01, p99, iqr, zero_ratio, delta_entropy; per-tensor A/B stats |
| **EntropyCompressionFeatures** | value_repeat_ratio, unique_value_ratio, compression_ratio (zlib), byte_entropy, sign_entropy, sign_balance, quantization_suspect_score |
| **InterLayerSimilarityFeatures** | pairwise cosine + Pearson; top-5 suspicious non-adjacent pairs (cosine > 0.85); per-module-type mean similarity |

### Score breakdown (7 families)

| Family | Weight | Primary signals |
|--------|--------|-----------------|
| `distribution` | 30% | kurtosis, skewness, percentiles, zero_ratio, delta_entropy |
| `similarity` | 20% | inter-layer cosine/Pearson, suspicious pairs |
| `parse` | 10% | parse_status, tensor errors |
| `metadata` | 10% | base_model, peft_type, target_modules, rank |
| `norm` | 10% | fro_norm_delta, delta_norm_ratio |
| `entropy` | 10% | value_repeat_ratio, byte_entropy, quantization_suspect_score |
| `training_pattern` | 10% | cross_layer_consistency, wasserstein, init_status |

### Init-only adapter detection

Standard PEFT LoRA initialisation sets `B = 0` and draws `A` from a uniform distribution.
M1 identifies this pattern when `std_B < 1e-6` and `entropy_A > 0.98` hold across all layers,
reports `training_status: INIT_ONLY`, and suppresses init-artifact flags.

`training_status: PARTIALLY_TRAINED` flags adapters where some layers are trained and others
remain at init — consistent with targeted-layer injection.

### Risk levels

| Level | Ensemble score | Meaning |
|-------|---------------|---------|
| LOW | 0–6 | No anomalies detected. |
| MEDIUM | 7–13 | Elevated signal; likely benign but warrants review. |
| HIGH | 14–35 | Multiple independent detectors agree. Manual inspection required. |
| CRITICAL | 36–100 | Strong multi-signal evidence. Do not load without thorough review. |

---

## Benchmark Results

### Real-World Hub Corpus (500 adapters, v0.4.0)

AdapterSentry M1 was run against 500 public LoRA adapters from HuggingFace Hub
(filter: `peft`, sorted by download count). Only `adapter_model.safetensors` downloaded;
no base model weights fetched. **This is an observational static scan, not a malware classifier.**

| Risk level | Count | Share |
|------------|-------|-------|
| LOW | 289 | 64.2% |
| MEDIUM | 132 | 29.3% |
| HIGH | 24 | 5.3% |
| CRITICAL | 5 | 1.1% |

Ensemble score p50 ≈ 4.35 · p90 ≈ 11.71 · p99 ≈ 36.0.

High-scoring adapters are **investigation candidates**, not confirmed malicious content.
A high ensemble score is the beginning of an investigation, not a conclusion.

### Throughput (v1.0.0, 8-CPU VPS)

| Mode | Backend | Workers | Throughput | Wall time (500) | vs baseline |
|------|---------|---------|-----------|-----------------|-------------|
| `fast` | mp | 8 | 203/min | 2.5 min | 168× |
| `fast` | ray | 8 | 211/min | 2.4 min | 176× |
| `full` | mp | 4 | 22/min | 22.5 min | 18× |
| `full` | ray | 8 | 38/min | 13.3 min | 31× |
| `full` | **ray + rust** | 8 | **69/min** | **7.2 min** | **57×** |

Baseline: v0.2.x sequential on 2-CPU VPS — 1.2 adapters/min, 195 min for 500 adapters.

AlgoCore single-adapter (168 layers, full mode): **5.9s** (was 40s pre-optimisation, −85%).

Benchmark methodology: [docs/benchmarks/methodology.md](docs/benchmarks/methodology.md).

### Small Benchmark

| Adapter | Training status | Ensemble | Risk |
|---------|----------------|----------|------|
| llamafactory/tiny-random-Llama-3-lora | TRAINED | 4.1 | LOW |
| peft-internal-testing/tiny_T5ForSeq2SeqLM-lora | TRAINED | 3.9 | LOW |
| ybelkada/opt-350m-lora | INIT_ONLY | 2.5 | LOW |
| artek0chumak/bloom-560m-safe-peft | INIT_ONLY | 8.0 | MEDIUM |
| **qylu4156/strongreject-15k-v1** | TRAINED | **14.6** | ⚠️ **HIGH** |

---

## Output Schema

<details>
<summary>ScanResult schema (summary-json — stable, schema_version 1.0.0)</summary>

```json
{
  "schema_version": "1.0.0",
  "identity": {
    "scan_id": "sha256:...",
    "analyzer_version": "1.0.2",
    "schema_version": "1.0.0"
  },
  "artifact": {
    "content_hash": "sha256:...",
    "file_size_bytes": 32768
  },
  "verdict": {
    "overall_score": 0,
    "overall_level": "LOW",
    "recommended_action": "allow",
    "m2_recommended": false,
    "training_status": "TRAINED"
  },
  "ensemble": {"score": 4.1, "risk_level": "LOW"},
  "findings": [],
  "errors": [],
  "status": "ok",
  "parse_status": "ok",
  "n_layers": 2,
  "n_layers_analyzed": 2
}
```

Full schema reference: [docs/output-schema/scan-result.md](docs/output-schema/scan-result.md)

</details>

<details>
<summary>Legacy AdapterReport schema (scan() / --format json)</summary>

```json
{
  "schema_version": "1.0.0",
  "tool": {"name": "adaptersentry", "version": "1.0.2"},
  "risk_summary": {
    "overall_risk": 0, "risk_level": "LOW",
    "ensemble_score": 4.1, "ensemble_risk_level": "LOW",
    "training_status": "TRAINED", "n_layers": 2
  },
  "findings": [],
  "errors": [],
  "analysis_mode": "full"
}
```

Full schema reference: [docs/output-schema/adapter-report.md](docs/output-schema/adapter-report.md)

</details>

---

## Architecture and Docs

| Document | Description |
|----------|-------------|
| [docs/architecture/m1-architecture.md](docs/architecture/m1-architecture.md) | Full parser → features → detectors → scoring → report pipeline |
| [docs/architecture/scan-engine.md](docs/architecture/scan-engine.md) | Batch scan engine: worker pool, cache, manifest, crash recovery |
| [docs/architecture/scan-modes.md](docs/architecture/scan-modes.md) | fast vs full mode: what changes, detection equivalence |
| [docs/architecture/open-core-boundary.md](docs/architecture/open-core-boundary.md) | What is OSS, integration contract |
| [docs/architecture/repo-layout.md](docs/architecture/repo-layout.md) | Repository structure |
| [docs/output-schema/scan-result.md](docs/output-schema/scan-result.md) | ScanResult v1.0.0 field reference |
| [docs/output-schema/adapter-report.md](docs/output-schema/adapter-report.md) | AdapterReport v1.0.0 field reference |
| [docs/output-schema/error-taxonomy.md](docs/output-schema/error-taxonomy.md) | Error categories, severity, scan phases |
| [docs/cli/usage.md](docs/cli/usage.md) | Full CLI flag reference, exit codes, SARIF integration |
| [docs/benchmarks/methodology.md](docs/benchmarks/methodology.md) | Benchmark intent, pipeline, and limitations |

---

## Development

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
pip install -e ".[dev]"
pytest tests/ -q                    # run all 773 tests
adaptersentry scan --help           # verify CLI

# Optional: build Rust extensions (OPT-04, requires Rust toolchain)
pip install maturin
cd adaptersentry-rs
VIRTUAL_ENV=$(python -c "import sys; print(sys.prefix)") maturin develop --release
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for code conventions and commit style.

---

## Requirements

```
python >= 3.11
safetensors >= 0.4.0
numpy >= 1.24.0
scipy >= 1.11.0
scikit-learn >= 1.3.0
pydantic >= 2.5.0
rich >= 13.0.0
psutil >= 5.9.0

# Optional extras
ray[default] >= 2.9.0      # pip install "adaptersentry[ray]"
huggingface_hub >= 0.20.0  # pip install "adaptersentry[bench]"
```

---

## Security

See [SECURITY.md](SECURITY.md) for the full security policy and disclosure procedures.

**Reporting a malicious adapter found in the wild:** Open a GitHub issue with the label
`malicious-adapter`. Include the HuggingFace repo ID and the M1 JSON report.

**Reporting a vulnerability in AdapterSentry:** Follow coordinated disclosure.
Do not open public GitHub issues for vulnerabilities in AdapterSentry itself.
See [SECURITY.md](SECURITY.md) for the full process.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
