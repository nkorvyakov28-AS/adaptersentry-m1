# Configuration Reference

> AdapterSentry v1.0.x — M1 Static Analyzer

This document is the authoritative reference for all configuration surface in AdapterSentry:
CLI flags, Python API options, environment variables, and the result cache.

---

## Scan modes (`--mode`)

Two modes control the depth-vs-speed trade-off. Both produce a full `ScanResult`; the
difference is in which computational steps are skipped or approximated.

```
--mode full   (default)   All detectors at full depth. Use for security audits and
                          final verification before deployment.
--mode fast               Optimised for throughput screening of large corpora.
```

### What changes between `full` and `fast`

| Step | `full` | `fast` |
|------|--------|--------|
| SVD | Complete spectrum on every matrix | Truncated SVD, top-50 singular values, for matrices ≥ 512×512 (`randomized_svd k=50`) |
| Statistical moments (kurtosis, std, etc.) | Full tensor | 50K-element random sample (deterministic seed) for tensors with > 100K elements |
| IsolationForest | Runs on every layer | Skipped for tensors with > 5M elements |
| Entropy/compression features | Full O(n) — same in both modes | Full O(n) — same in both modes |
| Inter-layer similarity | Full ΔW materialised (out×in ≤ 4M) or proxy | lora_A rows as ΔW proxy (faster, same vector size cap of 10K) |
| ΔW norm (Frobenius / max-abs) | Full float32 B@A | Cholesky approximation for layers where out×in > 4M |

**Detection quality:** energy concentration and kurtosis signals live in the top singular
values and in the distribution tail — both of which sampling captures. Fast mode preserves
detection quality for typical backdoor patterns. Estimated throughput difference: ~9×.

### Typical usage

```bash
# Step 1: fast screening across a corpus
adaptersentry batch --input-dir ./adapters --mode fast --workers 8

# Step 2: full audit on HIGH/CRITICAL results from step 1
adaptersentry batch --input-dir ./flagged --mode full --workers 4
```

---

## Batch engine options (`adaptersentry batch`)

### `--workers N`

Number of parallel worker processes. Default: `4`.

```bash
adaptersentry batch --input-dir ./adapters --workers 8
```

Guidelines:
- **fast mode:** safe at up to 8 workers (peak ~455 MB RSS/worker on real corpus).
- **full mode:** limit to 4 workers on a 16 GB machine (peak ~524 MB RSS/worker on
  worst-case real adapters). Exceeding this risks OOM for very large adapters.
- Set BLAS thread limits (see Environment Variables below) when running with more than
  1 worker to prevent CPU over-subscription.

### `--backend ray|mp`

Worker pool implementation. Default: `mp`.

```bash
adaptersentry batch --input-dir ./adapters --backend ray --workers 8
```

| Backend | Description | Requires |
|---------|-------------|----------|
| `mp` | `multiprocessing.Pool` — default, no extra deps | nothing |
| `ray` | Ray actor pool — crash isolation, BLAS thread fix baked into actor `__init__`, supports horizontal scaling and remote clusters | `pip install "adaptersentry[ray]"` |

Ray is recommended for large corpora and production use. Advantages over `mp`:
- OOM-killed actors are replaced without stalling the batch.
- BLAS thread limits applied automatically inside every actor (no env vars needed).
- `--ray-address` connects to an existing remote Ray cluster.

### `--ray-address ADDRESS`

Connect to an existing Ray cluster instead of starting a local one. Only used with
`--backend ray`. Omit to start Ray locally.

```bash
adaptersentry batch --input-dir ./adapters --backend ray \
    --ray-address ray://head-node:10001
```

### `--run-id ID`

Stable identifier for this batch run. Used as the output subdirectory name and as the
key for resume. Auto-generated from timestamp if omitted.

```bash
adaptersentry batch --input-dir ./adapters --run-id corp-audit-2026-05
```

### `--resume`

Resume a previous run after a crash or interruption. Resets non-terminal jobs and
continues from where the batch stopped. Pass the same `--run-id` used originally.

```bash
adaptersentry batch --input-dir ./adapters --run-id corp-audit-2026-05 --resume
```

### `--force-rescan`

Re-scan all adapters even if they already have terminal results in the manifest.
Overrides both the manifest state and the cache.

### `--input-list FILE`

Alternative to `--input-dir`: provide a text file with one adapter path per line.
Mutually exclusive with `--input-dir`.

```bash
adaptersentry batch --input-list flagged.txt --mode full --workers 4
```

### `--output-dir DIR`

Directory for per-adapter result files. Default: `./results`. A subdirectory named
after the `run-id` is created inside this directory.

Output layout:
```
results/<run_id>/
  adapter_0000.json         ScanResult (summary-json, stable)
  adapter_0000.debug.json   DebugReport (only with --debug)
  run_summary.json          batch stats: ok/degraded/failed/cached
  run.jsonl                 append-only audit trail (all results)
```

### `--debug`

Write `.debug.json` files (per-layer tensor records and feature families) alongside each
summary JSON. Not a stable schema — for local debugging only.

### `--no-cache`

Disable result caching. Every adapter is re-scanned regardless of cache state.

### `--cache-dir DIR`

Override the cache store root. Default: `~/.adaptersentry/cache`. Set to `/dev/null`
to disable caching (equivalent to `--no-cache`).

---

## Output format options (`--format`)

Applies to `adaptersentry scan`. The batch command always writes `summary-json` files.

```bash
adaptersentry scan adapter.safetensors --format summary-json
```

| Format | Schema | Stability | Use for |
|--------|--------|-----------|---------|
| `text` | Human-readable, ANSI colour | — | Local development, manual review |
| `summary-json` | `ScanResult` v1.0.0 | **Stable** | CI gates, machine consumers, automation |
| `debug-json` | `DebugReport` (extends `ScanResult`) | Not stable | Local debugging, per-layer detail |
| `json` | Legacy alias for `summary-json` | Deprecated | Backward compatibility only |
| `sarif` | SARIF 2.1.0 | Stable | GitHub code scanning integration |

### `--output FILE`

Write output to a file instead of stdout.

```bash
adaptersentry scan adapter.safetensors --format summary-json --output report.json
adaptersentry scan adapter.safetensors --format sarif --output results.sarif
```

---

## CI integration (`--fail-on`)

Exit with code `2` when any finding meets or exceeds the specified severity threshold.

```bash
adaptersentry scan adapter.safetensors --fail-on HIGH
adaptersentry batch --input-dir ./adapters --fail-on MEDIUM
```

Severity order: `LOW` < `MEDIUM` < `HIGH` < `CRITICAL`

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Completed; no findings at or above `--fail-on` threshold (or `--fail-on` not set) |
| `1` | Operational failure: file not found, parse error, non-LoRA format |
| `2` | Findings at or above `--fail-on` threshold detected |

### GitHub Actions example

```yaml
- name: Scan LoRA adapter
  run: adaptersentry scan adapter.safetensors --format sarif --output results.sarif --fail-on HIGH

- name: Upload to GitHub code scanning
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
  if: always()
```

SARIF `properties.security-severity` mapping: `CRITICAL` → 9.0, `HIGH` → 7.5,
`MEDIUM` → 5.0, `LOW` → 2.5.

### Two-phase triage pipeline

```bash
# Phase 1: fast screen to identify candidates
adaptersentry batch --input-dir ./adapters --mode fast --workers 8 --output-dir ./screen

# Phase 2: full audit on HIGH+ results, fail CI on confirmation
jq -r 'select(.verdict.overall_level == "HIGH" or .verdict.overall_level == "CRITICAL") | .artifact.local_path' \
  ./screen/*/run_summary.json > flagged.txt
adaptersentry batch --input-list flagged.txt --mode full --workers 4 --fail-on HIGH
```

---

## Display flags (`--verbose`, `--no-color`, `--quiet`)

These flags apply to `adaptersentry scan` with `--format text` (the default).

### `--verbose`

Show the full score breakdown, per-layer suspicious layer ranking, and analysis quality
block in addition to the compact default output.

```bash
adaptersentry scan adapter.safetensors --verbose
```

Compact output (default):
```
VERDICT      risk level, confidence, recommended action
TOP SIGNALS  top-3 sub-scores with lead reason
FINDINGS     truncated finding list
```

Verbose output (`--verbose`):
```
VERDICT               same as compact
TOP SIGNALS           same as compact
FINDINGS              same as compact
SCORE BREAKDOWN       all 7 families with weights and top reasons
TOP SUSPICIOUS LAYERS PerLayerFinding list (up to 10), ranked by severity_score
ANALYSIS QUALITY      parse coverage, metadata completeness, feature completeness
```

### `--no-color`

Disable ANSI colour output. Useful for CI log systems, piped output, or environments
that do not support terminal colour codes.

```bash
adaptersentry scan adapter.safetensors --no-color
```

### `--quiet`

Suppress informational output (only applies to the scan command's operational messages,
not the analysis result itself).

---

## Additional scan flags

### `--rank R`

Declare the LoRA rank `r` explicitly, overriding what the adapter's own `adapter_config`
metadata reports. Applied to all adapters in a batch when used with `adaptersentry batch`.

```bash
adaptersentry scan adapter.safetensors --rank 16
```

---

## Python API

### `scan()` — single adapter

```python
from pathlib import Path
from adaptersentry import scan

# Full analysis (default)
report = scan(Path("adapter.safetensors"))

# Fast mode
report = scan(Path("adapter.safetensors"), fast=True)

# With declared rank override
report = scan(Path("adapter.safetensors"), claimed_rank=16)
```

`scan()` returns an `AdapterReport`. Key fields:

```python
report.risk_summary.risk_level          # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
report.risk_summary.ensemble_score      # float 0–100
report.risk_summary.training_status     # "TRAINED" / "INIT_ONLY" / "PARTIALLY_TRAINED"
report.findings                         # list[Finding]
report.errors                           # list[ScanError]
report.analysis_mode                    # "full" / "degraded"
```

### `AnalyzerConfig` — batch engine configuration

`AnalyzerConfig` is the canonical representation of the active detector configuration.
Its SHA-256 hash (`config_hash()`) is the cache invalidation key: any change to
weights, thresholds, enabled families, schema version, or tool version produces a new
hash and automatically invalidates stale cache entries.

```python
from adaptersentry.engine.config import AnalyzerConfig, ScanMode

config = AnalyzerConfig(scan_mode=ScanMode.FAST)
print(config.config_hash())   # "sha256:..."
```

`AnalyzerConfig` fields (all optional — defaults shown):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scan_mode` | `ScanMode` | `ScanMode.FULL` | `FULL` or `FAST` |
| `schema_version` | `str` | `"1.0.0"` | Schema version (affects hash) |
| `tool_version` | `str` | current version | Tool version (affects hash) |
| `enabled_families` | `list[str]` | `["norm", "distribution", "entropy", "outlier", "spectral"]` | Active feature families |
| `detector_weights` | `dict[str, float]` | ensemble defaults | Per-detector weights |
| `thresholds` | `dict[str, float]` | detection defaults | Per-signal thresholds |

`AnalyzerConfig` is frozen (immutable after construction). The batch engine constructs
one instance per run and derives `config_hash` from it — do not construct separate
instances with different fields within a single batch.

### Score breakdown and confidence

```python
from adaptersentry.scoring.score_breakdown import compute_score_breakdown
from adaptersentry.scoring.confidence import compute_confidence_score, compute_quality_score

breakdown = compute_score_breakdown(report)
for sub in breakdown.sub_scores:
    print(f"{sub.family}: {sub.normalized_score:.2f}  {sub.top_reasons}")

quality = compute_quality_score(report)
conf = compute_confidence_score(report, quality)
print(conf.verdict_certainty)   # "high" / "medium" / "low"
```

---

## Environment variables

These variables control BLAS thread counts. They are **critical for batch performance**:
without them, each worker process spawns as many OS threads as there are CPU cores,
causing severe CPU over-subscription when multiple workers run simultaneously.

| Variable | Effect |
|----------|--------|
| `OMP_NUM_THREADS` | OpenMP thread count (numpy, scipy on many platforms) |
| `OPENBLAS_NUM_THREADS` | OpenBLAS thread count |
| `MKL_NUM_THREADS` | Intel MKL thread count |

Set all three to `1` for batch runs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    adaptersentry batch --input-dir ./adapters --workers 8 --mode fast
```

Without this, load average can reach `workers × CPU_cores` (e.g., 48 on an 8-core
machine with 8 workers), causing a ~6× throughput regression.

**Note:** When using `--backend ray`, BLAS limits are applied automatically inside
every Ray actor at initialisation time. No environment variables are needed.

---

## BLAKE3 result cache

The batch engine maintains a content-addressed result cache to avoid rescanning
adapters that have already been processed with the same configuration.

### Default location

```
~/.adaptersentry/
  cache/
    index.sqlite              lookup index (content_hash + config_hash → result)
    objects/
      {hash[:2]}/
        {hash[2:]}.gz         gzip-compressed ScanResult JSON
  manifest.sqlite             per-run job state machine
```

### Cache key

A cache entry is keyed on the tuple `(content_hash, analyzer_config_hash)`:
- `content_hash` — BLAKE3 hash of the adapter file bytes (content-addressed).
- `analyzer_config_hash` — SHA-256 of the canonical `AnalyzerConfig` JSON.

A version bump in AdapterSentry always changes `analyzer_config_hash`, which
automatically invalidates all cached results from previous versions. This prevents
silent false-negative regressions from stale cache entries.

### Cache integrity

On every cache read, the stored result file is re-hashed and compared to the
recorded `result_hash`. A mismatch deletes the entry and triggers a full rescan.
The cache is fail-closed: a result whose integrity cannot be verified is never served.

### Disabling the cache

```bash
# Per-run: skip cache lookup and writes
adaptersentry batch --input-dir ./adapters --no-cache

# Per-run: use a custom cache directory
adaptersentry batch --input-dir ./adapters --cache-dir /tmp/as-cache

# Effectively disable: point cache at /dev/null
adaptersentry batch --input-dir ./adapters --cache-dir /dev/null
```

### Cache invalidation

The cache is invalidated automatically when:
- The adapter file changes (new content hash).
- The AdapterSentry version changes (new config hash).
- Any detector weight, threshold, or enabled family changes.

There is no manual cache clear command. To force rescan without clearing the cache
for other adapters, use `--force-rescan` on the specific run.

---

## Quick reference

### `adaptersentry scan` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--format FORMAT` | `text` | Output format: `text`, `summary-json`, `debug-json`, `json`, `sarif` |
| `--mode MODE` | `full` | Scan depth: `full` or `fast` |
| `--output FILE` | stdout | Write output to file |
| `--rank R` | auto | Declared LoRA rank (overrides metadata) |
| `--fail-on SEVERITY` | off | Exit 2 if findings meet or exceed SEVERITY |
| `--verbose` | off | Full score breakdown + per-layer findings (text only) |
| `--no-color` | off | Disable ANSI colour (text only) |
| `--quiet` | off | Suppress informational output |
### `adaptersentry batch` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir DIR` | required* | Directory to scan recursively |
| `--input-list FILE` | required* | Text file with one path per line |
| `--workers N` | `4` | Number of parallel worker processes |
| `--mode MODE` | `full` | Scan depth: `full` or `fast` |
| `--backend BACKEND` | `mp` | Worker pool: `mp` or `ray` |
| `--ray-address ADDRESS` | local | Ray cluster address (ray backend only) |
| `--run-id ID` | timestamp | Batch run identifier |
| `--output-dir DIR` | `./results` | Output directory |
| `--cache-dir DIR` | `~/.adaptersentry/cache` | Cache store root |
| `--no-cache` | off | Disable result caching |
| `--resume` | off | Resume a previous run |
| `--force-rescan` | off | Ignore existing terminal state |
| `--debug` | off | Write debug JSON alongside summary JSON |
| `--fail-on SEVERITY` | off | Exit 2 if findings meet or exceed SEVERITY |
| `--rank R` | auto | Declared rank applied to all adapters |

*`--input-dir` and `--input-list` are mutually exclusive; one is required.
