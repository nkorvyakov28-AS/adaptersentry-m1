# CLI Usage

> Updated for v0.3.0 + M1 Analytics. New: `adaptersentry batch`, `--mode fast|full`,
> `--format summary-json|debug-json`, `--verbose` (text output), `--no-color`.

## Installation

```bash
pip install git+https://github.com/nkorvyakov28-AS/adaptersentry.git
# or for development:
pip install -e ".[dev]"
```

## Commands

### `adaptersentry scan`

Scan a single LoRA adapter file.

```
adaptersentry scan ADAPTER [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `text` | Output format: `text`, `summary-json`, `debug-json`, `json` (legacy), `sarif` |
| `--mode` | `full` | Scan depth: `full` (all detectors) or `fast` (~9× faster, equivalent detection) |
| `--output FILE` | stdout | Write output to FILE |
| `--rank R` | auto-detect | Declared LoRA rank r (overrides adapter_config metadata) |
| `--fail-on SEVERITY` | disabled | Exit 2 if any finding meets or exceeds SEVERITY |
| `--verbose` | off | Verbose text output: full score breakdown, per-layer findings, analysis quality block (text format only) |
| `--no-color` | off | Disable ANSI colour output (text format only) |
| `--quiet` | off | Suppress informational output |

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Completed; no findings at or above `--fail-on` threshold |
| 1 | Operational failure (file not found, parse error, non-LoRA format) |
| 2 | Findings at or above `--fail-on` threshold |

**Output formats:**

| Format | Schema | Use for |
|--------|--------|---------|
| `text` | Human-readable, ANSI colour | Local development, manual review |
| `summary-json` | `ScanResult` v1.0.0 — **stable** | CI gates, machine consumers |
| `debug-json` | `DebugReport` — **not stable** | Local debugging, per-layer detail |
| `json` | Legacy alias for `summary-json` | Backward compatibility |
| `sarif` | SARIF 2.1.0 | GitHub code scanning |

**Text output blocks (--format text):**

Compact (default):
```
VERDICT   — risk level, confidence, recommended action
TOP SIGNALS — top-3 sub-scores with lead reason
FINDINGS  — truncated finding list
```

Verbose (--verbose):
```
VERDICT           — same as compact
TOP SIGNALS       — same as compact
FINDINGS          — same as compact
SCORE BREAKDOWN   — all 7 sub-scores with weights and reasons
TOP SUSPICIOUS LAYERS — PerLayerFinding list (up to 10)
ANALYSIS QUALITY  — parse coverage, metadata completeness, feature completeness
```

---

### `adaptersentry batch`

Scan a directory of adapters with a parallel worker pool.

```
adaptersentry batch --input-dir DIR [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir DIR` | required | Directory to scan (recursive `.safetensors` glob) |
| `--workers N` | 4 | Number of parallel worker processes |
| `--mode` | `full` | Scan depth: `full` or `fast` |
| `--run-id ID` | auto-generated | Stable ID for this batch run (for resume) |
| `--output-dir DIR` | `./results` | Directory for result files |
| `--resume` | off | Resume a previous run after crash |
| `--force-rescan` | off | Re-scan adapters that already have persisted results |
| `--debug` | off | Write `.debug.json` files alongside summary JSON |
| `--fail-on SEVERITY` | disabled | Exit 2 if any adapter has findings at or above SEVERITY |
| `--rank R` | auto-detect | Applied to all adapters in the batch |

**Output layout:**

```
results/<run_id>/
  adapter_0000.json       ← ScanResult (summary-json, stable)
  adapter_0000.debug.json ← DebugReport (if --debug)
  run_summary.json        ← batch stats: ok/degraded/failed/cached
  run.jsonl               ← append-only audit trail
```

---

## Examples

### Single adapter scan

```bash
# Default text output — compact verdict + top signals
adaptersentry scan adapter.safetensors

# Full breakdown with per-layer findings
adaptersentry scan adapter.safetensors --verbose

# Stable JSON for CI gate
adaptersentry scan adapter.safetensors --format summary-json --output report.json

# Fast screening (~9× faster, equivalent detection)
adaptersentry scan adapter.safetensors --mode fast

# Fail CI on HIGH or CRITICAL
adaptersentry scan adapter.safetensors --fail-on HIGH

# SARIF for GitHub code scanning
adaptersentry scan adapter.safetensors --format sarif --output results.sarif

# No colour (CI logs, piped output)
adaptersentry scan adapter.safetensors --no-color
```

### Batch scanning

```bash
# Screen a corpus with fast mode
adaptersentry batch --input-dir ./adapters --mode fast --workers 8

# Full audit of flagged adapters
adaptersentry batch --input-dir ./flagged --mode full --workers 4 --fail-on HIGH

# Resume after crash
adaptersentry batch --input-dir ./adapters --run-id my-run --resume
```

### Recommended triage workflow

```bash
# Step 1: fast screen (quick risk overview)
adaptersentry batch --input-dir ./adapters --mode fast --workers 8 --output-dir ./screen

# Step 2: full audit on anything HIGH or CRITICAL
jq -r 'select(.verdict.overall_level == "HIGH" or .verdict.overall_level == "CRITICAL") | .artifact.local_path' \
  ./screen/*/run_summary.json > flagged.txt
adaptersentry batch --input-dir ./flagged --mode full --workers 4 --fail-on HIGH
```

---

## SARIF output

```yaml
# .github/workflows/adapter-scan.yml
- name: Scan LoRA adapter
  run: adaptersentry scan adapter.safetensors --format sarif --output results.sarif

- name: Upload SARIF results
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
  if: always()
```

`properties.security-severity` values: CRITICAL → 9.0, HIGH → 7.5, MEDIUM → 5.0, LOW → 2.5.

---

## Score breakdown (--format text --verbose)

The verbose text output includes a SCORE BREAKDOWN block showing how the final
risk score is decomposed across 7 feature families:

| Family | Weight | Signals |
|--------|--------|---------|
| `distribution` | 30% | kurtosis, skewness, percentiles, zero_ratio, entropy of ΔW |
| `similarity` | 20% | inter-layer cosine/Pearson, suspicious pairs |
| `parse` | 10% | parse_status, tensor errors |
| `metadata` | 10% | base_model, peft_type, target_modules, rank presence |
| `norm` | 10% | fro_norm_delta, delta_norm_ratio |
| `entropy` | 10% | value_repeat_ratio, byte_entropy, quantization_suspect_score |
| `training_pattern` | 10% | cross_layer_consistency, wasserstein, init_status |

Confidence verdict: `high` (≥0.75), `medium` (≥0.45), `low` (<0.45) — derived
from analysis coverage and completeness, not from anomaly signals.

---

## Legacy CLI

```bash
adaptersentry-m1 --adapter adapter.safetensors --output report.json
```
