# Getting Started with AdapterSentry

AdapterSentry is a static security scanner for LoRA adapters distributed as `.safetensors`
files. It inspects adapter weight tensors directly — without loading a base model or executing
any inference — to surface structural anomalies consistent with backdoor injection, safety
suppression, or behaviour redirection.

This guide takes you from installation through your first scan to a repeatable triage workflow.

---

## Prerequisites

- **Python 3.11 or later** (3.12 and 3.13 are also tested)
- **pip** (standard Python installer)

Core dependencies installed automatically with the package:

| Package | Minimum version |
|---------|----------------|
| `safetensors` | 0.4.0 |
| `numpy` | 1.24.0 |
| `scipy` | 1.11.0 |
| `scikit-learn` | 1.3.0 |
| `pydantic` | 2.5.0 |
| `rich` | 13.0.0 |
| `psutil` | 5.9.0 |

No PyTorch or base model weights are required.

---

## Installation

### Standard install

```bash
pip install adaptersentry
```

### With Ray backend (recommended for large corpora)

Ray provides better crash isolation and enables horizontal scaling. Use it for batch
scanning of more than ~50 adapters.

```bash
pip install "adaptersentry[ray]"
```

### With Rust hot-path extensions (maximum throughput)

The optional Rust extension (`adaptersentry-rs`) accelerates the full-mode hot path by up
to 57× on large corpora. Requires a Rust toolchain (`rustup`).

```bash
pip install maturin
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1/adaptersentry-rs
VIRTUAL_ENV=$(python -c "import sys; print(sys.prefix)") maturin develop --release
```

### Development install from source

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
pip install -e ".[dev]"
```

Verify the install:

```bash
adaptersentry scan --help
```

---

## Your First Scan

Run a scan on any `.safetensors` LoRA adapter file:

```bash
adaptersentry scan adapter.safetensors
```

The default output is a compact human-readable summary in the terminal. A typical clean
adapter looks like this:

```
VERDICT    LOW  |  confidence: high  |  action: allow
           Ensemble score 4.1 / 100

TOP SIGNALS
  distribution   0.12   kurtosis within expected range
  similarity     0.05   inter-layer cosine consistent across layers
  entropy        0.03   byte entropy normal

FINDINGS
  (none)
```

A flagged adapter looks like this:

```
VERDICT    HIGH  |  confidence: high  |  action: review
           Ensemble score 18.7 / 100

TOP SIGNALS
  distribution   0.61   kurtosis 47× above threshold in 12 layers
  similarity     0.42   3 non-adjacent layer pairs cosine > 0.85
  training_pattern  0.38   cross-layer consistency anomaly

FINDINGS
  [HIGH] Extreme kurtosis in model.layers.4.self_attn.q_proj — heavy-tailed weights
         consistent with sparse injection
  [MEDIUM] Selective layer targeting — modification concentrated in attention layers
```

### Understanding the output

**VERDICT block**

| Field | Description |
|-------|-------------|
| Risk level | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` — derived from ensemble score |
| Confidence | `high` / `medium` / `low` — derived from analysis coverage and data quality, not from anomaly signals |
| Action | `allow` / `review` / `block` — recommended action based on risk level |
| Ensemble score | Aggregate anomaly score from 0 to 100 |

**TOP SIGNALS block**

Shows the three feature families contributing most to the score. Each family has a
normalized sub-score (0–1) and the lead reason behind that score.

The seven families and their weights in the ensemble:

| Family | Weight | What it measures |
|--------|--------|-----------------|
| `distribution` | 30% | Kurtosis, skewness, percentiles, zero ratio, entropy of ΔW |
| `similarity` | 20% | Inter-layer cosine / Pearson; suspicious non-adjacent pairs |
| `parse` | 10% | Parse status and tensor-level errors |
| `metadata` | 10% | Presence of base_model, peft_type, target_modules, rank |
| `norm` | 10% | Frobenius norm of ΔW = B @ A; delta norm ratio |
| `entropy` | 10% | Value repeat ratio, byte entropy, quantization suspect score |
| `training_pattern` | 10% | Cross-layer consistency, Wasserstein, init status |

**FINDINGS block**

Each finding has a severity (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`), a stable rule
identifier from the rule catalog, and a plain-language description of the signal.

---

## Verbose Output

Add `--verbose` to see the full score breakdown, per-layer findings, and analysis
quality metrics:

```bash
adaptersentry scan adapter.safetensors --verbose
```

The verbose output adds three additional blocks:

```
SCORE BREAKDOWN
  distribution   weight=30%   score=0.61   kurtosis 47× above threshold ...
  similarity     weight=20%   score=0.42   cosine > 0.85 in pairs: ...
  ...

TOP SUSPICIOUS LAYERS
  #1  model.layers.4.self_attn.q_proj   severity=0.87   families=[distribution, norm]
  #2  model.layers.4.self_attn.k_proj   severity=0.81   families=[distribution]
  ...

ANALYSIS QUALITY
  parse_coverage: 100%   metadata_completeness: high   feature_completeness: 100%
```

The ANALYSIS QUALITY block tells you how complete the analysis was — low coverage
reduces confidence in the verdict.

---

## Risk Levels

| Level | Ensemble score | Meaning | Recommended action |
|-------|---------------|---------|-------------------|
| `LOW` | 0–6 | No anomalies detected | Load normally |
| `MEDIUM` | 7–13 | Elevated signal; likely benign | Review findings before loading |
| `HIGH` | 14–35 | Multiple independent detectors agree | Manual inspection required before loading |
| `CRITICAL` | 36–100 | Strong multi-signal evidence | Do not load without thorough review |

**A high score is the start of an investigation, not a conclusion.** Some legitimate
adapters — particularly those trained on narrow or synthetic corpora — produce elevated
scores due to distribution properties unrelated to malice. Use the per-layer findings
and score breakdown to understand which signals are driving the score.

When to act on each level:

- **LOW:** No action needed. The adapter is structurally consistent with benign training.
- **MEDIUM:** Review the TOP SIGNALS block. If the lead reason relates to missing metadata
  (no `base_model` field, no `peft_type`) rather than weight anomalies, this is an
  authoring quality issue rather than a security concern.
- **HIGH:** Run with `--verbose` and inspect the TOP SUSPICIOUS LAYERS block. Determine
  whether the anomaly concentration is in safety-critical module types (attention, MLP)
  or peripheral layers. If you need to load the adapter, consider running in a sandboxed
  environment first.
- **CRITICAL:** Do not load the adapter into a production system. If you control the
  source, investigate with `--verbose --format debug-json`. If the adapter came from a
  public hub, consider filing a report.

---

## Scan Modes

| Mode | When to use | Speed | Trade-off |
|------|-------------|-------|-----------|
| `--mode full` (default) | Security audits, final verification | 1× | Full SVD, IsolationForest, exact ΔW materialization |
| `--mode fast` | Corpus screening, CI pre-filter | ~9× faster | Randomised SVD top-50, 50K-element sample, no IsolationForest |

Fast mode preserves detection quality for typical backdoor patterns. Use it for initial
screening, then run full mode on anything that comes back `HIGH` or `CRITICAL`.

```bash
# Fast screening
adaptersentry scan adapter.safetensors --mode fast

# Full audit
adaptersentry scan adapter.safetensors --mode full
```

---

## Batch Scanning

Scan an entire directory of adapters in parallel:

```bash
adaptersentry batch --input-dir ./adapters --workers 4
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir DIR` | required | Directory to scan (recursively finds all `.safetensors` files) |
| `--workers N` | 4 | Number of parallel worker processes |
| `--mode full\|fast` | `full` | Scan depth |
| `--backend mp\|ray` | `mp` | Worker pool backend: multiprocessing or Ray |
| `--output-dir DIR` | `./results` | Directory for result files |
| `--run-id ID` | auto | Stable ID for this run (used with `--resume`) |
| `--resume` | off | Resume a previous run after crash |
| `--fail-on SEVERITY` | disabled | Exit code 2 if any adapter meets or exceeds this severity |
| `--debug` | off | Write `.debug.json` files with per-layer detail alongside each result |

### Output layout

Each batch run writes results under `--output-dir`:

```
results/<run_id>/
  adapter_0000.json        — ScanResult (stable, schema_version 1.0.0)
  adapter_0000.debug.json  — DebugReport (only with --debug)
  run_summary.json         — batch stats: ok / degraded / failed / cached
  run.jsonl                — append-only audit trail
```

### Recommended triage workflow

Use fast mode to screen the full corpus, then full mode on anything flagged:

```bash
# Step 1: fast screen
adaptersentry batch \
  --input-dir ./adapters \
  --mode fast \
  --workers 8 \
  --output-dir ./screen

# Step 2: extract HIGH and CRITICAL results
jq -r 'select(.verdict.overall_level == "HIGH" or .verdict.overall_level == "CRITICAL") | .artifact.local_path' \
  ./screen/*/run_summary.json > flagged.txt

# Step 3: full audit on flagged adapters
adaptersentry batch \
  --input-dir ./flagged \
  --mode full \
  --workers 4 \
  --fail-on HIGH \
  --output-dir ./audit
```

### Worker count guidelines

- **Fast mode:** up to 8 workers safely on a 16 GB machine (~455 MB peak RSS per worker)
- **Full mode, multiprocessing backend:** use at most 4 workers (up to ~524 MB peak per worker)
- **Full mode, Ray backend:** up to 8 workers with crash isolation; Ray replaces OOM-killed actors automatically

```bash
# Fast mode, Ray backend (recommended for large corpora)
adaptersentry batch \
  --input-dir ./adapters \
  --mode fast \
  --workers 8 \
  --backend ray

# Full audit, Ray backend
adaptersentry batch \
  --input-dir ./flagged \
  --mode full \
  --workers 8 \
  --backend ray
```

---

## CI Integration

### Fail on HIGH or CRITICAL findings

```bash
adaptersentry scan adapter.safetensors --fail-on HIGH
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | Completed; no findings at or above `--fail-on` threshold |
| 1 | Operational failure (file not found, parse error, non-LoRA format) |
| 2 | Findings at or above `--fail-on` threshold |

### Stable JSON for machine consumers

```bash
adaptersentry scan adapter.safetensors --format summary-json --output report.json
```

The `summary-json` format emits a versioned `ScanResult` document
(`schema_version: "1.0.0"`) — the stable public contract. The `scan_id` field is
deterministic: the same adapter file with the same config always produces the same ID,
which makes deduplication and caching straightforward.

### SARIF for GitHub code scanning

```bash
adaptersentry scan adapter.safetensors --format sarif --output results.sarif
```

Add to a GitHub Actions workflow:

```yaml
- name: Scan LoRA adapter
  run: adaptersentry scan adapter.safetensors --format sarif --output results.sarif

- name: Upload to GitHub code scanning
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: results.sarif
  if: always()
```

SARIF severity mapping: `CRITICAL` → 9.0, `HIGH` → 7.5, `MEDIUM` → 5.0, `LOW` → 2.5.

---

## Python API

Use the Python API to integrate scanning into your own tooling.

### Basic scan

```python
from pathlib import Path
from adaptersentry import scan

report = scan(Path("adapter.safetensors"))
print(report.risk_summary.risk_level)   # LOW / MEDIUM / HIGH / CRITICAL
print(report.risk_summary.ensemble_score)
```

### Fast mode

```python
report = scan(Path("adapter.safetensors"), fast=True)
```

### Score breakdown across all 7 families

```python
from adaptersentry.scoring.score_breakdown import compute_score_breakdown

breakdown = compute_score_breakdown(report)
for sub in breakdown.sub_scores:
    print(f"{sub.family:20s}  score={sub.normalized_score:.2f}  {sub.top_reasons}")
```

### Confidence in the result

Confidence is derived from analysis coverage and completeness — not from anomaly signals.
A low-confidence verdict means the analysis was incomplete (e.g. many parse errors or
missing metadata), not that the adapter is more or less suspicious.

```python
from adaptersentry.scoring.confidence import compute_confidence_score, compute_quality_score

quality = compute_quality_score(report)
conf = compute_confidence_score(report, quality)
print(conf.verdict_certainty)           # high / medium / low
print(conf.coverage_score)
```

### Check specific findings

```python
for finding in report.findings:
    print(finding.severity, finding.rule_id, finding.message)
```

### Full example: screen a directory

```python
from pathlib import Path
from adaptersentry import scan

adapter_dir = Path("./adapters")
for adapter_path in adapter_dir.glob("**/*.safetensors"):
    report = scan(adapter_path, fast=True)
    level = report.risk_summary.risk_level
    score = report.risk_summary.ensemble_score
    print(f"{adapter_path.name:50s}  {level:8s}  {score:.1f}")
```

---

## What AdapterSentry Does Not Do

- It does not load base model weights or execute inference.
- It does not confirm that a high-scoring adapter is malicious — only that its weight
  tensors contain structural anomalies warranting investigation.
- It does not scan non-LoRA adapter formats (IA3, prefix-tuning, full fine-tunes) by
  default. Files without `lora_A` / `lora_B` tensor pairs are rejected with a
  `NO_LORA_PAIRS` error, which is expected behaviour.
- It does not guarantee detection of novel attack patterns outside its current signal
  set. It surfaces known structural indicators; adversarially crafted adapters that
  mimic benign distributions may score LOW.

---

## Next Steps

- **CLI flag reference:** [docs/cli/usage.md](../cli/usage.md)
- **Output schema:** [docs/output-schema/scan-result.md](../output-schema/scan-result.md)
- **Scan modes in depth:** [docs/architecture/scan-modes.md](../architecture/scan-modes.md)
- **Batch scan engine:** [docs/architecture/scan-engine.md](../architecture/scan-engine.md)
- **Detection methods:** [docs/architecture/m1-architecture.md](../architecture/m1-architecture.md)
- **Error taxonomy:** [docs/output-schema/error-taxonomy.md](../output-schema/error-taxonomy.md)
