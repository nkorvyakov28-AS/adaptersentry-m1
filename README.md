# AdapterSentry

![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Version](https://img.shields.io/badge/version-2.0.0-blue)

AdapterSentry is a static security scanner for LoRA adapters distributed as `.safetensors`
files. A LoRA adapter is a small file that changes how a large language model behaves; anyone
can publish one, and a poisoned adapter can carry a backdoor that stays silent until a trigger
appears in the input. AdapterSentry inspects the adapter's weights **before** the adapter is
loaded, without running the model and without knowing the trigger.

**v2.0.0** replaces the analysis engine: exact statistics of the effective update
ΔW = s·B·A (no sampling, no proxies), streaming analysis that scales to 70B–405B and MoE
models in tens of megabytes of memory, a comparison of every module with its own family inside
the adapter, structural findings, and a new result contract (`ScanResult 2.0.0`).
See [CHANGELOG.md](CHANGELOG.md) and [Migrating from 1.x](#migrating-from-1x).

---

## What it tells you

```text
$ adaptersentry scan adapter_model.safetensors
AdapterSentry 2.0.0 · adapter_model.safetensors
================================================================
Verdict:  REVIEW (MEDIUM) · confidence medium · behavioural check recommended
Status:   ok · policy default · mode full
Why:
  - [LOW] NO_REFERENCE_PROFILE: No calibrated reference for this model family; verdict relies on intra-adapter comparison
  - [MEDIUM] INTRA_OUTLIER: Module departs from its family: down_proj layer 47, robust z = 9.4 (threshold 8.0, uncalibrated)
Adapter:  peft_lora, family llama (config), 80 layers, r=64 (declared 64), α=128, scaling alpha_over_r, training trained
Coverage: 560/560 modules analysed; 0 tensor(s) not analysed
Intra:    max robust z 9.4 · 2 outlier module(s) · energy Gini 0.41
Findings (1):
  [MEDIUM] INTRA_DEPTH_SPIKE — Module departs from its family  (layer 47, down_proj)
```
*(Illustrative output.)*

The verdict is one of **`allow`**, **`review`** or **`block`**, with the reasons, a confidence
level and what limits it. CI gates read `verdict.action`.

---

## Quick start

```bash
pip install adaptersentry

adaptersentry scan adapter_model.safetensors                     # text summary
adaptersentry scan adapter_model.safetensors --format json       # ScanResult 2.0.0
adaptersentry scan adapter_model.safetensors --format sarif \
    --output adaptersentry.sarif --fail-on review                # GitHub code scanning + CI gate
adaptersentry batch --input-dir adapters/ --workers 8 --fail-on review
```

```python
from pathlib import Path
from adaptersentry import scan

result = scan(Path("adapter_model.safetensors"))
print(result.verdict.action, result.verdict.level)
for reason in result.verdict.reasons:
    print(reason.code, reason.message)
```

Put `adapter_config.json` next to the `.safetensors` file (as PEFT saves it): the scan needs
`lora_alpha` to know the real size of the update.

| Option | Meaning |
|---|---|
| `--format text\|json\|full-json\|sarif` | output; `full-json` adds per-module features |
| `--policy default\|strict` | `strict` blocks on structural red flags (see below) |
| `--fail-on review\|block` | exit code 2 if the verdict reaches this action |
| `--mode full\|fast` | `fast` uses fewer samples for the kurtosis estimate |
| `--include-modules`, `--include-heads` | per-module / per-attention-head detail in JSON |
| `--full-paths` | report the absolute path instead of the file name |

Exit codes: `0` completed, `1` the adapter could not be analysed (the verdict is still
`review`), `2` the `--fail-on` threshold was reached.

---

## How it works

```
file ─► validate ─► inventory ─► per module (streamed) ─► compare within ─► structural ─► verdict
        header       every         exact ΔW features         each module       findings
        & limits     tensor        via r×r cores              family
```

1. **Validate before reading.** The file must be a regular `.safetensors` file; the header
   length, dtype, shape, byte ranges, element counts and LoRA rank are checked from the header
   before any tensor is allocated.
2. **Inventory.** Every tensor is classified: LoRA A/B pairs (with layer, expert and module),
   full weights shipped via `modules_to_save`, DoRA vectors, trainable-token deltas, and
   anything that cannot be analysed — nothing is skipped silently.
3. **Exact features, one module at a time.** For ΔW = s·B·A, where s = α/r (or α/√r with
   rsLoRA) comes from `adapter_config.json`, the singular values, norms and row/column norms
   are computed exactly through the r×r matrices AAᵀ and BᵀB — ΔW itself (235 M values for a
   70B MLP layer) is never built. Features describe the **shape** of the update, which the
   literature finds informative, rather than its size:
   stable rank, participation ratio, spectral entropy, concentration of the leading direction
   versus a random baseline, entry kurtosis (estimated by importance sampling of rows so rare
   spiky rows are not missed), and how concentrated the update is on a few output rows or
   input columns — for `lm_head` and embeddings, on a few tokens.
4. **Compare within the adapter.** Each module is compared with its family (all `q_proj`, all
   `down_proj`, …): the smooth trend with depth is removed with a Theil–Sen line, then robust
   z-scores (median/MAD) mark modules that depart from their family. This needs no reference
   data and is insensitive to the adapter's overall strength. It runs for families of at
   least 16 modules.
5. **Structural findings** that numbers cannot show: complete `lm_head` / embedding / router
   matrices shipped in the adapter, LoRA on a mixture-of-experts router, ranks that differ from
   the declared `r` (typical of merged adapters), code/pickle/archive files next to the adapter.
6. **Verdict** — one rule set, fail-closed:

| Situation | Action |
|---|---|
| File could not be parsed, or part of it was not analysed | at least `review` |
| A module departs from its family (robust z ≥ 8) | `review` |
| Structural red flag | `review`; `block` with `--policy strict` |
| Reference profile for the model family says p ≤ α | `review` / `block` *(reference profiles: next release)* |
| Otherwise | `allow` |

Without a calibrated reference profile the default policy never blocks: the false-positive
rate of a block would be unknown.

### Performance

Measured on an 8-CPU server with a synthetic Llama-3.3-70B-shaped adapter
(80 layers × 7 modules, r = 64, 1.66 GB bf16): full analysis in **~15 s**, peak memory
**+93 MB**. The 1.x engine needed ~3.3 GB just to load the same file.

---

## Limits — read before relying on a verdict

- **Static analysis is a first filter, not a proof.** Research on weight-space detection
  shows strong results in the lab and clear limits: detectors trained on known attacks fail on
  new ones, and on larger models the variation between honest adapters trained with different
  random seeds can exceed the signal of a backdoor. A low-risk verdict is not evidence that an
  adapter is safe; behavioural verification in an isolated environment remains necessary for
  adapters you do not trust.
- **Thresholds are not yet calibrated.** Intra-adapter thresholds (|z| > 3.5 outlier,
  ≥ 8 review) were set on synthetic data. Reference profiles per model family and a measured
  detection rate on labelled clean/poisoned adapters are the next step; until then the
  confidence of a verdict is at most `medium`.
- **An attacker who knows the detector** can try to spread a backdoor across many directions
  and modules, or pad the update with inert components. Static analysis without base-model
  information cannot fully rule this out.
- **Not analysed:** full weight matrices (they need the base model), trainable-token deltas;
  DoRA adapters are analysed partially. Each of these is reported and lowers the verdict's
  confidence or raises it to `review`.
- **Pickle-based formats** (`.bin`, `.pt`) are never deserialised. Use a dedicated
  serialisation scanner for them.

---

## Output: ScanResult 2.0.0

`--format json` produces one document per adapter: `scan` (who/when/how), `artifact` (hashes,
provenance — file name only unless `--full-paths`), `adapter` (format, family, rank, α,
scaling, modules), `status` + `coverage` (what was and was not analysed), `verdict`,
`anomaly` (what the verdict rests on), `findings` (with exact layer/module/tensor locations),
optional `modules`, and `errors`.

- Field reference: [docs/output-schema/scan-result.md](docs/output-schema/scan-result.md)
- JSON Schema: [docs/output-schema/scan-result-2.0.0.schema.json](docs/output-schema/scan-result-2.0.0.schema.json)
- `adaptersentry.load_scan_result()` validates a document and rejects other major versions.

---

## Migrating from 1.x

| 1.x | 2.0.0 |
|---|---|
| `analyze()`, `scan()` → `AdapterReport`, `scan_to_result()` | `scan()` → `ScanResult` |
| `verdict.recommended_action` | `verdict.action` |
| `overall_score`, `ensemble.score` (0–100) | removed — `verdict.level` + `anomaly` (z-scores, later p-values) |
| `status`, `parse_status`, `analysis_mode` | `status` + `coverage` |
| `--format summary-json` / `debug-json` | `--format json` / `full-json` |
| `--fail-on LOW…CRITICAL` (finding severity) | `--fail-on review\|block` (verdict action) |
| `--rank` | removed — rank is read from the tensors and checked against the config |
| `adaptersentry-m1` command | `adaptersentry scan` |

1.x results are not converted: re-scan the adapters.

---

## Development

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
uv sync --frozen --extra dev        # or: pip install -e ".[dev]"
pytest tests/ -q
```

Architecture: [docs/architecture/m1-architecture.md](docs/architecture/m1-architecture.md) ·
CLI: [docs/cli/usage.md](docs/cli/usage.md) · Batch engine:
[docs/architecture/scan-engine.md](docs/architecture/scan-engine.md)

AdapterSentry M1 is the open static-analysis layer. Planned: M2 Behavioral Sandbox — advanced
behavioral analysis.

## Background

The feature design follows published work on weight-space backdoor detection in LoRA adapters,
including *Detecting Backdoored LoRAs from Weights Alone* (arXiv:2602.15195),
*Token-Level Generalization in LoRA Adapter Backdoors* (arXiv:2605.30189) and Z-PEFT
(arXiv:2608.02271). Their results motivate the choices above: shape over magnitude,
per-module and per-head features, MLP coverage, and one-class calibration instead of a
classifier trained on known attacks.

## Security

See [SECURITY.md](SECURITY.md). Report a malicious adapter found in the wild with a GitHub
issue labelled `malicious-adapter`; report vulnerabilities in AdapterSentry privately.

## License

Apache 2.0 — see [LICENSE](LICENSE).
