# Getting Started

AdapterSentry scans a LoRA adapter (`.safetensors`) before you load it and tells you whether
to `allow` it, `review` it, or `block` it. It does not load a base model and does not run
inference.

Read this first: static analysis is a **first filter**. Detection thresholds are not yet
calibrated, and an `allow` verdict is not evidence that an adapter is safe. For adapters you do
not trust, behavioural testing in an isolated environment is still needed.

## Install

Python 3.11 or newer.

```bash
pip install adaptersentry              # scanner and CLI
pip install "adaptersentry[ray]"       # optional: Ray backend for batch scans
```

From source:

```bash
git clone https://github.com/nkorvyakov28-AS/adaptersentry-m1
cd adaptersentry-m1
uv sync --frozen --extra dev           # or: pip install -e ".[dev]"
```

## Scan one adapter

Keep `adapter_config.json` next to the weights, as PEFT saves them: the scan needs
`lora_alpha` to know the real size of the update.

```bash
adaptersentry scan path/to/adapter_model.safetensors
```

The text report shows:

- **Verdict** — `ALLOW`, `REVIEW` or `BLOCK`, with a level and a confidence.
- **Why** — the reasons (reason codes are listed in
  [scan-result.md](../output-schema/scan-result.md#verdict)).
- **Adapter** — format, model family, layers, rank, α, scaling, training state.
- **Coverage** — how many modules were analysed and what was not.
- **Intra** — how far the most unusual module is from its family (robust z).
- **Findings** — individual findings with layer and module.

For machines:

```bash
adaptersentry scan adapter_model.safetensors --format json --output result.json
adaptersentry scan adapter_model.safetensors --format full-json --include-heads   # with per-module/per-head features
```

## What a verdict means

| Action | When |
|---|---|
| `allow` | analysis completed and nothing stood out |
| `review` | a module departs strongly from its family, a structural red flag is present, or part of the file could not be analysed |
| `block` | only with `--policy strict` and a structural red flag (reference profiles that could justify a block by default are not released yet) |

A failed or partial parse is never `allow`.

## Scan many adapters

```bash
adaptersentry batch --input-dir adapters/ --workers 8 --output-dir results/
```

Results land in `results/<run_id>/` (one JSON per adapter, `run.jsonl`, `run_summary.json`).
Unchanged adapters are served from the cache on the next run. An interrupted run continues
with `--run-id <id> --resume`. See [scan-engine.md](../architecture/scan-engine.md).

## Use in CI

```bash
adaptersentry scan adapter_model.safetensors --format sarif \
    --output adaptersentry.sarif --fail-on review
```

Exit code `2` means the verdict reached the `--fail-on` action; `1` means the adapter could not
be analysed. Upload the SARIF file to GitHub code scanning to see findings in the Security tab.

## Python

```python
from pathlib import Path
from adaptersentry import scan

result = scan(Path("adapter_model.safetensors"))
print(result.verdict.action, result.verdict.level, result.verdict.confidence.level)
for reason in result.verdict.reasons:
    print(reason.code, reason.message)
for finding in result.findings:
    print(finding.rule_id, finding.severity, finding.title)
```

## What AdapterSentry does not do

- It does not deserialise pickle-based formats (`.bin`, `.pt`); use a dedicated
  serialisation scanner for those.
- It does not analyse full weight matrices shipped with the adapter (they need the base
  model); it reports them.
- It does not run the model, so it cannot confirm or rule out a backdoor.

## Next

- [configuration.md](configuration.md) — all flags and the Python API
- [../output-schema/scan-result.md](../output-schema/scan-result.md) — result fields
- [../architecture/overview.md](../architecture/overview.md) — how the scan works
