# Configuration Reference

AdapterSentry has no configuration file. Behaviour is set by command-line flags (or the
matching `scan()` arguments). Every setting that changes a result is recorded in
`scan.config_hash`.

## `adaptersentry scan ADAPTER`

| Flag | Default | Meaning |
|---|---|---|
| `--format text\|json\|full-json\|sarif` | `text` | `json` is ScanResult 2.0.0; `full-json` adds per-module features; `sarif` is SARIF 2.1.0 |
| `--output FILE` | stdout | write the report to a file |
| `--mode full\|fast` | `full` | see [scan-modes.md](../architecture/scan-modes.md) |
| `--policy default\|strict` | `default` | `strict` turns structural red flags into `block` |
| `--fail-on review\|block` | none | exit code 2 if the verdict action is at least this |
| `--include-modules` | off | include per-module features in JSON |
| `--include-heads` | off | also include per-attention-head features (implies `--include-modules`) |
| `--full-paths` | off | report the absolute path instead of the file name |
| `--no-color` | off | no ANSI colours in text output (also off when not writing to a terminal) |

Global flags: `adaptersentry --version`, `adaptersentry --verbose COMMAND ...` (debug logging
to stderr).

Exit codes: `0` completed, `1` the adapter could not be analysed (the verdict is still
`review`), `2` the `--fail-on` threshold was reached.

## `adaptersentry batch`

| Flag | Default | Meaning |
|---|---|---|
| `--input-dir DIR` / `--input-list FILE` | required (one of) | directory scanned recursively for `*.safetensors`, or a file with one path per line |
| `--output-dir DIR` | `./results` | results go to `DIR/<run_id>/` |
| `--run-id ID` | timestamp | reuse with `--resume` |
| `--workers N` | 4 | parallel workers |
| `--resume` | off | re-queue unfinished jobs of `--run-id` |
| `--force-rescan` | off | re-scan everything, ignoring finished jobs and the cache |
| `--no-cache` | off | do not read or write the result cache |
| `--cache-dir DIR` | `~/.adaptersentry/cache` | cache location |
| `--mode full\|fast` | `full` | as for `scan` |
| `--policy default\|strict` | `default` | as for `scan` |
| `--fail-on review\|block` | none | exit code 2 if any verdict reaches this action |
| `--backend mp\|ray` | `mp` | worker backend; `ray` needs `pip install "adaptersentry[ray]"` |
| `--ray-address ADDRESS` | local | existing Ray cluster; localhost or a trusted network only (Ray has no authentication) |

The job manifest is kept in `~/.adaptersentry/manifest.sqlite`. Details:
[scan-engine.md](../architecture/scan-engine.md).

## Policy

| Policy | Structural red flag | Intra-adapter outlier (robust z ≥ 8) |
|---|---|---|
| `default` | `review` | `review` |
| `strict` | `block` | `review` |

Without a calibrated reference profile, `default` never returns `block`. The thresholds
behind `review` are uncalibrated; treat a verdict as a triage signal.

## Python API

```python
from pathlib import Path
from adaptersentry import scan, load_scan_result

result = scan(
    Path("adapter_model.safetensors"),
    mode="full",              # or "fast"
    policy="default",         # or "strict"
    include_modules=False,    # per-module features in result.modules
    include_heads=False,      # per-head features (implies include_modules)
    full_paths=False,         # absolute path in provenance instead of the file name
    run_id=None,              # recorded in result.scan.run_id
    hf_repo_id=None,          # Hub provenance, if the file came from the Hub
    hf_revision=None,
)
print(result.verdict.action)

again = load_scan_result(result.model_dump(mode="json"))   # validate a stored document
```

## `adapter_config.json`

Place it next to the `.safetensors` file, as PEFT saves it. It supplies `r`, `lora_alpha`,
`use_rslora`, `rank_pattern` / `alpha_pattern`, `target_modules` and `base_model_name_or_path`.
Without it the scale is assumed to be 1 (α = r), the confidence of the verdict is lowered, and
the result says so. Pattern keys that contain regex metacharacters are not evaluated and are
reported as unresolved.

## Threads

The CLI sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` unless
they are already set: the analysis runs many small matrix products, where BLAS threads only
add overhead. When calling `scan()` from your own code, set them before importing numpy for
the same effect.
