# CLI usage

```bash
pip install adaptersentry
adaptersentry --version
```

Both commands emit `ScanResult 2.0.0` ([field reference](../output-schema/scan-result.md)).
Keep `adapter_config.json` next to the adapter file: without it the scan assumes α = r and
says so in `verdict.confidence.limiting_factors`.

## `adaptersentry scan ADAPTER`

| Option | Default | Meaning |
|---|---|---|
| `--format text\|json\|full-json\|sarif` | `text` | `json`: ScanResult; `full-json`: plus per-module features; `sarif`: SARIF 2.1.0 |
| `--output FILE` | stdout | write the report to a file |
| `--mode full\|fast` | `full` | `fast` uses 64 instead of 256 rows for the kurtosis estimate; every other feature is exact in both |
| `--policy default\|strict` | `default` | `strict` turns structural red flags into `block` |
| `--fail-on review\|block` | off | exit 2 if the verdict action reaches this |
| `--include-modules` | off | add `modules[]` to JSON output |
| `--include-heads` | off | also add per-attention-head features |
| `--full-paths` | off | report the absolute path instead of the file name |
| `--no-color` | off | no ANSI colours (colours are also off when not writing to a terminal) |

| Exit code | Meaning |
|---|---|
| 0 | scan completed; below the `--fail-on` threshold |
| 1 | the adapter could not be analysed (`status: failed`; the verdict is still `review`) |
| 2 | the verdict action reached `--fail-on` |

```bash
# CI gate: fail the build unless the adapter is clean
adaptersentry scan adapter_model.safetensors --format json --output result.json --fail-on review

# GitHub code scanning
adaptersentry scan adapter_model.safetensors --format sarif --output adaptersentry.sarif

# Isolated environment: block anything with a structural red flag
adaptersentry scan adapter_model.safetensors --policy strict --fail-on block

# Per-module detail for an investigation
adaptersentry scan adapter_model.safetensors --format full-json --include-heads --output detail.json
```

In SARIF output every finding is a result with its layer/module as a logical location, and a
non-`allow` verdict is itself a result (rule `ADAPTER_VERDICT`).

## `adaptersentry batch`

Scans many adapters with a process pool (or Ray), a content-addressed cache and a resumable
manifest.

| Option | Default | Meaning |
|---|---|---|
| `--input-dir DIR` / `--input-list FILE` | — | all `*.safetensors` under DIR, or one path per line |
| `--output-dir DIR` | `results` | results go to `DIR/<run-id>/` |
| `--run-id ID` | timestamp | name of the run |
| `--workers N` | 4 | worker processes |
| `--mode`, `--policy` | `full`, `default` | as for `scan` |
| `--fail-on review\|block` | off | exit 2 if any adapter's verdict reaches this |
| `--resume` | off | continue an interrupted run |
| `--force-rescan` | off | ignore the manifest's completed state and the cache |
| `--no-cache`, `--cache-dir DIR` | cache in `~/.adaptersentry/cache` | result cache (keyed by file content + settings) |
| `--backend mp\|ray` | `mp` | Ray needs `pip install "adaptersentry[ray]"` |
| `--ray-address ADDR` | local | connect to an existing Ray cluster — **localhost or a trusted network only** (Ray has no authentication) |

Output of a run:

```
<output-dir>/<run-id>/
  <scan_id>.json        ScanResult 2.0.0 per adapter
  run.jsonl             every result, one per line
  run_summary.json      counts by status and by verdict action
```

Exit codes: 0 completed; 1 some adapters could not be analysed or the run was interrupted;
2 `--fail-on` reached.
