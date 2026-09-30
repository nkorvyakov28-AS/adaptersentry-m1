# AdapterSentry HuggingFace Hub Benchmark

`adaptersentry-bench` discovers public LoRA adapter repositories on HuggingFace Hub,
downloads only `adapter_model.safetensors` (and `adapter_config.json` when present),
runs the AdapterSentry M1 scanner (`adaptersentry.scanner.scan`, ScanResult 2.0.0) on
each, and produces four output files per run.

## Important framing

**This is an observational benchmark, not a malware classifier.**
No labeled ground truth exists for the public Hub adapter population.
A `review` verdict or a high intra-adapter robust z flags an adapter as an
*investigation candidate*; it does not confirm malicious intent or content. The
intra-adapter thresholds are uncalibrated, and the reported distributions describe
scanner behaviour on this population, not detection accuracy. Use terms like "anomalous", "suspicious", or
"prioritised for review" — not "malicious" or "backdoored" — when interpreting results.

---

## Quick start

```bash
# Scan 500 adapters (default)
adaptersentry-bench --limit 500 --output-dir output/hf_benchmark_500

# Scan 1000 adapters with conservative rate limiting
adaptersentry-bench --limit 1000 --output-dir output/hf_benchmark_1000 --sleep-seconds 1.0

# Resume a stopped run
adaptersentry-bench --limit 500 --output-dir output/hf_benchmark_500 --resume

# Restrict to adapters ≤ 100 MB, minimum 100 HF downloads
adaptersentry-bench --limit 500 --max-download-mb 100 --min-downloads 100
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--limit N` | 500 | Target number of repos to scan |
| `--output-dir DIR` | `output/hf_benchmark_<limit>` | Directory for all outputs |
| `--resume` | off | Skip repos already present in `results.jsonl` |
| `--sleep-seconds S` | 0.5 | Sleep between HF API/download calls |
| `--max-download-mb MB` | 500 | Reject adapters larger than this |
| `--min-downloads N` | 0 | Minimum HF download count to include a repo |
| `--top-n N` | 20 | Top-N entries in aggregate suspicious lists |
| `--sample-seed SEED` | 42 | Recorded in `candidates.json` for reproducibility |
| `--workers N` | 1 | Parallel scan workers |
| `--local-only` | off | Rescan cached adapters without HF Hub calls |
| `--candidates-from FILE` | — | `candidates.json` to rescan with `--local-only` |
| `--mode {full,fast}` | full | M1 scan mode |
| `--verbose` | off | Enable DEBUG logging |

---

## Output files

All outputs are written to `--output-dir` (default: `output/hf_benchmark_<limit>/`).

| File | Format | Description |
|---|---|---|
| `candidates.json` | JSON | Discovered repos with HF metadata — written once, reused on resume |
| `results.jsonl` | JSONL | One JSON object per adapter, appended incrementally |
| `results.csv` | CSV | Flat summary, one row per adapter — sortable in Excel or pandas |
| `aggregate.json` | JSON | Aggregate statistics, distributions, percentiles, top-suspicious lists |
| `report.md` | Markdown | Human-readable benchmark report with methodology and findings |
| `adapters/` | directory | Cached downloaded `adapter_model.safetensors` files |

### results.jsonl schema (one object per line)

```json
{
  "repo_id": "author/model-name",
  "scan_timestamp": "2026-04-27T19:56:01+00:00",
  "status": "success",
  "error_message": null,
  "skip_reason": null,
  "hf_downloads": 12345,
  "hf_tags": ["peft", "lora"],
  "adapter_size_bytes": 6304960,
  "scan_status": "ok",
  "action": "allow",
  "level": "LOW",
  "training_state": "trained",
  "max_robust_z": 2.7,
  "n_outlier_modules": 0,
  "reason_codes": ["NO_REFERENCE_PROFILE"],
  "n_findings": 0,
  "top_findings": [],
  "rank_declared": 8,
  "n_modules": 64,
  "error_type": null,
  "error_detail": null,
  "tensor_keys_sample": []
}
```

Field sources in ScanResult 2.0.0: `scan_status` = `status`, `action`/`level` =
`verdict.action`/`verdict.level`, `training_state` = `adapter.training_state`,
`max_robust_z`/`n_outlier_modules` = `anomaly.intra.*` (null when no intra-adapter
comparison was possible), `reason_codes` = `verdict.reasons[].code`, `top_findings` =
first five `"RULE_ID: title"` strings (≤ 120 chars), `rank_declared` =
`adapter.rank_declared`, `n_modules` = `coverage.n_modules`.

Possible `status` values: `success` (scan status ok or degraded), `unsupported_architecture`
(the scanner found no LoRA A/B pair), `analysis_failed`, `download_failed`, `size_exceeded`,
`not_cached`, `skipped`. Records written by the 1.x benchmark load, but their removed
fields (ensemble score, flags, …) are dropped.

### aggregate.json top-level keys

```
generated_at, framing, run_params, totals, failure_breakdown, failure_reason_counts,
scan_status_distribution, action_distribution, level_distribution,
training_state_distribution, reason_code_distribution,
max_robust_z_percentiles, max_robust_z_mean, n_with_intra_anomaly, counts,
top_suspicious_by_max_robust_z, top_suspicious_by_structural_codes
```

Structural reason codes used for `top_suspicious_by_structural_codes`:
`FULL_WEIGHT_REPLACEMENT`, `ROUTER_ADAPTED`, `RANK_MISMATCH`, `SIBLING_EXECUTABLE`.

---

## Discovery pipeline

1. `HfApi.list_models(filter="peft", sort="downloads", expand=["siblings", "sha", "downloads", "tags"])`
   — one batch call returns file names inline, avoiding per-repo `list_repo_files` round-trips.
   The commit `sha` is stored in `candidates.json`; downloads are pinned to it and it is
   recorded as `artifact.provenance.hf_revision` in the scan.
2. Repos without `adapter_model.safetensors` in their file list are discarded immediately.
3. For repos that pass the name filter, `list_repo_tree` is called to get the file size.
   Repos exceeding `--max-download-mb` are excluded from the candidate list.
4. The candidate list is written to `candidates.json` and reused on subsequent runs,
   making the candidate selection deterministic and reproducible.

Selection is biased toward popular adapters (sorted by download count) and covers only
single-file adapters. The Hub population is not labeled benign or malicious.

---

## Resume safety

`results.jsonl` is the single source of truth for resume state. On `--resume`:

1. All `repo_id` values already in `results.jsonl` are skipped (regardless of status).
2. All statuses (success, failure, skipped) count as processed — failed repos are not retried automatically.
3. The candidate list is loaded from the existing `candidates.json` (not re-queried).

To retry failed repos, delete their entries from `results.jsonl` before resuming.

---

## Running tests

The benchmark utility functions have unit tests that require no network access:

```bash
pytest tests/test_bench.py -v
```

---

## Methodological notes

- **No accuracy claims** — without labeled ground truth, precision/recall/F1 cannot be computed.
- **Threshold calibration** — the intra-adapter thresholds are uncalibrated and no reference profile is used; false-positive and false-negative rates at the Hub population scale are unknown.
- **Coverage bias** — popular repos are over-represented; niche, recently published, or low-download adapters are under-represented.
- **Static analysis only** — M1 inspects weight tensors; behavioural confirmation requires M2 (not yet implemented).
