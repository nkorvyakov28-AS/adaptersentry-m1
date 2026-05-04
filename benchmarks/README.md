# AdapterSentry HuggingFace Hub Benchmark

`adaptersentry-bench` discovers public LoRA adapter repositories on HuggingFace Hub,
downloads only `adapter_model.safetensors`, runs AdapterSentry M1 static analysis on
each, and produces four output files per run.

## Important framing

**This is an observational benchmark, not a malware classifier.**
No labeled ground truth exists for the public Hub adapter population.
High ensemble scores flag adapters as *investigation candidates*; they do not confirm
malicious intent or content. Use terms like "anomalous", "suspicious", or
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
  "training_status": "TRAINED",
  "overall_risk": 0,
  "risk_level": "LOW",
  "ensemble_score": 4.1,
  "ensemble_risk_level": "LOW",
  "false_positive_suppressed": 0,
  "n_flags": 0,
  "top_flags": [],
  "cross_layer_consistency": 1.0,
  "wasserstein_mean": 0.0046,
  "claimed_rank": 8,
  "n_layers": 2
}
```

Possible `status` values: `success`, `download_failed`, `analysis_failed`, `size_exceeded`, `skipped`.

### aggregate.json top-level keys

```
generated_at, framing, run_params, totals, failure_reason_counts,
risk_level_distribution, training_status_distribution,
ensemble_score_percentiles, ensemble_score_mean, counts,
top_suspicious_by_ensemble_score, top_suspicious_by_rule_score
```

---

## Discovery pipeline

1. `HfApi.list_models(filter="peft", sort="downloads", expand=["siblings"])` — one batch
   call returns file names inline, avoiding per-repo `list_repo_files` round-trips.
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
- **Threshold calibration** — M1 thresholds were calibrated on a small development set; false-positive and false-negative rates at the Hub population scale are unknown.
- **Coverage bias** — popular repos are over-represented; niche, recently published, or low-download adapters are under-represented.
- **Static analysis only** — M1 inspects weight tensors; behavioural confirmation requires M2 (not yet implemented).
