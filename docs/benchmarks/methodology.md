# Benchmark Methodology

## Purpose

The `benchmarks/` tooling runs AdapterSentry M1 at scale against public LoRA adapters
from HuggingFace Hub to characterize M1's output distribution and flag patterns.

**This is an observational benchmark, not a malware classifier.** No labeled ground
truth exists for the public Hub adapter population.  High ensemble scores flag adapters
as *investigation candidates*; they do not confirm malicious content.

## Candidate selection

- HuggingFace Hub queried with filter `peft`, sorted by download count (most popular first).
- Only repositories containing a single-file `adapter_model.safetensors` are included.
- Sharded multi-file adapters are excluded.
- `adapter_config.json` is fetched when present to extract declared LoRA rank.
- Base model weights are never downloaded.

## Scan pipeline

1. **Discovery**: `benchmarks.hub_scanner.load_or_discover_candidates()` — queries Hub API,
   filters by file presence and size, saves a `candidates.json` for reproducibility.
2. **Download**: adapter and `adapter_config.json`, pinned to the Hub commit `sha` found at
   discovery, so a re-run scans the same files.
3. **M1 analysis**: `adaptersentry.scanner.scan()` — a `ScanResult 2.0.0` per adapter. A failed
   scan whose error is "no LoRA A/B pair" is classified `unsupported_architecture`; other
   failures are `analysis_failed`.
4. **Reporting**: `benchmarks.report.write_aggregate()` — distributions of verdict action,
   level, reason codes and training state, `max_robust_z` percentiles, and top lists by
   intra-adapter z and by structural reason codes. These distributions describe the
   scanner's behaviour on public adapters; they are not a detection rate (intra-adapter
   thresholds are uncalibrated and the adapters are unlabelled).

## Output artifacts

All artifacts live under `output/` (gitignored) and are promoted to `releases/` as small
derived summaries:

| Artifact | Description |
|---|---|
| `results.jsonl` | One JSON line per adapter (raw, not committed) |
| `results.csv` | Flat CSV for spreadsheet analysis (not committed) |
| `aggregate.json` | Small aggregate statistics (committed to releases/) |
| `report.md` | Human-readable benchmark report (committed to releases/) |

## Limitations

- No labeled ground truth → precision, recall, F1 cannot be reported.
- Sample biased toward popular adapters (sorted by downloads).
- M1 thresholds calibrated on a small development set; false-positive/negative rates
  at scale are unknown.
- Private and gated repositories are excluded.
