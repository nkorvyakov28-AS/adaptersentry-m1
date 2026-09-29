# Architecture Overview

AdapterSentry M1 is a static security scanner for LoRA adapters stored as `.safetensors`
files. It reads the adapter's weights without loading a base model and without running
inference, and returns one `ScanResult 2.0.0` per adapter with a verdict of `allow`,
`review` or `block`.

Static analysis is a first filter, not a proof of safety. Detection thresholds are not yet
calibrated (see [Limits](../../README.md#limits--read-before-relying-on-a-verdict)).

## Entry points

| Entry point | Code |
|---|---|
| `adaptersentry.scan(path, ...) -> ScanResult` | `src/adaptersentry/scanner.py` |
| `adaptersentry.load_scan_result(data) -> ScanResult` | `src/adaptersentry/schemas/result.py` |
| `adaptersentry scan ADAPTER` | `src/adaptersentry/cli/scan.py` |
| `adaptersentry batch --input-dir DIR` | `src/adaptersentry/cli/batch.py` → `engine/` |

`scan()` never raises: any failure becomes a structured result whose action is at least
`review` (fail-closed).

## Pipeline

```
file ─► identity ─► config ─► inventory ─► per module (streamed) ─► intra ─► structural ─► verdict
```

| Step | Module | What it does |
|---|---|---|
| Header guards | `parsers/safetensors.py` | Regular file, bounded file and header size, dtype allowlist (F32/F16/BF16), shape / byte-range / offset checks, ≤ 1 B elements per tensor, LoRA rank ≤ 1024 — all before any tensor is allocated |
| Identity | `engine/identity.py` | SHA-256 of the file and of the header; provenance (file name unless `full_paths=True`) |
| Config | `parsers/adapter_config.py` | Reads `adapter_config.json` next to the file (bounded, untrusted). Effective scale s = α/r, or α/√r with rsLoRA; without α, s = 1. `rank_pattern` / `alpha_pattern` keys are matched literally — regexes from the file are never compiled |
| Inventory | `parsers/adapter_file.py`, `parsers/names.py` | Classifies every tensor from the header: LoRA A/B pairs (layer, expert, module), full weights (`modules_to_save`), DoRA vectors, trainable-token deltas, and anything not analysable, with a reason. Nothing is dropped silently |
| Streaming | `parsers/adapter_file.py` | Loads one A/B pair at a time as float32 and releases it before the next; peak memory is bounded by the largest pair |
| Exact ΔW statistics | `features/lowrank_core.py` | Singular values, Frobenius / row / column norms of ΔW = s·B·A computed exactly through the r×r Gram matrices A·Aᵀ and Bᵀ·B; ΔW is never materialised. Entry kurtosis is estimated by importance sampling of rows |
| Module features | `features/spectral.py` | Shape features of ΔW (stable rank, participation ratio, spectral entropy, excess concentration, kurtosis, row/column concentration) plus two log-magnitude features; optional per attention head |
| Intra-adapter comparison | `detectors/intra_adapter.py` | Per module family (all `q_proj`, all `down_proj`, …): remove the depth trend with a Theil–Sen line, then robust z-scores (median/MAD). Families need ≥ 16 modules. Thresholds `OUTLIER_Z = 3.5`, `REVIEW_Z = 8.0` are uncalibrated |
| Structural findings | `detectors/structural.py` | Red flags: full `lm_head` / embedding / router matrices, LoRA on a MoE router, ranks that differ from the declared `r`, code / pickle / archive files next to the adapter. Informational: DoRA, unresolved scale patterns, config problems, target-module mismatch, trainable tokens |
| Verdict | `scoring/verdict.py` | One rule set; policy `default` or `strict` (below) |
| Result | `schemas/result.py` | `ScanResult 2.0.0` (pydantic); re-checks fail-closed invariants on construction |
| Output | `reporters/` | `text`, `json`, `full-json`, `sarif` (SARIF 2.1.0). Text output passes file-derived strings through `reporting/sanitize.py` |

### Verdict rules

1. Parsing failed, or part of the file was not analysed → at least `review`.
2. A module departs from its family with robust |z| ≥ 8 → `review`.
3. Structural red flag → `review`; `block` under `--policy strict`.
4. Otherwise → `allow`.

Without a calibrated reference profile the default policy never blocks, because the
false-positive rate of a block would be unknown. Reference profiles per model family are
planned; the result schema already has a place for them (`anomaly.reference`).

## Memory and speed

The scan holds one LoRA pair in memory at a time. On a synthetic Llama-3.3-70B-shaped adapter
(560 modules, r = 64, 1.66 GB bf16) a full scan took about 15 s with about +93 MB peak RSS
on an 8-CPU machine; streaming the file costs about +32 MB, versus about 3.3 GB to load the
same file with the 1.x loader.

## Security invariants

- No `pickle`, `eval` or `exec` on anything from the adapter; pickle-based formats are not
  read at all.
- Every size, dtype, shape and offset is validated from the header before allocation.
- Missing metadata and partial parsing are reported as signals and lower the verdict; they
  are never ignored.
- No base model is loaded and no inference is run.
- Strings from the file are sanitised before they reach a terminal.

## Further reading

- [scan-modes.md](scan-modes.md) — `full` vs `fast`
- [scan-engine.md](scan-engine.md) — batch engine: manifest, cache, workers
- [repo-layout.md](repo-layout.md) — directory layout
- [open-core-boundary.md](open-core-boundary.md) — public API and project scope
- [../output-schema/scan-result.md](../output-schema/scan-result.md) — result fields
