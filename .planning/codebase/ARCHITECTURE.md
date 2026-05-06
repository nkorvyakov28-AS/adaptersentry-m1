# Architecture

**Analysis Date:** 2026-05-06

## Pattern

**Static analysis pipeline with multi-process worker pool.**

AdapterSentry M1 follows a pipeline architecture: untrusted `.safetensors` files flow through a chain of read-only inspection stages (parse → identify → feature extract → detect → score → report) without ever executing adapter weights. The orchestrator runs single-process; CPU-bound analysis runs in a `multiprocessing.Pool` of persistent workers using the `spawn` start method (no `fork`).

## Layers

```
CLI / API entry
    ↓
Orchestrator (single-process coordinator)
    ↓  imap_unordered, chunksize=1
Worker Pool (multiprocessing, spawn)
    ↓
Worker Pipeline
    ① ArtifactIdentityResolver  — content_hash, header_hash, logical_id
    ② CacheResolver             — check CacheStore before analysis
    ③ analyzer.scan()           — triggers full M1 analysis
         ├─ Parser              — safetensors header + tensor metadata
         ├─ FeatureExtractor    — typed per-layer feature families
         └─ EnsembleDetector    — weighted logistic-regression scorer
    ④ Assemble ScanResult       — typed public output contract
    ⑤ Assemble DebugReport      — extended per-layer stats (debug mode only)
    ↓
ResultSink → ManifestDB (SQLite) + optional JSON output
```

## Entry Points

| Entry point | File | Notes |
|---|---|---|
| `adaptersentry` CLI | `src/adaptersentry/cli/main.py` | argparse dispatcher to scan/batch subcommands |
| `adaptersentry.analyze()` | `src/adaptersentry/analyzer.py` | Legacy dict API (stable) |
| `adaptersentry.scan()` | `src/adaptersentry/analyzer.py` | Typed `AdapterReport` API (v1.0.0+) |
| `adaptersentry.scan_to_result()` | `src/adaptersentry/analyzer.py` | Bridge to `ScanResult` typed contract |
| `worker_main()` | `src/adaptersentry/engine/worker.py` | Per-adapter unit of work (picklable, module-level) |
| `Orchestrator.run()` | `src/adaptersentry/engine/orchestrator.py` | Batch coordinator |
| `OrchestratorRay` | `src/adaptersentry/engine/orchestrator_ray.py` | Ray actor pool variant |
| GitHub Action | `src/adaptersentry/integrations/github_action.py` | CI integration layer |

## Data Flow

```
adapter.safetensors (untrusted input)
    → Parser: header + tensor metadata extracted via safetensors.numpy (memory-mapped, read-only)
    → FeatureExtractor: per-layer TensorRecord + FeatureFamilyResult list (norm, distribution, entropy, outlier, spectral)
    → EnsembleDetector: weighted combination → sigmoid-normalised 0–100 score + majority-vote gate (≥2 detectors)
    → RiskScorer: risk_level string (LOW/MEDIUM/HIGH/CRITICAL)
    → ScanResult (schema_version="1.0.0"): public stable output consumed by CLI reporters and CI gates
    → DebugReport (debug-1.0.0): extends ScanResult with tensor_records (not part of stable contract)
```

## Key Abstractions

| Abstraction | Location | Purpose |
|---|---|---|
| `ScanResult` | `src/adaptersentry/engine/schemas/scan_result.py` | Stable public output contract; schema_version pinned |
| `DebugReport` | `src/adaptersentry/engine/schemas/scan_result.py` | Extended debug output; not stable contract |
| `FeatureExtractor` | `src/adaptersentry/engine/feature_extractor.py` | Typed per-layer feature pipeline; replaces `_run_analysis()` flat-dict path |
| `EnsembleDetector` | `src/adaptersentry/scoring/ensemble.py` | Weighted scorer with majority-vote gate |
| `RiskScorer` | `src/adaptersentry/scoring/risk_scorer.py` | Flag-based risk level classification |
| `AdapterArtifactIdentity` | `src/adaptersentry/engine/schemas/identity.py` | Content hash + header hash + logical_id |
| `CacheStore` | `src/adaptersentry/engine/cache.py` | SQLite-backed result cache; not thread-safe |
| `ManifestDB` | `src/adaptersentry/engine/manifest.py` | SQLite batch state tracker; not thread-safe |
| `AdapterScanRequest` | `src/adaptersentry/engine/schemas/requests.py` | Picklable unit of work passed to workers |

## Dual Pipeline Paths (Active Concern)

Two code paths co-exist during a migration from legacy to typed:

- **Legacy:** `_run_analysis()` in `src/adaptersentry/analyzer.py` returns a flat dict. `worker_main()` currently calls `analyzer.scan()` which internally uses this path.
- **Typed (new):** `FeatureExtractor.extract_layer()` + `EnsembleDetector.score_families()` returns typed `FeatureFamilyResult` objects. Used via `FeatureExtractor.families_from_record()` as a migration bridge.

New feature logic MUST go in `FeatureExtractor`/`EnsembleDetector`, not in `_run_analysis()`.

## Security Architecture

- No `eval()`, `exec()`, or `pickle` on adapter-controlled input — enforced project-wide
- Tensor bomb guard: tensors > 1 B elements rejected before allocation
- File paths resolved with `pathlib.Path.resolve()` before use
- Worker pool uses `spawn` (not `fork`) to avoid numpy/scipy fork-safety issues
- Missing or degraded metadata treated as security signal (not ignored)
- Workers never raise exceptions; all failures captured in `ScanResult.errors`

## Concurrency Model

- **Orchestrator:** single-process, controls all manifest/cache writes
- **Workers:** `multiprocessing.Pool` with `spawn` context; persistent for full batch (modules imported once via `_pool_initializer`)
- **Backpressure:** `imap_unordered(chunksize=1)` blocks input generator when all workers busy
- **Ray variant:** `OrchestratorRay` replaces the pool with Ray actor pool for distributed workloads
- **Thread safety:** `CacheStore` and `ManifestDB` are NOT thread-safe (single-writer orchestrator design)

---

*Architecture analysis: 2026-05-06*
