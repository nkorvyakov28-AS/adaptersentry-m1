# Scan Engine — Architecture

> v1.0.0. The scan engine enables large-scale batch scanning with resumable execution,
> content-addressed caching, typed schema-stable output, a Ray actor pool backend (OPT-03),
> and Rust hot-path extensions (OPT-04).

## Overview

The scan engine wraps the M1 analyzer in a production-grade batch execution pipeline.
A single-node orchestrator manages a multiprocessing worker pool, a SQLite manifest
for job state, a content-addressed cache, and atomic result persistence.

```
adaptersentry batch --input-dir ./adapters --workers 4 --mode fast
        │
        ▼
┌─────────────────────┐
│  cli/batch.py       │  argument parsing, run_id generation, directory setup
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  engine/            │
│  orchestrator.py    │  build_manifest() — resolve paths, dedup, write manifest rows
│                     │  run_batch()      — Pool(spawn, n_workers), imap_unordered
│                     │  resume_after_failure() — reset non-terminal jobs
└────────┬────────────┘
         │  per-adapter request
         ▼
┌─────────────────────┐
│  engine/worker.py   │  worker_main() — runs inside each worker process
│                     │
│  Phase ①  Identity  │  ArtifactIdentityResolver → content_hash (BLAKE3/SHA256)
│  Phase ②  Cache     │  CacheStore.lookup() → ScanResult (cache hit) or miss
│  Phase ③  Analysis  │  analyzer.scan(fast=) → AdapterReport
│  Phase ④  Assemble  │  AdapterReport → ScanResult + DebugReport
└────────┬────────────┘
         │  (ScanResult, DebugReport, request_id)
         ▼
┌─────────────────────┐
│  engine/            │
│  result_sink.py     │  write() — atomic rename+fsync, JSONL append, idempotency check
│  cache.py           │  write_entry() — object store write
│  manifest.py        │  update_state() → persisted
└─────────────────────┘
```

## Components

### ArtifactIdentityResolver (`engine/identity.py`)

Computes a content-addressed identity for each adapter file:

- `content_hash` — BLAKE3 (SHA256 fallback) of full file bytes
- `header_hash`  — BLAKE3 of safetensors header only (fast re-identification)
- `logical_id`   — SHA256(content_hash) — stable short identifier

`scan_id` is deterministic: `sha256(content_hash + ':' + analyzer_config_hash + ':' + schema_version)`.
Same file + same config always produces the same `scan_id`.

### ManifestDB (`engine/manifest.py`)

SQLite (WAL mode) database tracking per-adapter job state:

```
pending → queued → leased → persisted
                          → failed
```

- `resume_after_failure()` resets `leased` and `failed` jobs back to `queued`
- Lease expiry (120s) prevents orphaned jobs from blocking a resume
- Duplicate paths are deduplicated by canonical resolved path

### CacheStore (`engine/cache.py`)

Content-addressed object store:

```
cache/
  objects/
    ab/          ← first 2 hex chars of content_hash
      cd1234…    ← full content_hash — stores ScanResult JSON bytes
  index.sqlite   ← lookup index: (content_hash, config_hash) → entry
```

Cache key = `(content_hash, analyzer_config_hash)`. A config change
(including `scan_mode`) automatically invalidates stale entries.
BLAKE3 integrity check on every read — tampered objects treated as miss.

### ResultSink (`engine/result_sink.py`)

Atomic persistence:

1. Write to `<path>.tmp`
2. `fsync` the file
3. `os.rename()` (atomic on POSIX) to final path
4. Update manifest state

A process kill between steps 1–3 leaves the old file intact — never a partial write.
JSONL append to `run.jsonl` provides an immutable audit trail of all results.

### AnalyzerConfig (`engine/config.py`)

Canonical representation of the active detector configuration. Every field that
affects analysis output is included — weights, thresholds, enabled families,
`scan_mode`, schema version, tool version.

`config_hash()` = SHA256 of canonical JSON. Any config change produces a new hash,
automatically invalidating cached results from the previous configuration.

## Ray Actor Pool Backend (OPT-03, v0.4.1)

The default backend (`--backend mp`) uses `multiprocessing.Pool(spawn)`. The optional
Ray backend (`--backend ray`) replaces it with a persistent actor pool:

```
adaptersentry batch --backend ray --workers 8 --mode full --input-dir ./adapters
        │
        ▼
┌─────────────────────┐
│  orchestrator_ray.py│  run_batch_ray() — drop-in for run_batch()
│                     │  ScanWorkerActor — @ray.remote, max_restarts=3
│                     │  ray.wait() loop — future→actor tracking
└────────┬────────────┘
         │  actor.scan.remote(req) per adapter
         ▼
┌─────────────────────┐
│  ScanWorkerActor    │  __init__(): sets OMP/BLAS threads=1, pre-imports modules
│  (Ray actor)        │  scan():     calls worker_main() — same as mp path
└─────────────────────┘
```

**Why Ray over multiprocessing:**

| Property | `--backend mp` | `--backend ray` |
|----------|---------------|----------------|
| Worker restart on OOM kill | No — `imap_unordered` deadlocks | Yes — `max_restarts=3`, replaced automatically |
| BLAS thread fix | `_pool_initializer()` sets before import | `__init__()` sets before import |
| Horizontal scaling | Single machine | Multi-node (pass `--ray-address`) |
| Priority queues | Not supported | Actors can be partitioned by tenant |
| Dashboard | None | http://localhost:8265 |
| Overhead | ~0ms/task | ~2ms/task dispatch |

**Throughput on 498 real HuggingFace adapters (8-CPU VPS):**

| Mode | Workers | Backend | Throughput | Wall time |
|------|---------|---------|-----------|-----------|
| fast | 8 | mp | 203.2/min | 2.5 min |
| fast | 8 | ray | **211.1/min** | **2.4 min** |
| full | 4 | mp | 22.1/min | 22.5 min |
| full | 8 | ray | **37.6/min** | **13.3 min** |

Ray full 8 workers: +70% throughput vs mp 4 workers; 0 OOM kills (peak 710 MB/worker).

**Install:** `pip install "adaptersentry[ray]"` (adds `ray[default]>=2.9.0`).

## Crash recovery

```bash
# First run (crashes at adapter 300/500)
adaptersentry batch --input-dir ./adapters --run-id my-run --workers 4

# Resume — re-queues jobs 301–500, skips 1–300
adaptersentry batch --input-dir ./adapters --run-id my-run --resume
```

`resume_after_failure()` resets all non-terminal rows (leased, failed) back to
queued and returns the count of reset jobs. Completed jobs (`persisted`) are never
re-processed.

## Output layout

```
results/<run_id>/
  adapter_0000.json          ← ScanResult (summary-json, stable contract)
  adapter_0000.debug.json    ← DebugReport (if --debug, not stable)
  run_summary.json           ← batch stats: ok/degraded/failed/cached counts
  run.jsonl                  ← append-only audit trail (one line per result)
```

## Worker isolation

Workers use `multiprocessing.spawn` (not fork) for safety with numpy, scipy,
and safetensors native extensions. Each worker process is independent — a crash
in one worker does not affect others. The orchestrator detects the crash when
the pool iterator raises and marks the job as failed in the manifest.

`MAX_TASKS_PER_CHILD = 50` — workers are recycled after 50 tasks to prevent
memory accumulation across many large adapters.

## Performance characteristics (2 CPU / 3.8GB RAM baseline)

| Scenario | Throughput | Notes |
|----------|-----------|-------|
| Cold, `--mode full`, 2 workers | ~1.2 adapters/min | Real 7B adapters, 36GB corpus |
| Cold, `--mode fast`, 2 workers | ~5–7 adapters/min | Estimated after OPT-02 |
| Warm cache | ~100+ adapters/min | Cache hit = identity check only |

On recommended hardware (8 CPU / 16GB RAM / NVMe):

| Scenario | Throughput | Notes |
|----------|-----------|-------|
| Cold, `--mode fast`, 7 workers | ~50–70 adapters/min | Estimated |
| Cold, `--mode full`, 7 workers | ~10–15 adapters/min | Estimated |
| Warm cache | ~500+ adapters/min | I/O bound on NVMe |

## Performance characteristics (8-CPU / 16 GB VPS, v0.4.1)

| Scenario | Backend | Workers | Throughput | Peak/worker |
|----------|---------|---------|-----------|-------------|
| Cold, `--mode fast` | mp | 8 | 203 adapters/min | 455 MB |
| Cold, `--mode fast` | ray | 8 | 211 adapters/min | 721 MB |
| Cold, `--mode full` | mp | 4 | 22 adapters/min | 473 MB |
| Cold, `--mode full` | ray | 8 | 38 adapters/min | 710 MB |
| Warm cache | mp | 8 | 500+ adapters/min | I/O bound |

## Rust Hot-Path Extensions (OPT-04, v1.0.0)

`adaptersentry-rs/` Rust crate built with PyO3 + maturin. Each function has a Python
fallback — the scanner works without Rust, just slower.

| Function | Speedup | Replaces |
|----------|---------|---------|
| `isolation_score_1d` | **334×** | sklearn IsolationForest (1D exact ECDF solution) |
| `tensor_stats_f32` | 2.4× | numpy multi-pass kurtosis + percentiles |
| `byte_entropy` | 4.5× | numpy bincount + log2 |
| `sign_stats` | 2× | numpy sum + log2 |

IsolationForest replacement: for 1D data the exact expected isolation depth is
`E[h(x)] = H(rank) + H(n-rank) - 1` (harmonic numbers). With 20 trees this is
approximated; the Rust version uses the exact formula — more accurate AND 334× faster.

**Impact on AlgoCore (168 layers, full mode):**

| Metric | Before OPT-04 | After OPT-04 |
|--------|--------------|-------------|
| IsolationForest | 6.7s (168 × 40ms) | 0.01s (168 × 0.06ms) |
| tensor_stats | 0.34s | 0.14s |
| byte_entropy | 0.07s | 0.016s |
| **Total adapter** | **~11s** | **5.9s** |

**Corpus benchmark (498 adapters, 8 workers, Ray + Rust, full mode): 68.8/min, 7.2 min.**

## Optimisation history

| Card | Description | Status |
|------|-------------|--------|
| OPT-01 | Persistent workers (`_pool_initializer`) | ✅ v0.3.0 |
| OPT-03 | Ray actor pool (`--backend ray`) | ✅ v0.4.1 |
| OPT-04 | Rust hot-path (`adaptersentry-rs`) | ✅ v1.0.0 |
