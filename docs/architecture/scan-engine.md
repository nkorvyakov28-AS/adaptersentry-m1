# Batch Scan Engine

`adaptersentry batch` scans a corpus of adapters with a worker pool, a content-addressed
result cache and a resumable job manifest. Each adapter is scanned by the same
`adaptersentry.scan()` used for single files, so batch and single-file results are identical.

```
cli/batch.py ──► engine/orchestrator.py (mp)  or  engine/orchestrator_ray.py (Ray)
                     │  build_manifest(): resolve paths, dedup, write manifest rows
                     ▼
                 engine/worker.py  worker_main(req, config_hash, cache_root)
                     │  1. identity (content hash) → cache lookup
                     │  2. on miss: scanner.scan(path, mode, policy, run_id, hf_*)
                     │  returns (ScanResult, from_cache); never raises
                     ▼
                 engine/result_sink.py  atomic JSON write + JSONL append
                 engine/cache.py        cache write
                 engine/manifest.py     state → persisted
```

Workers only compute. All manifest and cache writes happen in the orchestrator process.

## Components

| Component | File | Role |
|---|---|---|
| Identity | `engine/identity.py` | `content_hash` = SHA-256 of the file (streamed), `header_hash` = SHA-256 of the safetensors header, `logical_id` from Hub repo/revision/file name or the resolved path |
| Manifest | `engine/manifest.py` | SQLite (WAL) job table per `run_id`; duplicate paths are deduplicated. Terminal states: `persisted`, `cached_hit`, `skipped_duplicate`, `failed` |
| Cache | `engine/cache.py` | Content-addressed store under `~/.adaptersentry/cache` (index in SQLite, gzip-compressed results in `objects/`) |
| Result sink | `engine/result_sink.py` | Atomic writes (temp file → fsync → rename), JSONL audit log, idempotent rewrites |
| Worker | `engine/worker.py` | Cache lookup, then `scan()`; any crash becomes a fail-closed `failed` result |
| Orchestrators | `engine/orchestrator.py`, `engine/orchestrator_ray.py` | Multiprocessing (`spawn`) pool or Ray actor pool |

## Cache

- **Key:** `(content_hash, config_hash)`, where `config_hash = scanner.config_hash(mode, policy)`
  covers the schema version, mode, policy and intra-adapter thresholds. Changing any of them
  invalidates old entries.
- **Integrity:** each stored object is re-hashed on read and compared with the index; a
  mismatch deletes the entry and the adapter is re-scanned. Entries written by another
  AdapterSentry version are not served.
- Disable with `--no-cache`; relocate with `--cache-dir DIR`.

## Output layout

```
<output-dir>/<run_id>/
  <scan_id>.json        one ScanResult 2.0.0 per adapter (file name = scan_id hex)
  run.jsonl             append-only log, one ScanResult per line
  run_summary.json      counts by status (ok/degraded/failed/cached) and by verdict action
```

`scan_id = sha256(content_hash : config_hash : schema_version)`, so the same file scanned with
the same settings always lands in the same file.

## Resume

```bash
adaptersentry batch --input-dir adapters/ --run-id nightly --workers 8
# interrupted …
adaptersentry batch --input-dir adapters/ --run-id nightly --resume
```

`--resume` resets non-terminal jobs of that run to queued; finished jobs are not re-run.
`--force-rescan` re-scans everything and bypasses the cache lookup.

## Backends

| | `--backend mp` (default) | `--backend ray` |
|---|---|---|
| Install | built in | `pip install "adaptersentry[ray]"` |
| Workers | persistent `multiprocessing` pool, `spawn` context | Ray actors (`num_cpus=1`, restarted up to 3 times) |
| A worker killed by the OS | may stall the pool | the actor is replaced; in-flight job is marked failed |
| Scope | one machine | local Ray, or an existing cluster via `--ray-address` |

Ray has no authentication of its own: use `--ray-address` only for a cluster on localhost or
a trusted private network.

## Threads

The analysis multiplies many small matrices, where multithreaded BLAS only adds overhead.
The CLI sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` unless
they are already set, and pool workers and Ray actors do the same. Parallelism comes from
`--workers`.

## Exit codes

`0` all jobs finished; `1` at least one adapter could not be analysed (or no input was
found); `2` a verdict reached the `--fail-on` threshold (`review` or `block`).
