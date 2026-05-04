"""Orchestrator — batch scan coordinator.

Responsibilities:
  1. build_manifest()     — resolve paths, dedup, write manifest rows
  2. schedule_jobs()      — feed worker pool with queued requests
  3. _process_results()   — consume results from pool, persist via ResultSink
  4. resume_after_failure() — reset non-terminal rows before re-running

Architecture: single-process orchestrator, multiprocessing worker pool.
Workers communicate results back via the pool iterator; they do NOT write
to the manifest or cache directly. All manifest/cache writes go through
the orchestrator process.

The worker pool uses 'spawn' context (not 'fork') for safety with numpy,
scipy, and safetensors native extensions. Workers are persistent for the
full batch — heavy modules are imported once via _pool_initializer.

Backpressure: imap_unordered with chunksize=1 gives natural backpressure.
When all workers are busy, the input generator blocks — preventing unbounded
in-memory request accumulation for large corpora.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path

from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
from adaptersentry.engine.schemas.scan_result import ScanResult, ScanStatus, DebugReport

logger = logging.getLogger(__name__)

LEASE_DURATION_SECS = 60

# ── Per-worker globals — populated once by _pool_initializer ─────────────────
# Workers live for the full batch; heavy modules are imported once at startup.
_WORKER_CONFIG_HASH: str = ""
_WORKER_CACHE_ROOT: Path | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


# ── Worker pool initializer — runs once per worker process ───────────────────

def _pool_initializer(analyzer_config_hash: str, cache_root_str: str) -> None:
    """Run once at worker startup — pre-import heavy modules and store config.

    All imports that would otherwise happen on every task (numpy, scipy,
    sklearn, safetensors, the full analyzer stack) are resolved here and
    cached in sys.modules. Subsequent calls in the same worker process are
    no-ops at the module level.
    """
    global _WORKER_CONFIG_HASH, _WORKER_CACHE_ROOT

    # BLAS thread cap — must be set BEFORE numpy is imported.
    # Without this, each of N workers spawns its own OMP thread pool:
    # 8 workers × default OMP threads = load avg 47+ on an 8-core VPS.
    # os.environ.setdefault leaves the caller's env intact if already set.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    _WORKER_CONFIG_HASH = analyzer_config_hash
    _WORKER_CACHE_ROOT = Path(cache_root_str) if cache_root_str else None

    # Pre-import in dependency order — heavier last so errors surface early
    import adaptersentry.engine.identity          # BLAKE3, path resolution
    import adaptersentry.engine.cache             # CacheStore
    import adaptersentry.engine.feature_extractor # FeatureExtractor
    import adaptersentry.analyzer                 # numpy + scipy + sklearn + safetensors

    logging.getLogger(__name__).debug(
        "Worker initialised (pid=%d, config=%s)", os.getpid(), analyzer_config_hash[:16]
    )


# ── Module-level worker entry point (must be picklable) ───────────────────────

def _worker_entry(req: AdapterScanRequest) -> tuple[ScanResult, DebugReport, str]:
    """Top-level worker function — dispatched by multiprocessing.Pool.

    Config hash and cache root come from module globals set by _pool_initializer;
    they are not re-pickled on every task dispatch.

    Returns (ScanResult, DebugReport, request_id).
    Never raises — all failures are captured in ScanResult.
    """
    from adaptersentry.engine.worker import worker_main
    result, debug = worker_main(req, _WORKER_CONFIG_HASH, _WORKER_CACHE_ROOT)
    return result, debug, req.request_id


# ── build_manifest ────────────────────────────────────────────────────────────

def build_manifest(
    input_paths: list[Path],
    run_id: str,
    manifest_db: "ManifestDB",  # noqa: F821 (type-checking import below)
    enabled_families: list[str] | None = None,
    force_rescan: bool = False,
    scan_mode: str = "full",
) -> list[AdapterScanRequest]:
    """Resolve, deduplicate, and register input paths in the manifest.

    Returns the list of AdapterScanRequest objects ready to be dispatched.
    Paths that fail to resolve or have unsupported extensions are silently
    skipped with a warning log.

    Args:
        input_paths:      Raw paths from CLI or directory glob (may be unresolved).
        run_id:           Batch run identifier.
        manifest_db:      Open ManifestDB instance.
        enabled_families: Feature families to enable (default: norm+distribution+entropy+outlier+spectral).
        force_rescan:     If True, ignore terminal manifest state and re-queue everything.
    """
    from adaptersentry.engine.manifest import TERMINAL_STATES

    if enabled_families is None:
        enabled_families = ["norm", "distribution", "entropy", "outlier", "spectral"]

    seen_canonical: set[str] = set()
    requests: list[AdapterScanRequest] = []

    for raw_path in input_paths:
        try:
            canonical = raw_path.resolve(strict=True)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.warning("Skipping unresolvable path %s: %s", raw_path, exc)
            continue

        if canonical.suffix not in {".safetensors", ".bin"}:
            logger.debug("Skipping non-adapter file: %s", canonical)
            continue

        canonical_str = str(canonical)
        if canonical_str in seen_canonical:
            logger.debug("Deduplicating path: %s", canonical_str)
            continue
        seen_canonical.add(canonical_str)

        request_id = "sha256:" + hashlib.sha256(
            f"{canonical_str}:{run_id}".encode()
        ).hexdigest()

        # Skip already-terminal jobs unless force_rescan
        existing = manifest_db.get(request_id)
        if existing and existing.state in TERMINAL_STATES and not force_rescan:
            continue

        source = ArtifactSource(kind="local_path", local_path=canonical_str)
        req = AdapterScanRequest(
            request_id=request_id,
            run_id=run_id,
            adapter_path=canonical_str,
            source=source,
            enabled_families=enabled_families,
            scan_mode=scan_mode,
            force_rescan=force_rescan,
            submitted_at=_utcnow(),
        )

        manifest_db.upsert(
            request_id=request_id,
            run_id=run_id,
            adapter_path=canonical_str,
            request_json=req.model_dump_json(),
            state="queued",
            submitted_at=req.submitted_at,
        )
        requests.append(req)

    logger.info(
        "Manifest: %d jobs queued for run=%s (%d paths skipped)",
        len(requests), run_id, len(input_paths) - len(requests),
    )
    return requests


# ── run_batch ─────────────────────────────────────────────────────────────────

def run_batch(
    requests: list[AdapterScanRequest],
    manifest_db: "ManifestDB",  # noqa: F821
    cache_store: "CacheStore | None",  # noqa: F821
    results_dir: Path,
    run_jsonl_path: Path,
    analyzer_config_hash: str,
    cache_root: Path | None = None,
    n_workers: int = 4,
    write_debug: bool = False,
) -> dict[str, int]:
    """Dispatch requests to a worker pool and persist results.

    Returns a stats dict: {'ok': N, 'degraded': N, 'failed': N, 'cached': N}.

    Args:
        requests:             Requests from build_manifest().
        manifest_db:          Open ManifestDB.
        cache_store:          Open CacheStore or None.
        results_dir:          Directory to write per-adapter JSON files.
        run_jsonl_path:       Path to the batch JSONL audit log.
        analyzer_config_hash: Config hash from AnalyzerConfig.config_hash().
        cache_root:           Cache store root (passed to workers as a string).
        n_workers:            Number of worker processes.
        write_debug:          If True, write .debug.json alongside .json files.
    """
    from adaptersentry.engine.result_sink import ResultSink

    if not requests:
        logger.info("No requests to process.")
        return {}

    cache_root_str = str(cache_root) if cache_root else ""
    stats: dict[str, int] = {"ok": 0, "degraded": 0, "failed": 0, "cached": 0}

    ctx = multiprocessing.get_context("spawn")
    sink = ResultSink(results_dir, run_jsonl_path)

    with ctx.Pool(
        processes=n_workers,
        initializer=_pool_initializer,
        initargs=(analyzer_config_hash, cache_root_str),
        # maxtasksperchild=None — workers persist for the full batch;
        # heavy modules (numpy/scipy/sklearn) are imported once at startup.
    ) as pool:
        # Mark all as leased before dispatching
        lease_at = _utcnow()
        for req in requests:
            manifest_db.update_state(req.request_id, "leased", started_at=lease_at)

        try:
            for result, debug, request_id in pool.imap_unordered(
                _worker_entry, requests, chunksize=1
            ):
                _handle_result(
                    result, debug, request_id,
                    sink, manifest_db, cache_store, stats,
                    write_debug=write_debug,
                )
        except KeyboardInterrupt:
            logger.warning("Batch interrupted by user — partial results persisted.")
            pool.terminate()
            pool.join()
        except Exception as exc:
            logger.error("Worker pool error: %s", exc, exc_info=True)
            pool.terminate()
            pool.join()

    logger.info(
        "Batch complete: ok=%d degraded=%d failed=%d cached=%d",
        stats["ok"], stats["degraded"], stats["failed"], stats["cached"],
    )
    return stats


def _handle_result(
    result: ScanResult,
    debug: DebugReport,
    request_id: str,
    sink: "ResultSink",
    manifest_db: "ManifestDB",
    cache_store: "CacheStore | None",
    stats: dict[str, int],
    *,
    write_debug: bool,
) -> None:
    """Persist one result and update stats."""
    try:
        if result.status == ScanStatus.CACHED:
            sink.write_cached_hit(result, manifest_db, request_id)
            stats["cached"] = stats.get("cached", 0) + 1
        elif result.status == ScanStatus.FAILED:
            err_msg = result.errors[0].message if result.errors else "unknown error"
            sink.write_failed(result, manifest_db, request_id, err_msg)
            stats["failed"] = stats.get("failed", 0) + 1
        else:
            sink.write(result, debug, manifest_db, cache_store, request_id, write_debug=write_debug)
            if result.status == ScanStatus.DEGRADED:
                stats["degraded"] = stats.get("degraded", 0) + 1
            else:
                stats["ok"] = stats.get("ok", 0) + 1
    except Exception as exc:
        logger.error("Failed to persist result for request %s: %s", request_id, exc)
        stats["failed"] = stats.get("failed", 0) + 1
        try:
            manifest_db.update_state(request_id, "failed", error_message=str(exc))
        except Exception:
            pass


# ── resume_after_failure ──────────────────────────────────────────────────────

def resume_after_failure(run_id: str, manifest_db: "ManifestDB") -> tuple[int, int]:
    """Reset non-terminal jobs to queued so a batch can be restarted.

    Returns (n_reset, n_terminal).
    """
    n_reset = manifest_db.reset_non_terminal(run_id)
    stats = manifest_db.stats(run_id)
    n_terminal = sum(
        count for state, count in stats.items()
        if state in {"persisted", "cached_hit", "skipped_duplicate", "failed"}
    )
    logger.info(
        "Resume: %d jobs reset to queued, %d already terminal for run=%s",
        n_reset, n_terminal, run_id,
    )
    return n_reset, n_terminal
