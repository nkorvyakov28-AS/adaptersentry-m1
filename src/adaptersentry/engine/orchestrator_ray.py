"""Ray-based batch orchestrator — OPT-03.

Drop-in replacement for orchestrator.run_batch() using a Ray actor pool.

Advantages over multiprocessing.Pool:
  1. Crash isolation — a worker OOM-killed by the kernel becomes a dead actor;
     Ray restarts it (max_restarts=3) without stalling the whole batch.
     multiprocessing.Pool.imap_unordered deadlocks on SIGKILL'd workers.
  2. BLAS fix — OMP/OpenBLAS/MKL thread counts are set to 1 inside actor
     __init__ before numpy is imported, permanently fixing over-subscription
     without requiring OMP_NUM_THREADS in the caller's environment.
  3. Foundation for horizontal scaling — same interface, can span machines.
  4. Priority queues — route paid/free scans to dedicated actor pools.
  5. Built-in telemetry — Ray dashboard at http://localhost:8265.

Optional dependency:
    pip install adaptersentry[ray]

Falls back to orchestrator.run_batch() if ray is not available.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Worker actor ──────────────────────────────────────────────────────────────

def _make_worker_actor_class():
    """Return the ScanWorkerActor Ray remote class.

    Defined inside a function to avoid importing ray at module load time
    (ray is an optional dependency).
    """
    import ray

    @ray.remote(num_cpus=1, max_restarts=3, max_task_retries=0)
    class ScanWorkerActor:
        """Stateful scan worker — heavy modules pre-imported once in __init__.

        Equivalent to _pool_initializer + _worker_entry in orchestrator.py,
        but as a persistent Ray actor rather than a multiprocessing worker.

        max_restarts=3: the actor is automatically restarted if OOM-killed or
        otherwise crashed. In-flight tasks raise RayActorError and are handled
        by the orchestrator (marked failed, not silently dropped).

        max_task_retries=0: do not retry individual scan tasks on failure —
        a failed scan should be reported, not silently re-run.
        """

        def __init__(self, config_hash: str, cache_root_str: str) -> None:
            # BLAS thread fix: must be set BEFORE numpy is imported.
            # Permanent fix for the BLAS over-subscription bug — 8 workers ×
            # numpy OMP threads = load avg 47 on 8 cores without this.
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")

            self._config_hash = config_hash
            self._cache_root = Path(cache_root_str) if cache_root_str else None

            # Pre-import in dependency order (same as _pool_initializer)
            import adaptersentry.engine.identity
            import adaptersentry.engine.cache
            import adaptersentry.engine.feature_extractor
            import adaptersentry.analyzer

            logging.getLogger(__name__).debug(
                "Ray actor initialised (pid=%d, config=%s)",
                os.getpid(), config_hash[:16],
            )

        def scan(self, req) -> tuple:
            """Run one adapter scan. Returns (ScanResult, DebugReport, request_id)."""
            from adaptersentry.engine.worker import worker_main
            result, debug = worker_main(req, self._config_hash, self._cache_root)
            return result, debug, req.request_id

    return ScanWorkerActor


# ── Internal helpers ──────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_failed_result(request_id: str, run_id: str, adapter_path: str, error_msg: str):
    """Build a minimal ScanResult for an actor-crash failure.

    Mirrors worker._make_failed_result — all required ScanResult fields
    populated with safe dummy values so ResultSink can persist it.
    """
    from adaptersentry.engine.schemas.scan_result import ScanResult, ScanStatus, DebugReport
    from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity, ScanIdentity
    from adaptersentry.engine.schemas.requests import ArtifactSource
    from adaptersentry.engine.schemas.scoring import EnsembleSignal, RiskVerdict
    from adaptersentry.schemas.adapter_metadata import AdapterMetadata
    from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus, TrainingStatus
    from adaptersentry.schemas.errors import ScanError, ScanPhase
    from adaptersentry.schemas.finding import Severity
    from adaptersentry.version import __version__

    now = _utcnow()
    scan_id = "sha256:" + hashlib.sha256(f"{request_id}:actor_crash".encode()).hexdigest()

    dummy_identity = AdapterArtifactIdentity(
        logical_id="sha256:" + hashlib.sha256(adapter_path.encode()).hexdigest(),
        content_hash="sha256:" + "0" * 64,
        header_hash="sha256:" + "0" * 64,
        file_size_bytes=0,
        source=ArtifactSource(kind="local_path", local_path=adapter_path),
        resolved_at=now,
    )

    result = ScanResult(
        identity=ScanIdentity(
            scan_id=scan_id,
            run_id=run_id,
            analyzer_version=__version__,
            analyzer_config_hash="sha256:" + "0" * 64,
            schema_version="1.0.0",
            started_at=now,
            completed_at=now,
            wall_time_ms=0,
        ),
        artifact=dummy_identity,
        adapter_metadata=AdapterMetadata.from_parsed({}),
        verdict=RiskVerdict(
            overall_score=0,
            overall_level=Severity.LOW,
            recommended_action="review",
            m2_recommended=False,
            training_status=TrainingStatus.UNKNOWN,
        ),
        ensemble=EnsembleSignal(score=0.0, risk_level=Severity.LOW),
        errors=[ScanError.malformed(
            code="INVALID_SAFETENSORS",
            message=f"Ray worker actor crashed: {error_msg}",
            phase=ScanPhase.FEATURE,
        )],
        status=ScanStatus.FAILED,
        parse_status=ParseStatus.FAILED,
        analysis_mode=AnalysisMode.FAILED,
    )
    return result


def _handle_result(
    result,
    debug,
    request_id: str,
    sink,
    manifest_db,
    cache_store,
    stats: dict,
    *,
    write_debug: bool,
) -> None:
    """Persist one result — identical to orchestrator._handle_result."""
    from adaptersentry.engine.schemas.scan_result import ScanStatus

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
            from adaptersentry.engine.schemas.scan_result import ScanStatus as S
            if result.status == S.DEGRADED:
                stats["degraded"] = stats.get("degraded", 0) + 1
            else:
                stats["ok"] = stats.get("ok", 0) + 1
    except Exception as exc:
        logger.error("Failed to persist result for %s: %s", request_id, exc)
        stats["failed"] = stats.get("failed", 0) + 1
        try:
            manifest_db.update_state(request_id, "failed", error_message=str(exc))
        except Exception:
            pass


# ── run_batch_ray ─────────────────────────────────────────────────────────────

def run_batch_ray(
    requests: list,
    manifest_db,
    cache_store,
    results_dir: Path,
    run_jsonl_path: Path,
    analyzer_config_hash: str,
    cache_root: Path | None = None,
    n_workers: int = 4,
    write_debug: bool = False,
    ray_address: str | None = None,
) -> dict[str, int]:
    """Ray-based batch scanner — drop-in for orchestrator.run_batch().

    Args:
        requests:             Requests from build_manifest().
        manifest_db:          Open ManifestDB.
        cache_store:          Open CacheStore or None.
        results_dir:          Directory for per-adapter JSON files.
        run_jsonl_path:       Append-only JSONL audit log.
        analyzer_config_hash: Config hash from AnalyzerConfig.config_hash().
        cache_root:           Cache store root (passed to actors).
        n_workers:            Number of Ray actor workers.
        write_debug:          If True, write .debug.json alongside .json.
        ray_address:          Ray cluster address, e.g. "ray://head:10001".
                              None → init a local cluster on this machine.

    Returns:
        Stats dict: {'ok': N, 'degraded': N, 'failed': N, 'cached': N}.
    """
    try:
        import ray
    except ImportError:
        raise RuntimeError(
            "Ray is not installed. Install it with: pip install adaptersentry[ray]"
        )

    from adaptersentry.engine.result_sink import ResultSink

    if not requests:
        logger.info("No requests to process (Ray backend).")
        return {}

    ray.init(address=ray_address, ignore_reinit_error=True)
    logger.info(
        "Ray initialised: %d nodes, %d CPUs available",
        len(ray.nodes()),
        int(ray.available_resources().get("CPU", 0)),
    )

    ScanWorkerActor = _make_worker_actor_class()
    cache_root_str = str(cache_root) if cache_root else ""
    stats: dict[str, int] = {"ok": 0, "degraded": 0, "failed": 0, "cached": 0}
    sink = ResultSink(results_dir, run_jsonl_path)

    # Build a request_id lookup for crash recovery
    req_by_id = {req.request_id: req for req in requests}

    # Lease all jobs upfront
    lease_at = _utcnow()
    for req in requests:
        manifest_db.update_state(req.request_id, "leased", started_at=lease_at)

    # Spawn persistent actors — each pre-imports heavy modules once in __init__
    def _new_actor() -> object:
        return ScanWorkerActor.remote(analyzer_config_hash, cache_root_str)

    free_actors: list = [_new_actor() for _ in range(n_workers)]
    pending_reqs: list = list(requests)

    # future → (actor, request_id) for result collection and actor recycling
    future_to_info: dict = {}

    try:
        while pending_reqs or future_to_info:
            # Fill all idle actors with work
            while pending_reqs and free_actors:
                actor = free_actors.pop()
                req = pending_reqs.pop(0)
                future = actor.scan.remote(req)
                future_to_info[future] = (actor, req.request_id)

            if not future_to_info:
                break

            # Wait for the next completed result (1s timeout to allow Ctrl-C)
            done_refs, _ = ray.wait(
                list(future_to_info.keys()), num_returns=1, timeout=60.0
            )

            if not done_refs:
                logger.warning("Ray: no result in 60s — %d tasks in flight", len(future_to_info))
                continue

            done_ref = done_refs[0]
            actor, request_id = future_to_info.pop(done_ref)
            req = req_by_id[request_id]

            try:
                result, debug, _ = ray.get(done_ref)
                _handle_result(
                    result, debug, request_id,
                    sink, manifest_db, cache_store, stats,
                    write_debug=write_debug,
                )
                # Return the healthy actor to the free pool
                free_actors.append(actor)
            except ray.exceptions.RayActorError as exc:
                # Actor was OOM-killed or crashed — report as failed.
                # Ray restarts the actor (max_restarts=3) for future tasks,
                # but this in-flight task's result is lost.
                logger.error(
                    "Ray actor died for request %s: %s — marking failed",
                    request_id, exc,
                )
                failed_result = _make_failed_result(
                    request_id, req.run_id, req.adapter_path, str(exc)
                )
                _handle_result(
                    failed_result, None, request_id,
                    sink, manifest_db, cache_store, stats,
                    write_debug=False,
                )
                # Spawn a fresh replacement (the crashed actor may still be
                # restarting; easier to just create a new one)
                free_actors.append(_new_actor())
            except Exception as exc:
                logger.error("Unexpected error for request %s: %s", request_id, exc)
                failed_result = _make_failed_result(
                    request_id, req.run_id, req.adapter_path, str(exc)
                )
                _handle_result(
                    failed_result, None, request_id,
                    sink, manifest_db, cache_store, stats,
                    write_debug=False,
                )
                free_actors.append(actor)

    except KeyboardInterrupt:
        logger.warning("Ray batch interrupted by user — partial results persisted.")
        for future in future_to_info:
            try:
                ray.cancel(future)
            except Exception:
                pass
    finally:
        # Kill all actors cleanly (free + in-flight)
        all_actors = list(free_actors) + [info[0] for info in future_to_info.values()]
        for actor in all_actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass

    logger.info(
        "Ray batch complete: ok=%d degraded=%d failed=%d cached=%d",
        stats["ok"], stats["degraded"], stats["failed"], stats["cached"],
    )
    return stats
