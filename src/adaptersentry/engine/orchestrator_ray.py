"""Ray-based batch orchestrator.

Drop-in replacement for orchestrator.run_batch() using a Ray actor pool.

Advantages over multiprocessing.Pool:
  1. Crash isolation — a worker OOM-killed by the kernel becomes a dead actor;
     Ray restarts it (max_restarts=3) without stalling the whole batch.
     multiprocessing.Pool.imap_unordered deadlocks on SIGKILL'd workers.
  2. BLAS fix — OMP/OpenBLAS/MKL thread counts are set to 1 inside actor
     __init__ before numpy is imported, permanently fixing over-subscription
     without requiring OMP_NUM_THREADS in the caller's environment.
  3. Horizontal scaling — same interface can span machines of a trusted cluster
     (Ray has no authentication; never expose it beyond localhost or a trusted network).

Optional dependency:
    pip install adaptersentry[ray]

Falls back to orchestrator.run_batch() if ray is not available.
"""

from __future__ import annotations

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
            import adaptersentry.scanner

            logging.getLogger(__name__).debug(
                "Ray actor initialised (pid=%d, config=%s)",
                os.getpid(), config_hash[:16],
            )

        def scan(self, req) -> tuple:
            """Run one adapter scan. Returns (ScanResult, from_cache, request_id)."""
            from adaptersentry.engine.worker import worker_main
            result, from_cache = worker_main(req, self._config_hash, self._cache_root)
            return result, from_cache, req.request_id

    return ScanWorkerActor


# ── Internal helpers ──────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _actor_failed(req, error_msg: str):
    """Fail-closed result for a scan whose Ray actor crashed (e.g. OOM-killed)."""
    from adaptersentry.scanner import failed_result

    mode = "fast" if req.scan_mode == "fast" else "full"
    return failed_result(Path(req.adapter_path), f"Ray worker actor crashed: {error_msg}",
                         mode=mode, policy=req.policy, run_id=req.run_id)


# ── run_batch_ray ─────────────────────────────────────────────────────────────

from adaptersentry.engine.orchestrator import handle_result  # noqa: E402


def run_batch_ray(
    requests: list,
    manifest_db,
    cache_store,
    results_dir: Path,
    run_jsonl_path: Path,
    analyzer_config_hash: str,
    cache_root: Path | None = None,
    n_workers: int = 4,
    ray_address: str | None = None,
) -> dict[str, int]:
    """Ray-based batch scanner — drop-in for orchestrator.run_batch().

    Args:
        requests:             Requests from build_manifest().
        manifest_db:          Open ManifestDB.
        cache_store:          Open CacheStore or None.
        results_dir:          Directory for per-adapter JSON files.
        run_jsonl_path:       Append-only JSONL audit log.
        analyzer_config_hash: scanner.config_hash(mode, policy) of the batch.
        cache_root:           Cache store root (passed to actors).
        n_workers:            Number of Ray actor workers.
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
                result, from_cache, _ = ray.get(done_ref)
                handle_result(result, from_cache, request_id, sink, manifest_db, cache_store, stats)
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
                handle_result(
                    _actor_failed(req, str(exc)), False, request_id,
                    sink, manifest_db, cache_store, stats,
                )
                # Spawn a fresh replacement (the crashed actor may still be
                # restarting; easier to just create a new one)
                free_actors.append(_new_actor())
            except Exception as exc:
                logger.error("Unexpected error for request %s: %s", request_id, exc)
                handle_result(
                    _actor_failed(req, str(exc)), False, request_id,
                    sink, manifest_db, cache_store, stats,
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
