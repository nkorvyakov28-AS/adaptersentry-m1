"""batch subcommand — scan a corpus of LoRA adapters.

Scans multiple .safetensors files with a process-based worker pool,
incremental caching, and full resumability after crashes.

Exit codes:
  0  — batch completed; all jobs in terminal state
  1  — batch failed or was interrupted
  2  — any job produced findings at or above --fail-on threshold

Output:
  results/<run_id>/                    — per-adapter JSON files (summary-json)
  results/<run_id>/run.jsonl           — append-only audit log (all results)
  results/<run_id>/run_summary.json    — batch stats and top findings
  [results/<run_id>/*.debug.json]      — debug-json per adapter (if --debug)

Cache:
  ~/.adaptersentry/cache/              — content-addressed result cache
  ~/.adaptersentry/manifest.sqlite     — per-run state machine

Resume:
  adaptersentry batch --input-dir ... --run-id <id> --resume
  Resets non-terminal jobs and re-runs from where the batch stopped.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def build_parser(subparsers: Any) -> None:
    """Register the ``batch`` subcommand."""
    p = subparsers.add_parser(
        "batch",
        help="Scan a corpus of LoRA adapters (batch mode with caching and resume)",
        description=__doc__,
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        metavar="DIR",
        help="Directory to scan recursively for .safetensors files",
    )
    source.add_argument(
        "--input-list",
        type=Path,
        metavar="FILE",
        help="Text file with one adapter path per line",
    )
    p.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help=(
            "Batch run identifier. If omitted, auto-generated from timestamp. "
            "Use the same ID with --resume to continue a previous run."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel worker processes (default: 4, minimum: 1)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        metavar="DIR",
        help="Output directory for per-adapter JSON files (default: ./results)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Cache store root (default: ~/.adaptersentry/cache). "
            "Set to /dev/null to disable caching."
        ),
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching (always re-scan)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run: reset non-terminal jobs and continue",
    )
    p.add_argument(
        "--force-rescan",
        action="store_true",
        help="Re-scan all adapters, ignoring existing terminal manifest state",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Write .debug.json files with per-layer statistics alongside summary JSON",
    )
    p.add_argument(
        "--fail-on",
        choices=_SEVERITIES,
        default=None,
        metavar="SEVERITY",
        help="Exit with code 2 if any finding meets or exceeds SEVERITY",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=None,
        metavar="R",
        help="Declared LoRA rank r (applied to all adapters in the batch)",
    )
    p.add_argument(
        "--mode",
        choices=("full", "fast"),
        default="full",
        dest="scan_mode",
        help=(
            "Scan depth: full (default) — all detectors at full depth; "
            "fast — truncated SVD, sampling, no IsolationForest on large tensors. "
            "Use fast for high-throughput corpus screening."
        ),
    )
    p.add_argument(
        "--backend",
        choices=("mp", "ray"),
        default="mp",
        help=(
            "Worker backend: mp (default) — multiprocessing.Pool; "
            "ray — Ray actor pool (requires pip install adaptersentry[ray]). "
            "Ray adds crash isolation, BLAS thread fix, and horizontal scaling."
        ),
    )
    p.add_argument(
        "--ray-address",
        default=None,
        metavar="ADDRESS",
        help=(
            "Ray cluster address (e.g. ray://head-node:10001). "
            "Omit to start a local Ray cluster on this machine. "
            "Only used with --backend ray."
        ),
    )


def run(args: Any) -> int:
    """Execute the batch subcommand."""
    from adaptersentry.engine.config import AnalyzerConfig, ScanMode
    from adaptersentry.engine.manifest import ManifestDB
    from adaptersentry.engine.cache import CacheStore
    from adaptersentry.engine.orchestrator import (
        build_manifest, run_batch, resume_after_failure,
    )
    from adaptersentry.schemas.finding import Severity

    backend = getattr(args, "backend", "mp")

    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 1

    if getattr(args, "ray_address", None) and backend != "ray":
        logger.warning("--ray-address is ignored when --backend is not 'ray'")

    # ── Run ID ────────────────────────────────────────────────────────────────
    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info("Batch run: %s", run_id)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    input_paths = _collect_paths(args)
    if not input_paths:
        print("error: no .safetensors files found in the specified input", file=sys.stderr)
        return 1

    logger.info("Found %d adapter path(s)", len(input_paths))

    # ── Directories ───────────────────────────────────────────────────────────
    output_dir: Path = args.output_dir
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_jsonl = run_dir / "run.jsonl"

    # ── Analyzer config hash ──────────────────────────────────────────────────
    scan_mode = ScanMode(getattr(args, "scan_mode", "full"))
    config = AnalyzerConfig(scan_mode=scan_mode)
    config_hash = config.config_hash()
    logger.info("Analyzer config hash: %s  scan_mode=%s", config_hash, scan_mode.value)

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest_path = Path.home() / ".adaptersentry" / "manifest.sqlite"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with ManifestDB.open(manifest_path) as manifest_db:
        # ── Resume ────────────────────────────────────────────────────────────
        if args.resume:
            n_reset, n_terminal = resume_after_failure(run_id, manifest_db)
            print(f"Resume: {n_reset} job(s) reset, {n_terminal} already complete.")

        # ── Build manifest ────────────────────────────────────────────────────
        requests = build_manifest(
            input_paths=input_paths,
            run_id=run_id,
            manifest_db=manifest_db,
            force_rescan=args.force_rescan,
            scan_mode=scan_mode.value,
        )

        if not requests:
            stats = manifest_db.stats(run_id)
            print(
                f"All {sum(stats.values())} adapter(s) already in terminal state "
                f"for run {run_id}. Use --force-rescan to re-scan."
            )
            return 0

        print(f"Scanning {len(requests)} adapter(s) with {args.workers} worker(s)...")

        # ── Cache ─────────────────────────────────────────────────────────────
        cache_root: Path | None = None
        cache_store: CacheStore | None = None

        if not args.no_cache:
            if args.cache_dir:
                cache_root = args.cache_dir
            else:
                cache_root = Path.home() / ".adaptersentry" / "cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_store = CacheStore.open(cache_root)

        try:
            # ── Run batch ─────────────────────────────────────────────────────
            if backend == "ray":
                from adaptersentry.engine.orchestrator_ray import run_batch_ray
                stats = run_batch_ray(
                    requests=requests,
                    manifest_db=manifest_db,
                    cache_store=cache_store,
                    results_dir=run_dir,
                    run_jsonl_path=run_jsonl,
                    analyzer_config_hash=config_hash,
                    cache_root=cache_root,
                    n_workers=args.workers,
                    write_debug=args.debug,
                    ray_address=getattr(args, "ray_address", None),
                )
            else:
                stats = run_batch(
                    requests=requests,
                    manifest_db=manifest_db,
                    cache_store=cache_store,
                    results_dir=run_dir,
                    run_jsonl_path=run_jsonl,
                    analyzer_config_hash=config_hash,
                    cache_root=cache_root,
                    n_workers=args.workers,
                    write_debug=args.debug,
                )
        finally:
            if cache_store is not None:
                cache_store.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(stats.values())
    print(
        f"\nBatch complete: {total} adapter(s) scanned\n"
        f"  ok={stats.get('ok', 0)}  degraded={stats.get('degraded', 0)}  "
        f"failed={stats.get('failed', 0)}  cached={stats.get('cached', 0)}"
    )
    print(f"Results written to: {run_dir}")
    print(f"Audit log:          {run_jsonl}")

    _write_run_summary(run_dir, run_id, stats)

    # ── fail-on threshold ─────────────────────────────────────────────────────
    if args.fail_on and run_jsonl.exists():
        threshold = _SEVERITIES.index(args.fail_on)
        triggered = _check_fail_on(run_jsonl, threshold)
        if triggered:
            print(f"\nFINDINGS at or above {args.fail_on} detected. Exiting with code 2.")
            return 2

    if stats.get("failed", 0) > 0:
        return 1

    return 0


def _collect_paths(args: Any) -> list[Path]:
    """Collect adapter paths from --input-dir or --input-list."""
    paths: list[Path] = []
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            logger.error("--input-dir is not a directory: %s", d)
            return []
        paths = list(d.rglob("*.safetensors"))
    elif args.input_list:
        lf = Path(args.input_list)
        if not lf.exists():
            logger.error("--input-list file not found: %s", lf)
            return []
        lines = lf.read_text().splitlines()
        paths = list(dict.fromkeys(Path(line.strip()) for line in lines if line.strip()))
    return paths


def _write_run_summary(run_dir: Path, run_id: str, stats: dict) -> None:
    from datetime import datetime, timezone
    summary = {
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


_SEV_ORDER = {s: i for i, s in enumerate(_SEVERITIES)}


def _check_fail_on(run_jsonl: Path, threshold_idx: int) -> bool:
    """Return True if any result in the JSONL has a finding at or above threshold."""
    try:
        for line in run_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                for finding in obj.get("findings", []):
                    sev = finding.get("severity", "LOW")
                    if _SEV_ORDER.get(sev, 0) >= threshold_idx:
                        return True
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return False
