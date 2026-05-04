#!/usr/bin/env python3
"""
prepare_release.py — copy benchmark artifacts from output/ into the versioned releases/ layout.

Usage:
    python tools/prepare_release.py --tag v0.2.0 --module m1 \\
        --output-dir output/hf_benchmark_1000 --benchmark-id hf_1000_v1 \\
        --description "1000-adapter Hub scan" \\
        [--mode full|local-only] [--limit 1000] [--workers 4] [--local-only]

The script:
  1. Creates releases/<tag>/<module>/benchmark/ if it doesn't exist.
  2. Copies aggregate.json  → releases/<tag>/<module>/benchmark/<id>_aggregate.json
  3. Copies report.md       → releases/<tag>/<module>/benchmark/<id>_report.md
  4. Creates or updates     releases/<tag>/<module>/benchmark/META.json
     with a new benchmark entry (idempotent — updates existing entry if id already present).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_generated_at(aggregate_path: Path) -> str:
    try:
        data = json.loads(aggregate_path.read_text())
        return data.get("generated_at", datetime.now(timezone.utc).isoformat())
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a versioned release benchmark artifact.")
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.2.0")
    parser.add_argument("--module", required=True, choices=["m1", "m2", "m3", "m4"],
                        help="Module name")
    parser.add_argument("--output-dir", required=True,
                        help="Source output directory containing aggregate.json and report.md")
    parser.add_argument("--benchmark-id", required=True,
                        help="Unique benchmark id, e.g. hf_1000_v1")
    parser.add_argument("--description", default="",
                        help="Human-readable description for META.json")
    parser.add_argument("--mode", default="full", help="Run mode: full or local-only")
    parser.add_argument("--limit", type=int, default=500, help="Adapter limit used in the run")
    parser.add_argument("--workers", type=int, default=1, help="Worker count used in the run")
    parser.add_argument("--local-only", action="store_true",
                        help="Set local_only=true in META.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src = Path(args.output_dir)
    if not src.is_absolute():
        src = repo_root / src

    aggregate_src = src / "aggregate.json"
    report_src = src / "report.md"
    for required in (aggregate_src, report_src):
        if not required.exists():
            print(f"ERROR: source file not found: {required}", file=sys.stderr)
            return 1

    dest_dir = repo_root / "releases" / args.tag / args.module / "benchmark"
    dest_dir.mkdir(parents=True, exist_ok=True)

    agg_dst = dest_dir / f"{args.benchmark_id}_aggregate.json"
    rpt_dst = dest_dir / f"{args.benchmark_id}_report.md"
    agg_dst.write_bytes(aggregate_src.read_bytes())
    rpt_dst.write_bytes(report_src.read_bytes())
    print(f"Copied {aggregate_src} -> {agg_dst}")
    print(f"Copied {report_src} -> {rpt_dst}")

    meta_path = dest_dir / "META.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {
            "release": args.tag,
            "module": f"M{args.module[1]} Static Analyzer",
            "benchmarks": [],
        }

    entry = {
        "id": args.benchmark_id,
        "description": args.description,
        "source_output_dir": str(src.relative_to(repo_root)),
        "mode": args.mode,
        "limit": args.limit,
        "workers": args.workers,
        "local_only": args.local_only,
        "git_commit": git_head(),
        "generated_at": load_generated_at(aggregate_src),
    }

    existing_ids = [b["id"] for b in meta["benchmarks"]]
    if args.benchmark_id in existing_ids:
        idx = existing_ids.index(args.benchmark_id)
        meta["benchmarks"][idx] = entry
        print(f"Updated existing META.json entry for {args.benchmark_id}")
    else:
        meta["benchmarks"].append(entry)
        print(f"Appended new META.json entry for {args.benchmark_id}")

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"META.json written to {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
