"""CI regression gate — compare current benchmark run against a committed baseline.

Fails (exit code 1) if p95 latency exceeds FACTOR * baseline p95 for any scenario.
Also checks throughput and error rate regressions.

Usage:
    python benchmarks/check_regression.py \\
        --current  benchmarks/results/current.json \\
        --baseline benchmarks/results/baseline.json \\
        --factor   2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data]  # single-scenario file
    return data


def check_regression(
    current: list[dict],
    baseline: list[dict],
    factor: float = 2.0,
    throughput_floor_factor: float = 0.5,
) -> tuple[bool, list[str]]:
    """Compare current metrics against baseline.

    A regression is detected when:
      - p95 latency > factor * baseline p95           (latency regression)
      - throughput < throughput_floor_factor * baseline throughput  (throughput regression)
      - error_rate > baseline error_rate + 0.05       (reliability regression)

    Args:
        current:                  List of metric dicts from the current run.
        baseline:                 List of metric dicts from the committed baseline.
        factor:                   Multiplier for p95 regression threshold (default 2.0).
        throughput_floor_factor:  Minimum fraction of baseline throughput (default 0.5).

    Returns:
        (passed: bool, failure_messages: list[str])
    """
    baseline_by_scenario = {m["scenario"]: m for m in baseline}
    failures: list[str] = []

    for cur in current:
        scenario = cur["scenario"]
        base = baseline_by_scenario.get(scenario)
        if base is None:
            # No baseline for this scenario — skip, don't fail
            continue

        base_p95 = base.get("latency_p95_ms", 0.0)
        cur_p95 = cur.get("latency_p95_ms", 0.0)
        if base_p95 > 0 and cur_p95 > factor * base_p95:
            failures.append(
                f"[{scenario}] p95 latency regression: "
                f"{cur_p95:.0f}ms > {factor:.1f}× baseline {base_p95:.0f}ms"
            )

        base_tput = base.get("throughput_per_min", 0.0)
        cur_tput = cur.get("throughput_per_min", 0.0)
        if base_tput > 0 and cur_tput < throughput_floor_factor * base_tput:
            failures.append(
                f"[{scenario}] throughput regression: "
                f"{cur_tput:.1f}/min < {throughput_floor_factor:.1f}× baseline {base_tput:.1f}/min"
            )

        base_err = base.get("error_rate", 0.0)
        cur_err = cur.get("error_rate", 0.0)
        if cur_err > base_err + 0.05:
            failures.append(
                f"[{scenario}] error rate regression: "
                f"{cur_err:.1%} > baseline {base_err:.1%} + 5pp tolerance"
            )

    return len(failures) == 0, failures


def _print_comparison(current: list[dict], baseline: list[dict]) -> None:
    base_by = {m["scenario"]: m for m in baseline}
    header = f"{'scenario':<8}  {'metric':<22}  {'baseline':>12}  {'current':>12}  {'delta':>10}"
    print(header)
    print("-" * len(header))

    for cur in current:
        sc = cur["scenario"]
        base = base_by.get(sc, {})
        rows = [
            ("throughput/min",   base.get("throughput_per_min"), cur.get("throughput_per_min"), "+"),
            ("p50_ms",           base.get("latency_p50_ms"),     cur.get("latency_p50_ms"),     "-"),
            ("p95_ms",           base.get("latency_p95_ms"),     cur.get("latency_p95_ms"),     "-"),
            ("p99_ms",           base.get("latency_p99_ms"),     cur.get("latency_p99_ms"),     "-"),
            ("peak_per_worker_mb", base.get("peak_per_worker_mb"), cur.get("peak_per_worker_mb"), "-"),
            ("error_rate",       base.get("error_rate"),         cur.get("error_rate"),         "-"),
            ("cache_hit_rate",   base.get("cache_hit_rate"),     cur.get("cache_hit_rate"),     "+"),
        ]
        for metric, bval, cval, direction in rows:
            if bval is None or cval is None:
                continue
            delta = cval - bval
            sign = "+" if delta >= 0 else ""
            good = (delta >= 0) if direction == "+" else (delta <= 0)
            marker = "" if good else " ◄ REGR"
            print(f"{sc:<8}  {metric:<22}  {bval:>12.1f}  {cval:>12.1f}  {sign}{delta:>9.1f}{marker}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python benchmarks/check_regression.py",
        description="Compare benchmark metrics against a committed baseline",
    )
    p.add_argument("--current", required=True, type=Path,
                   help="JSON file from the current benchmark run")
    p.add_argument("--baseline", required=True, type=Path,
                   help="Committed baseline JSON file")
    p.add_argument("--factor", type=float, default=2.0,
                   help="p95 regression factor (default: 2.0)")
    p.add_argument("--throughput-floor", type=float, default=0.5,
                   dest="throughput_floor",
                   help="Minimum fraction of baseline throughput (default: 0.5)")
    args = p.parse_args()

    if not args.current.exists():
        print(f"ERROR: current metrics file not found: {args.current}", file=sys.stderr)
        return 1
    if not args.baseline.exists():
        print(f"ERROR: baseline file not found: {args.baseline}", file=sys.stderr)
        return 1

    current = _load(args.current)
    baseline = _load(args.baseline)

    _print_comparison(current, baseline)

    passed, failures = check_regression(
        current, baseline,
        factor=args.factor,
        throughput_floor_factor=args.throughput_floor,
    )

    if passed:
        print("OK — no regressions detected.")
        return 0

    print("REGRESSIONS DETECTED:")
    for f in failures:
        print(f"  {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
