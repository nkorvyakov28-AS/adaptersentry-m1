"""``adaptersentry scan`` — scan one LoRA adapter and print a ScanResult 2.0.0.

Output formats
--------------
text        human-readable summary (default)
json        ScanResult 2.0.0 — the stable contract; CI gates should read verdict.action
full-json   ScanResult 2.0.0 with per-module features (modules[])
sarif       SARIF 2.1.0 for GitHub code scanning

Exit codes
----------
0  scan completed; verdict below the --fail-on threshold (or no threshold)
1  the adapter could not be analysed (status "failed"; verdict is still "review")
2  verdict action reached the --fail-on threshold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_FORMATS = ("text", "json", "full-json", "sarif")
_ACTION_RANK = {"allow": 0, "review": 1, "block": 2}


def build_parser(subparsers: Any) -> None:
    """Register the ``scan`` subcommand."""
    p = subparsers.add_parser(
        "scan",
        help="Scan a LoRA adapter .safetensors file",
        description=__doc__,
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    p.add_argument("adapter", type=Path, metavar="ADAPTER", help="Path to the .safetensors adapter file")
    p.add_argument("--format", choices=_FORMATS, default="text", dest="fmt",
                   help="Output format (default: text)")
    p.add_argument("--output", type=Path, default=None, metavar="FILE",
                   help="Write the report to FILE instead of stdout")
    p.add_argument("--mode", choices=("full", "fast"), default="full",
                   help="full (default) or fast (fewer samples for the kurtosis estimate)")
    p.add_argument("--policy", choices=("default", "strict"), default="default",
                   help="default: never block without a reference profile; "
                        "strict: block on structural red flags")
    p.add_argument("--fail-on", choices=("review", "block"), default=None, dest="fail_on",
                   help="Exit with code 2 if the verdict action is at least this")
    p.add_argument("--include-modules", action="store_true",
                   help="Include per-module features in JSON output (implied by full-json)")
    p.add_argument("--include-heads", action="store_true",
                   help="Also include per-attention-head features")
    p.add_argument("--full-paths", action="store_true",
                   help="Report the absolute file path instead of the file name")
    p.add_argument("--no-color", action="store_true", dest="no_color",
                   help="Disable ANSI colours in text output")


def run(args: Any) -> int:
    """Execute the scan subcommand and return the process exit code."""
    from adaptersentry.reporters import json as json_reporter
    from adaptersentry.reporters import sarif as sarif_reporter
    from adaptersentry.reporters import text as text_reporter
    from adaptersentry.scanner import scan

    include_modules = args.include_modules or args.include_heads or args.fmt == "full-json"
    result = scan(
        args.adapter,
        mode=args.mode,
        policy=args.policy,
        include_modules=include_modules,
        include_heads=args.include_heads,
        full_paths=args.full_paths,
    )

    if args.fmt == "text":
        no_color = args.no_color or args.output is not None or not sys.stdout.isatty()
        output = text_reporter.render(result, no_color=no_color)
    elif args.fmt == "sarif":
        output = json.dumps(sarif_reporter.render(result), indent=2) + "\n"
    else:
        output = json_reporter.render(result)

    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)

    if result.status == "failed":
        return 1
    if args.fail_on and _ACTION_RANK[result.verdict.action] >= _ACTION_RANK[args.fail_on]:
        return 2
    return 0
