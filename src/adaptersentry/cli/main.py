"""Top-level CLI entry point for AdapterSentry.

Commands
--------
adaptersentry scan ADAPTER [options]   — run M1 static analysis
adaptersentry --version                — print version and exit
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_root_parser() -> argparse.ArgumentParser:
    from adaptersentry.version import __version__

    parser = argparse.ArgumentParser(
        prog="adaptersentry",
        description="AdapterSentry — static security scanner for LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  adaptersentry scan adapter.safetensors\n"
            "  adaptersentry scan adapter.safetensors --format json --output report.json\n"
            "  adaptersentry scan adapter.safetensors --format sarif --fail-on HIGH\n"
            "  adaptersentry --version\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"adaptersentry {__version__}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def main() -> None:
    """Primary entry point for ``adaptersentry`` and ``python -m adaptersentry``."""
    from adaptersentry.cli.scan import build_parser as build_scan_parser, run as run_scan
    from adaptersentry.cli.batch import build_parser as build_batch_parser, run as run_batch

    root = _build_root_parser()
    subparsers = root.add_subparsers(dest="command", metavar="COMMAND")
    build_scan_parser(subparsers)
    build_batch_parser(subparsers)

    args = root.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.command == "scan":
        sys.exit(run_scan(args))
    elif args.command == "batch":
        sys.exit(run_batch(args))
    else:
        root.print_help()
        sys.exit(0)
