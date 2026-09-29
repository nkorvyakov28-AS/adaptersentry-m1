"""AdapterSentry M1 — static security scanner for LoRA adapters.

Quick start
-----------
>>> from pathlib import Path
>>> from adaptersentry import scan
>>> result = scan(Path("adapter_model.safetensors"))
>>> result.verdict.action          # "allow" | "review" | "block"
"""

from typing import TYPE_CHECKING, Any

from adaptersentry.version import __version__

if TYPE_CHECKING:
    from adaptersentry.scanner import scan
    from adaptersentry.schemas.result import ScanResult, load_scan_result

__all__ = ["ScanResult", "__version__", "load_scan_result", "scan"]


def __getattr__(name: str) -> Any:
    # Lazy: importing the package must not import numpy, so the CLI and batch
    # workers can pin BLAS to one thread before numpy loads.
    if name == "scan":
        from adaptersentry.scanner import scan
        return scan
    if name in ("ScanResult", "load_scan_result"):
        from adaptersentry.schemas import result
        return getattr(result, name)
    raise AttributeError(f"module 'adaptersentry' has no attribute {name!r}")
