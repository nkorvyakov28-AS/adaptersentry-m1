"""AdapterSentry M1 — static security scanner for LoRA adapters.

Quick start
-----------
>>> from pathlib import Path
>>> from adaptersentry import scan
>>> result = scan(Path("adapter_model.safetensors"))
>>> result.verdict.action          # "allow" | "review" | "block"
"""

from adaptersentry.scanner import scan
from adaptersentry.schemas.result import ScanResult, load_scan_result
from adaptersentry.version import __version__

__all__ = ["ScanResult", "__version__", "load_scan_result", "scan"]
