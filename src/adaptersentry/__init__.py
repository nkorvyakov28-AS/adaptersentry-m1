"""AdapterSentry — static security scanner for LoRA adapters.

Quick start
-----------
>>> from adaptersentry import analyze, scan
>>> report_dict = analyze(Path("adapter.safetensors"))   # legacy dict API
>>> report = scan(Path("adapter.safetensors"))           # typed AdapterReport API
>>> print(report.risk_summary.ensemble_risk_level)
"""

from adaptersentry.version import __version__
from adaptersentry.analyzer import analyze, load_adapter, scan

__all__ = ["analyze", "load_adapter", "scan", "__version__"]
