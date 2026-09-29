"""Engine schema contracts: batch requests, artifact identity, cache entries, combined report."""

from adaptersentry.engine.schemas.cache import CacheEntry
from adaptersentry.engine.schemas.combined_report import BehavioralResult, CombinedReport, PolicyGateResult
from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity
from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource

__all__ = [
    "AdapterArtifactIdentity",
    "AdapterScanRequest",
    "ArtifactSource",
    "BehavioralResult",
    "CacheEntry",
    "CombinedReport",
    "PolicyGateResult",
]
