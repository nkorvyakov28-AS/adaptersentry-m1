"""CacheEntry — the record stored in the local cache index.

Cache key: (content_hash, analyzer_config_hash).
Cache store layout:
    ~/.adaptersentry/cache/
        index.sqlite              — CacheEntry rows
        objects/
            {hash[:2]}/
                {hash[2:]}.gz     — compressed ScanResult JSON

Poisoning guard: result_hash is SHA256 of the compressed result file.
On every cache read, the file is re-hashed and compared to result_hash.
A mismatch causes the entry to be deleted and a full rescan to be triggered.
We never serve a result whose integrity cannot be verified.

writer_version: entries written by a future adaptersentry version are not
consumed by older readers. This prevents silent behavior mismatches when
downgrading or running mixed versions in a team environment.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CacheEntry(BaseModel):
    """Index record for one cached ScanResult."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    # Cache key components — all must match for a hit
    content_hash: str = Field(description="sha256 of full adapter file. Format: 'sha256:<hex>'.")
    analyzer_config_hash: str = Field(
        description="sha256 of canonical AnalyzerConfig JSON. Invalidated on any config change."
    )
    schema_version: str = Field(description="schema_version of the cached ScanResult.")

    # Cached result location
    scan_id: str
    result_path: str = Field(
        description=(
            "Relative path within cache store to the compressed ScanResult JSON. "
            "Relative to the cache store root — never an absolute path."
        )
    )
    result_hash: str = Field(
        description=(
            "sha256 of the compressed result file. "
            "Recomputed on every read as a poisoning guard. "
            "Format: 'sha256:<hex>'."
        )
    )

    # Cache metadata
    cached_at: str = Field(description="ISO 8601 UTC when entry was written.")
    hit_count: int = Field(default=0, ge=0)
    last_hit_at: str | None = None
    ttl_days: int | None = Field(
        default=None,
        description="None = no expiry. Set for time-sensitive scans.",
    )

    # Poisoning / version guard
    writer_version: str = Field(
        description=(
            "adaptersentry version that wrote this entry. "
            "Older readers reject entries from future writer versions."
        )
    )
