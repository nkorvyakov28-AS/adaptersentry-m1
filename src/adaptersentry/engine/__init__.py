"""AdapterSentry scan engine — batch orchestration, caching, and typed schema contracts.

This package runs adaptersentry.scanner.scan over large adapter corpora. It is designed for scanning large adapter corpora
(10K+) with incremental re-scan, resumable batches, and stable public contracts.

Public surface
--------------
engine.schemas      — batch requests, artifact identity, cache entries, CombinedReport
engine.identity     — ArtifactIdentityResolver (SHA-256 content + header hashes)
engine.manifest     — ManifestDB (SQLite-backed batch state machine)
engine.cache        — CacheStore (content-addressed local result cache)
engine.worker       — worker_main() (per-adapter scan pipeline)
engine.result_sink  — ResultSink (atomic write, JSONL append)
engine.orchestrator — Orchestrator (batch coordinator)
"""
