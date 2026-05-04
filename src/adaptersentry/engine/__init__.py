"""AdapterSentry scan engine — batch orchestration, caching, and typed schema contracts.

This package provides the production-scale scan infrastructure built on top of
the M1 static analyzer core. It is designed for scanning large adapter corpora
(10K+) with incremental re-scan, resumable batches, and stable public contracts.

Public surface
--------------
engine.schemas      — versioned Pydantic contracts (ScanResult, ScanIdentity, ...)
engine.config       — AnalyzerConfig and deterministic config hash
engine.identity     — ArtifactIdentityResolver (SHA-256 content + header hashes)
engine.manifest     — ManifestDB (SQLite-backed batch state machine)
engine.cache        — CacheStore (content-addressed local result cache)
engine.worker       — worker_main() (per-adapter scan pipeline)
engine.result_sink  — ResultSink (atomic write, JSONL append)
engine.orchestrator — Orchestrator (batch coordinator)
"""
