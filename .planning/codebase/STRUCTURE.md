# Directory Structure

**Analysis Date:** 2026-05-06

## Top-Level Layout

```
adaptersentry-m1/
├── src/adaptersentry/      # Main Python package (src-layout)
├── tests/                  # Test suite — mirrors src structure
├── benchmarks/             # Benchmark harness and corpus runners
├── adaptersentry-rs/       # Rust extension (hot-path numerics, optional)
├── docs/                   # Documentation (public + internal)
│   ├── internal/           # Architecture, decisions, benchmarks (gitignored context)
│   └── user-guide/         # Public usage docs
├── tools/                  # Dev tooling (release prep)
├── .github/workflows/      # CI: ci.yml (test + lint), release.yml
├── pyproject.toml          # Single source of truth for package config
├── requirements.txt        # Mirrors [project.dependencies] for pip workflows
├── CHANGELOG.md            # Public version history
├── SECURITY.md             # Vulnerability reporting policy
└── CONTRIBUTING.md         # Contributor guide
```

## Package Structure (`src/adaptersentry/`)

```
src/adaptersentry/
├── __init__.py             # Public API: analyze, scan, scan_to_result, __version__
├── __main__.py             # python -m adaptersentry entry point
├── analyzer.py             # Core M1 pipeline + legacy API (_run_analysis, analyze, scan)
├── version.py              # __version__ string (single source of truth)
│
├── cli/                    # CLI layer (argparse)
│   ├── main.py             # Top-level dispatcher (scan / batch subcommands)
│   ├── scan.py             # Single-file scan subcommand
│   └── batch.py            # Batch scan subcommand
│
├── engine/                 # Batch orchestration layer
│   ├── orchestrator.py     # Single-process batch coordinator + worker pool
│   ├── orchestrator_ray.py # Ray actor pool variant
│   ├── worker.py           # worker_main() — picklable per-adapter pipeline unit
│   ├── feature_extractor.py # Typed per-layer feature extraction (new path)
│   ├── result_sink.py      # Persists ScanResults to manifest + output files
│   ├── cache.py            # CacheStore — SQLite result cache
│   ├── manifest.py         # ManifestDB — SQLite batch state tracker
│   ├── identity.py         # ArtifactIdentityResolver (content_hash, header_hash)
│   ├── config.py           # EngineConfig — worker pool / cache settings
│   └── schemas/            # Engine-internal typed schemas
│       ├── scan_result.py  # ScanResult (stable v1.0.0 contract), DebugReport
│       ├── identity.py     # AdapterArtifactIdentity, ScanIdentity
│       ├── requests.py     # AdapterScanRequest, ArtifactSource
│       ├── signals.py      # FeatureFamilyResult (per-layer typed signals)
│       ├── scoring.py      # EnsembleSignal, RiskVerdict
│       ├── combined_report.py # CombinedReport (batch summary)
│       └── cache.py        # CacheEntry schema
│
├── parsers/                # Adapter file parsing
│   ├── safetensors.py      # Header + tensor loading via safetensors.numpy
│   └── metadata.py         # Adapter metadata extraction and depth checking
│
├── features/               # Per-layer feature computation
│   ├── delta_norm.py       # Norm features (L1/L2/Frobenius)
│   ├── distribution.py     # Distribution features (kurtosis, skewness, etc.)
│   ├── entropy_compression.py # Entropy + compressibility features
│   ├── inter_layer_similarity.py # Cross-layer cosine similarity
│   ├── layer_stats.py      # Per-layer statistical anomaly flags
│   └── tensor_stats.py     # SVD stats, energy concentration
│
├── detectors/              # Anomaly detectors
│   ├── entropy.py          # Entropy-based detection
│   ├── outlier.py          # IsolationForest outlier detection
│   ├── wasserstein.py      # Wasserstein distance detector
│   ├── cross_layer.py      # Cross-layer consistency detection
│   └── init_detector.py    # LoRA initialization status detection
│
├── scoring/                # Risk scoring
│   ├── ensemble.py         # EnsembleDetector — weighted logistic-regression scorer
│   ├── risk_scorer.py      # RiskScorer — flag → risk_level mapping
│   ├── confidence.py       # ConfidenceScore computation
│   └── score_breakdown.py  # ScoreBreakdown — per-detector contribution
│
├── schemas/                # Shared typed schemas (public API surface)
│   ├── adapter_report.py   # AdapterReport, AnalysisMode, ParseStatus
│   ├── adapter_metadata.py # AdapterMetadata
│   ├── finding.py          # Finding (top-level anomaly finding)
│   ├── per_layer_finding.py # PerLayerFinding
│   ├── tensor_record.py    # TensorRecord (per-layer typed record)
│   ├── errors.py           # ScanError, ErrorCode, ScanPhase
│   ├── confidence_score.py # ConfidenceScore, AnalysisQualityScore
│   ├── score_breakdown.py  # ScoreBreakdown schema
│   ├── scoring_policy.py   # ScoringPolicy
│   ├── distribution_features.py
│   ├── entropy_compression_features.py
│   ├── inter_layer_similarity_features.py
│   └── norm_features.py
│
├── reporters/              # Output format reporters
│   ├── json.py             # JSON reporter
│   ├── sarif.py            # SARIF reporter (GitHub Code Scanning format)
│   └── text.py             # Plain text reporter
│
├── reporting/              # Human-readable report formatting
│   ├── human_summary.py    # Rich console summary
│   └── per_layer.py        # Per-layer finding formatter
│
└── integrations/
    └── github_action.py    # GitHub Actions output (GITHUB_OUTPUT, step summary)
```

## Test Structure (`tests/`)

```
tests/
├── test_m1.py              # Core analyzer integration tests
├── test_analyzer.py        # Analyzer module tests
├── test_ensemble.py        # EnsembleDetector tests
├── test_detectors.py       # Detector-level tests
├── test_risk_scorer.py     # RiskScorer tests
├── test_init_detector.py   # InitDetector tests
├── test_harness.py         # Benchmark harness tests
├── test_bench.py           # Benchmark runner tests
├── test_m1_backward_compat.py # Backward compatibility regression tests
├── cli/                    # CLI tests
├── engine/                 # Engine layer tests (orchestrator, worker, cache, manifest)
├── features/               # Feature extraction tests
├── parsers/                # Parser tests
├── reporters/              # Reporter tests
├── reporting/              # Human summary tests
├── schemas/                # Schema validation and migration tests
├── scoring/                # Scoring tests
└── fixtures/               # Test data files (e.g., scan_result_v1.0.0.json)
```

## Rust Extension (`adaptersentry-rs/`)

```
adaptersentry-rs/
├── Cargo.toml              # pyo3 0.22, numpy 0.22, opt-level 3, lto=thin
└── src/
    └── lib.rs              # PyO3 extension — hot-path numeric operations
```

## Key File Locations

| What you need | Where to look |
|---|---|
| Public API surface | `src/adaptersentry/__init__.py` |
| Stable output contract | `src/adaptersentry/engine/schemas/scan_result.py` |
| Core analysis pipeline | `src/adaptersentry/analyzer.py` (legacy), `src/adaptersentry/engine/feature_extractor.py` (typed) |
| CLI entry point | `src/adaptersentry/cli/main.py` |
| Package metadata + deps | `pyproject.toml` |
| Version string | `src/adaptersentry/version.py` |
| Schema fixture | `tests/fixtures/scan_result_v1.0.0.json` |
| Internal architecture docs | `docs/internal/` |
| ADR decisions | `docs/internal/decisions/` |

## Naming Conventions

- **Modules:** `snake_case` throughout
- **Public classes:** `PascalCase` (e.g., `EnsembleDetector`, `ScanResult`, `FeatureExtractor`)
- **Private functions:** leading underscore (e.g., `_run_analysis`, `_pool_initializer`, `_sha256`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DETECTOR_WEIGHTS`, `LEASE_DURATION_SECS`)
- **Schema version:** `schema_version = "1.0.0"` string field on `ScanResult`
- **Test files:** `test_<module_name>.py`, mirrors `src/` subdirectory layout
- **Fixtures:** helper functions prefixed `_make_` (e.g., `_make_adapter`, `_make_req`)

---

*Structure analysis: 2026-05-06*
