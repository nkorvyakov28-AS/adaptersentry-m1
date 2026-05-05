# Open-Core Boundary

AdapterSentry follows an open-core model. The static analysis engine is fully
open source. Advanced detection capabilities are planned for future releases.

## What lives in this repository (OSS)

| Component | Location | Description |
|-----------|----------|-------------|
| M1 Static Analyzer | `src/adaptersentry/` | Read-only weight tensor inspection |
| Parsers | `src/adaptersentry/parsers/` | safetensors loading, metadata extraction |
| Features | `src/adaptersentry/features/` | Tensor statistics, SVD, layer-level features |
| Detectors | `src/adaptersentry/detectors/` | Entropy, outlier, init, cross-layer, Wasserstein |
| Scoring | `src/adaptersentry/scoring/` | Rule-based and ensemble risk scoring |
| Schemas | `src/adaptersentry/schemas/` | Pydantic report contracts |
| Reporters | `src/adaptersentry/reporters/` | Text, JSON, SARIF output |
| Batch scan engine | `src/adaptersentry/engine/` | Worker pool, cache, manifest, result sink |
| CLI | `src/adaptersentry/cli/` | `adaptersentry scan` and `adaptersentry batch` |
| GitHub Action helper | `src/adaptersentry/integrations/` | GitHub Actions output helpers |
| Benchmark harness | `benchmarks/` | Throughput and regression testing |

## What is not in this repository

Advanced capabilities beyond static analysis are planned for future releases
and are not part of this open-source package.

## Integration contract

External integrations MUST consume AdapterSentry only through its public API:

1. `adaptersentry.scan()` — typed `AdapterReport` result
2. `adaptersentry.scan_to_result()` — engine-level `ScanResult` with `.identity`, `.verdict`, `.artifact` (added v1.0.2)
3. `adaptersentry batch` — `ScanResult` JSON output (`schema_version = "1.0.0"`)
4. `--format summary-json` — stable machine-readable contract for CI gates
5. `--format sarif` — SARIF 2.1.0 for GitHub code scanning

Do not depend on internal modules — these are not part of the public contract
and may change between minor versions.
