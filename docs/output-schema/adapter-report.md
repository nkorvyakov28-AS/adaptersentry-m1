# AdapterReport Schema — v1.0.0

`AdapterReport` is the top-level output contract for an M1 scan.  It is produced
by `adaptersentry.analyzer.scan()` and is the input to all three reporters
(text, JSON, SARIF).

## Top-level fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | `str` | Schema version string, currently `"1.0.0"` |
| `tool` | `ToolInfo` | Scanner name, version, and information URI |
| `scan_target` | `ScanTarget` | Adapter file path and optional file size |
| `adapter_metadata` | `AdapterMetadata` | Structured metadata from safetensors header and adapter_config |
| `tensor_records` | `list[TensorRecord]` | Per-layer analysis records (see tensor-record.md) |
| `findings` | `list[Finding]` | Structured anomaly findings |
| `errors` | `list[ScanError]` | Normalized errors (malformed / unsupported / degraded) |
| `risk_summary` | `RiskSummary` | Aggregate risk scores and training status |
| `analysis_mode` | `str` | `"full"` / `"degraded"` / `"failed"` |
| `started_at` | `str` | ISO 8601 UTC timestamp |
| `completed_at` | `str` | ISO 8601 UTC timestamp |

## RiskSummary fields

| Field | Type | Description |
|---|---|---|
| `overall_risk` | `int` | Rule-based additive score (0–100) |
| `risk_level` | `Severity` | LOW / MEDIUM / HIGH / CRITICAL |
| `ensemble_score` | `float` | 7-detector weighted ensemble score (0–100) |
| `ensemble_risk_level` | `Severity` | Ensemble-derived severity |
| `training_status` | `str` | TRAINED / INIT_ONLY / PARTIALLY_TRAINED |
| `false_positive_suppressed` | `int` | Init-artifact flags suppressed |
| `n_layers` | `int` | Number of LoRA layer pairs analyzed |
| `n_findings` | `int` | Number of Finding objects |
| `cross_layer_consistency` | `float` | 0–1; low = anomaly concentration |
| `wasserstein_mean` | `float\|null` | Mean W1 A↔B distance across layers |

## Finding fields

| Field | Type | Description |
|---|---|---|
| `rule_id` | `str` | Machine-readable rule identifier (flag prefix) |
| `title` | `str` | Human-readable rule name |
| `severity` | `Severity` | LOW / MEDIUM / HIGH / CRITICAL |
| `confidence` | `float` | Detection confidence in [0, 1] |
| `affected_layers` | `list[str]` | Layer names where finding was observed |
| `evidence` | `dict` | Raw flag strings and supporting data |
| `remediation` | `str\|null` | Suggested investigation step |

## SARIF output

See [SARIF 2.1.0 output](../cli/usage.md#sarif-output) for GitHub code scanning integration notes.

## Schema versioning

`schema_version` follows semantic versioning.  Breaking changes bump the major version.
Consumers should check `schema_version` before deserializing.

The M1 → M2/M3/M4 extension contract is: M2 receives `AdapterReport.findings` and
`AdapterReport.tensor_records`.  No M1 internals are exposed.
