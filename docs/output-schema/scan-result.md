# ScanResult Schema

> v0.3.0. Emitted by `--format summary-json` (single scan) and by `adaptersentry batch`.
> This is the stable public contract for machine consumers and CI gates.
> See [adapter-report.md](adapter-report.md) for the legacy `AdapterReport` schema.

## Overview

`ScanResult` is the engine-level output schema. It extends `AdapterReport` with:

- Deterministic `scan_id` (content-addressed, reproducible across runs)
- `ScanIdentity` — which analyzer version, config hash, and run produced this result
- `AdapterArtifactIdentity` — content hash and provenance of the scanned file
- `EnsembleSignal` — pure statistical score separated from policy verdict
- `RiskVerdict` — policy-level decision: `allow` / `review` / `block`
- `schema_version = "1.0.0"` with `extra="ignore"` for forward compatibility

## Top-level fields

```json
{
  "schema_version": "1.0.0",
  "identity":        { ScanIdentity },
  "artifact":        { AdapterArtifactIdentity },
  "adapter_metadata":{ AdapterMetadata },
  "verdict":         { RiskVerdict },
  "ensemble":        { EnsembleSignal },
  "findings":        [ Finding, … ],
  "errors":          [ ScanError, … ],
  "status":          "ok | degraded | failed | cached",
  "parse_status":    "ok | degraded | failed",
  "analysis_mode":   "full | degraded | failed",
  "n_layers":        2,
  "n_layers_analyzed": 2
}
```

## ScanIdentity

Identifies which analyzer run produced this result.

```json
{
  "scan_id":             "sha256:<hex>",
  "run_id":              "batch-run-20260501",
  "analyzer_version":    "0.3.0",
  "analyzer_config_hash":"sha256:<hex>",
  "schema_version":      "1.0.0",
  "started_at":          "2026-05-01T04:00:00+00:00",
  "completed_at":        "2026-05-01T04:01:41+00:00",
  "wall_time_ms":        101250
}
```

`scan_id` is deterministic: `sha256(content_hash + ':' + analyzer_config_hash + ':' + schema_version)`.
Same file + same config always produces the same `scan_id`. Safe to use as a cache key.

`analyzer_config_hash` includes scan mode (`full`/`fast`), detector weights, thresholds,
enabled feature families, and tool version. Any config change produces a new hash.

## AdapterArtifactIdentity

Content-addressed identity of the scanned file.

```json
{
  "logical_id":     "sha256:<hex>",
  "content_hash":   "sha256:<hex>",
  "header_hash":    "sha256:<hex>",
  "file_size_bytes": 75497472,
  "source": {
    "kind":       "local_path",
    "local_path": "/adapters/my-model/adapter_model.safetensors"
  },
  "resolved_at": "2026-05-01T04:00:00+00:00"
}
```

`content_hash` = BLAKE3 (SHA256 fallback) of full file bytes.
`header_hash` = BLAKE3 of safetensors header only — fast re-identification without re-hashing the file.

## RiskVerdict

Policy-level decision. This is what CI gates and enforcement policies should read.

```json
{
  "overall_score":            14,
  "overall_level":            "HIGH",
  "recommended_action":       "block",
  "m2_recommended":           true,
  "false_positive_suppressed":0,
  "training_status":          "TRAINED",
  "policy_signals":           []
}
```

| Field | Description |
|-------|-------------|
| `recommended_action` | `allow` / `review` / `block` — what automated enforcement should do |
| `m2_recommended` | `true` when M2 behavioral sandbox is recommended (score ≥ 14 or missing metadata) |
| `overall_score` | Additive rule-based score 0–100 |
| `overall_level` | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| `false_positive_suppressed` | Init-artifact flags suppressed (INIT_ONLY adapters) |
| `training_status` | `TRAINED` / `INIT_ONLY` / `PARTIALLY_TRAINED` / `UNKNOWN` |
| `policy_signals` | Non-statistical signals (missing metadata, degraded parse) |

## EnsembleSignal

Pure statistical output of the detector ensemble — no policy logic.

```json
{
  "score":      23.4,
  "risk_level": "HIGH",
  "top_contributors": [
    {
      "name":      "HIGH_ENERGY_CONCENTRATION",
      "family":    "spectral",
      "value":     0.97,
      "threshold": 0.95,
      "direction": "above",
      "layer":     "model.layers.0.self_attn.q_proj",
      "severity":  "HIGH",
      "confidence":1.0
    }
  ],
  "detector_weights": {}
}
```

Tune `score` thresholds for your threat model. `top_contributors` are ranked by
weighted contribution to the ensemble score.

## ScanStatus

| Value | Meaning |
|-------|---------|
| `ok` | All phases completed; all enabled detectors ran |
| `degraded` | Some phases/detectors failed; result is partial but meaningful |
| `failed` | File-level failure; no meaningful result produced |
| `cached` | Result served from cache; no recomputation performed |

## DebugReport (not stable)

`--format debug-json` emits `DebugReport` which extends `ScanResult` with:

```json
{
  "debug_schema_version":     "debug-1.0.0",
  "tensor_records":           [ TensorRecord, … ],
  "feature_family_results":   [ FeatureFamilyResult, … ],
  "raw_layer_stats":          { … },
  "raw_flags":                [ "HIGH_KURTOSIS: …", … ],
  "wasserstein_distances":    { "layer.q_proj": 0.23, "_mean": 0.18 },
  "cross_layer_consistency":  0.91
}
```

`DebugReport` is **not** part of the stable public contract. Its schema may change
between minor versions. Do not use it in CI gates or external tooling.

## Forward compatibility

`schema_version` must be checked before deserializing. `extra="ignore"` ensures
fields added by newer writers are safely dropped by older readers.

```python
data = json.loads(result_json)
if data["schema_version"] != "1.0.0":
    raise ValueError(f"Unsupported schema version: {data['schema_version']}")
result = ScanResult.model_validate(data)
```

## Migration tests

`tests/fixtures/scan_result_v1.0.0.json` is a frozen baseline fixture.
`tests/schemas/test_migration.py` verifies round-trip compatibility on every test run.

When `schema_version` is bumped, generate a new fixture:

```bash
python scripts/snapshot_schema.py --version 1.1.0
```
