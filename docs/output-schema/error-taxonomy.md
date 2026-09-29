# Error Taxonomy

> Updated for v0.3.0. New fields: `severity` (auto-inferred from code), `phase`.

AdapterSentry normalizes all scan failures into structured `ScanError` objects
in `AdapterReport.errors` (single-scan) or `ScanResult.errors` (engine output).

## ScanError fields

```python
ScanError(
    category: ErrorCategory,    # malformed | unsupported | degraded
    code:     str,              # ErrorCode or freeform string
    message:  str,              # human-readable summary
    detail:   str | None,       # optional exception string
    severity: ErrorSeverity,    # fatal | degraded | warning  (auto-inferred)
    phase:    ScanPhase | None, # parse | metadata | feature | scoring | reporting
)
```

`severity` is automatically inferred from `code` when not supplied explicitly.

---

## ErrorCategory

### `malformed`

| Code | Severity | Description |
|------|----------|-------------|
| `INVALID_SAFETENSORS` | fatal | File is not a valid safetensors file |
| `TRUNCATED_FILE` | fatal | File appears cut off or incomplete |
| `TENSOR_TOO_LARGE` | fatal | Tensor exceeds 1B element safety limit |
| `TENSOR_SHAPE_INVALID` | degraded | Tensor declares an invalid or zero-element shape |
| `METADATA_PARSE_ERROR` | degraded | Metadata header present but cannot be parsed |

### `unsupported`

| Code | Severity | Description |
|------|----------|-------------|
| `NO_LORA_PAIRS` | fatal | No `lora_A.weight` / `lora_B.weight` pairs found |
| `UNSUPPORTED_PEFT_TYPE` | fatal | IA³, LoHa, LoCon or other non-LoRA PEFT method |
| `SHARDED_ADAPTER` | fatal | Multi-file sharded adapter (single-file only in M1) |

### `degraded`

| Code | Severity | Description |
|------|----------|-------------|
| `PARTIAL_LAYER_ANALYSIS` | degraded | One or more layers could not be fully analyzed |
| `SVD_FAILED` | degraded | SVD computation failed for a layer |
| `ISOLATION_FOREST_SKIPPED` | degraded | IsolationForest skipped (size guard or fast mode) |
| `METADATA_DEPTH_EXCEEDED` | warning | Metadata nesting > 5 levels — security signal |

---

## ErrorSeverity

| Value | Meaning | Effect |
|-------|---------|--------|
| `fatal` | Prevents meaningful output | `status=failed` or layer skipped |
| `degraded` | Partial output; signals may be incomplete | `status=degraded` |
| `warning` | Informational; no quality impact | Result unaffected |

---

## ScanPhase

| Phase | Description |
|-------|-------------|
| `parse` | File I/O, safetensors header and tensor loading |
| `metadata` | Adapter config / safetensors metadata extraction |
| `feature` | Per-layer feature extraction (SVD, kurtosis, entropy, …) |
| `scoring` | Ensemble scoring and verdict derivation |
| `reporting` | Result serialisation and output formatting |

---

## Security significance

A degraded scan is itself a security signal. Deliberately crafted adapters may
trigger analysis failures to evade detection.

| Code | Security interpretation |
|------|------------------------|
| `METADATA_DEPTH_EXCEEDED` | Possible metadata evasion attempt |
| `TENSOR_TOO_LARGE` | Possible tensor bomb / resource exhaustion |
| `NO_LORA_PAIRS` | Mislabelled adapter (non-LoRA published as LoRA) |
| `PARTIAL_LAYER_ANALYSIS` | Selective layer corruption to hide anomalies |
| `TENSOR_NOT_ANALYZED` | A tensor was skipped (unsupported dtype, failed header validation or load, or not part of a `lora_A`/`lora_B` pair). The payload may live in what was not analysed |
| `INVALID_SAFETENSORS` | The file could not be parsed (bounded header, regular-file and structure checks). Scan is `FAILED`; verdict is at least `review` |

Layers whose weights contain NaN or Inf are excluded from analysis and carry the
`NON_FINITE_WEIGHTS` flag; the scan is `DEGRADED`.

Since v1.0.3 a `degraded` or `failed` scan never produces `recommended_action: "allow"`.

---

## Non-LoRA format handling (v0.3.0)

`has_lora_pairs()` reads only tensor key names before loading any tensor data.
Files without `lora_A` / `lora_B` keys are rejected immediately with `NO_LORA_PAIRS`.

| Format | Key pattern |
|--------|-------------|
| IA³ | `…ia3_l` |
| Prefix tuning | `prompt_embeddings` |
| LoHa / LoCon | `…hada_w1_a` |
| Full fine-tune | no standard adapter pattern |
