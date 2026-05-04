# TensorRecord Schema

`TensorRecord` captures per-layer analysis results.  One record is produced for each
matched `lora_A.weight` / `lora_B.weight` tensor pair.

## Fields

| Field | Type | Description |
|---|---|---|
| `layer_name` | `str` | Canonical LoRA layer path (e.g. `base_model.model.layers.0.self_attn.q_proj`) |
| `shape_a` | `list[int]` | lora_A matrix shape `[rank, in_features]` |
| `shape_b` | `list[int]` | lora_B matrix shape `[out_features, rank]` |
| `dtype` | `str` | Weight dtype (typically `"float32"`) |
| `rank` | `int` | SVD effective rank (captures 99% of weight energy) |
| `energy_concentration` | `float` | Fraction of energy in the dominant singular value (0–1) |
| `kurtosis_a` | `float` | Excess kurtosis of lora_A weights (Fisher definition) |
| `kurtosis_b` | `float` | Excess kurtosis of lora_B weights |
| `mean_a` | `float` | Mean of lora_A weights |
| `std_a` | `float` | Standard deviation of lora_A weights |
| `mean_b` | `float` | Mean of lora_B weights |
| `std_b` | `float` | Standard deviation of lora_B weights |
| `skewness_a` | `float` | Skewness of lora_A weights |
| `entropy_a` | `float` | Normalized Shannon entropy of lora_A (0–1) |
| `entropy_b` | `float` | Normalized Shannon entropy of lora_B (0–1) |
| `zscore_outlier_rate_a` | `float` | Fraction of lora_A weights beyond ±3σ |
| `zscore_outlier_rate_b` | `float` | Fraction of lora_B weights beyond ±3σ |
| `isolation_score_a` | `float\|null` | IsolationForest mean score for lora_A (negative = anomalous) |
| `flags` | `list[str]` | Per-layer anomaly flag strings |

## Anomaly thresholds

| Signal | Threshold | Flag emitted |
|---|---|---|
| `kurtosis_a > 10` | | `HIGH_KURTOSIS_A` |
| `kurtosis_b > 10` | | `HIGH_KURTOSIS_B` |
| `energy_concentration > 0.95` | | `HIGH_ENERGY_CONCENTRATION` |
| `std_b < 1e-6` | | `NEAR_ZERO_B_MATRIX` |
| `entropy_a < 0.1` | | `LOW_ENTROPY_A` |
| `entropy_a > 0.99` | | `HIGH_ENTROPY_A` |
| `zscore_outlier_rate_a > 0.02` | | `HIGH_ZSCORE_OUTLIER_RATE_A` |
| `isolation_score_a < -0.1` | | `HIGH_ISOLATION_ANOMALY_A` |
