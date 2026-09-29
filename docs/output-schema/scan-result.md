# ScanResult 2.0.0

One JSON document per scanned adapter, produced by `adaptersentry scan --format json`,
`adaptersentry batch` and `adaptersentry.scan()`. The machine-readable definition is
[scan-result-2.0.0.schema.json](scan-result-2.0.0.schema.json) (generated from the model and
checked for drift in CI).

**Read `verdict.action`.** Everything else explains it.

Stability: within 2.x, fields may be added (readers ignore unknown fields); meaning of
existing fields does not change. `load_scan_result()` rejects documents whose
`schema_version` is not 2.x.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `"2.x.y"` | contract version |
| `scan` | object | who scanned, when, how |
| `artifact` | object \| null | which file; null only if the file could not be read at all |
| `adapter` | object | what the adapter is, declared and measured |
| `status` | `ok` \| `degraded` \| `failed` | `degraded`: part of the file was not analysed; `failed`: nothing could be analysed |
| `coverage` | object | what was and was not analysed |
| `verdict` | object | the decision |
| `anomaly` | object | what the decision rests on |
| `findings` | list | human-readable findings with locations |
| `modules` | list \| null | per-module features; `null` unless requested |
| `errors` | list | technical errors |

## `scan`
| Field | Meaning |
|---|---|
| `scan_id` | `sha256(content_hash : config_hash : schema_version)` — deterministic |
| `run_id` | batch run, or null |
| `analyzer_version` | package version |
| `config_hash` | hash of every setting that changes the result (mode, policy, thresholds) |
| `mode` | `full` \| `fast` |
| `policy` | `default` \| `strict` |
| `started_at`, `completed_at`, `wall_time_ms` | timing |

## `artifact`
| Field | Meaning |
|---|---|
| `content_hash`, `header_hash` | `sha256:` of the whole file / of the safetensors header |
| `logical_id` | stable identity (Hub repo + revision + file name, or the path) |
| `file_size_bytes` | size |
| `provenance.kind` | `local_path` \| `hf_hub` |
| `provenance.path` | file name (absolute path only with `--full-paths`) |
| `provenance.hf_repo_id`, `hf_revision` | Hub provenance when known |
| `provenance.config_file` | `adapter_config.json` if found next to the file |
| `provenance.sibling_files[]` | `{name, kind}` for code / pickle / archive files next to the adapter |

## `adapter`
| Field | Meaning |
|---|---|
| `format` | `peft_lora` \| `kohya_lora` \| `unknown` |
| `peft_type`, `base_model_declared` | from `adapter_config.json` or header metadata |
| `base_family`, `base_family_source` | e.g. `llama` from `config`, or `unknown` |
| `n_model_layers` | depth, from layer indices in tensor names |
| `rank_declared` | `r` from the config/metadata |
| `rank_actual` | `{min, max, distinct}` measured from tensor shapes |
| `lora_alpha`, `scaling` | `alpha_over_r` \| `rslora` \| `pattern` \| `unknown` (no config) |
| `init_method`, `use_dora` | PEFT init method; DoRA present |
| `target_modules_declared`, `target_modules_actual` | declared vs found in tensors |
| `modules_to_save` | modules shipped as full matrices |
| `trainable_token_indices` | token-level deltas present |
| `moe` | `{n_experts, router_adapted}` or null |
| `training_state` | `trained` \| `init_only` \| `partial` \| `unknown` (B = 0 means untrained) |
| `metadata_present`, `config_present` | provenance signals |

## `coverage`
| Field | Meaning |
|---|---|
| `n_tensors_total`, `n_tensors_analyzed` | tensors in the file / in analysed LoRA pairs |
| `n_modules`, `n_modules_analyzed` | LoRA pairs found / analysed |
| `n_not_analyzed` | tensors not analysed (the list below is capped at 50) |
| `not_analyzed[]` | `{tensor_key, reason, category}` |
| `non_finite_modules[]` | modules excluded because of NaN/Inf weights |

## `verdict`
| Field | Meaning |
|---|---|
| `action` | `allow` \| `review` \| `block` |
| `level` | `LOW` (allow) \| `MEDIUM` (review) \| `HIGH` (block) \| `CRITICAL` (block, reference p ≤ α_block/10) |
| `reasons[]` | `{code, severity, message}` — see below |
| `m2_recommended` | behavioural verification recommended (any non-allow action, or missing metadata) |
| `confidence.level` | `high` \| `medium` \| `low` |
| `confidence.limiting_factors[]` | what limits it (no reference profile, uncalibrated thresholds, tensors not analysed, DoRA, no config, …) |

Invariants enforced by the model: `status != ok` ⇒ `action != allow`; policy `default` without
a reference profile ⇒ `action != block`; a `block` without a reference profile carries reason
`STRICT_POLICY_STRUCTURAL`.

**Reason codes**

| Code | Meaning |
|---|---|
| `PARSE_FAILED` | nothing could be analysed |
| `DEGRADED_PARSE` | some tensors were not analysed |
| `NO_REFERENCE_PROFILE` | no calibrated reference for the model family |
| `REFERENCE_P_BELOW_ALPHA` | the adapter is unusual against the reference profile |
| `INTRA_OUTLIER` | a module departs from its family (robust z ≥ 8) |
| `FULL_WEIGHT_REPLACEMENT`, `ROUTER_ADAPTED`, `RANK_MISMATCH`, `SIBLING_EXECUTABLE` | structural red flags |
| `STRICT_POLICY_STRUCTURAL` | strict policy turned a red flag into `block` |
| `MISSING_METADATA` | no metadata in the header and no config |

## `anomaly`
| Field | Meaning |
|---|---|
| `intra` | `{max_robust_z, n_outlier_modules, depth_spikes[{module_type, layer, z}], energy_gini, logrms_dispersion{module: MAD}, attn_mlp_ratio}` — null when no family has ≥ 16 modules |
| `reference` | `{profile_id, profile_version, stratum, n_reference, distance, p_value, alpha_review, alpha_block}` — null until reference profiles ship |
| `token_concentration[]` | for LoRA on `lm_head` / embeddings: top token ids, their energy shares, Hoyer, Gini |
| `top_contributors[]` | ≤ 10 `{feature, module_type, layer, head, expert, value, z}` |

## `findings[]`
`{rule_id, severity, title, message, evidence, locations[{layer, module, head, expert, tensor_key}], remediation}`.

| Rule | Severity | Meaning |
|---|---|---|
| `INTRA_DEPTH_SPIKE` | MEDIUM | module departs from its family (|z| ≥ 8) |
| `INTRA_OUTLIER` | LOW | mild outlier (3.5 < |z| < 8) |
| `FULL_WEIGHT_REPLACEMENT` | MEDIUM | full `lm_head` / embedding / router matrix shipped |
| `FULL_WEIGHT_NOT_ANALYSED` | LOW | other full matrices (e.g. a classification head) |
| `ROUTER_ADAPTED` | MEDIUM | LoRA on a MoE router |
| `RANK_MISMATCH` | MEDIUM | ranks differ from declared `r` without a `rank_pattern` |
| `SIBLING_EXECUTABLE` | MEDIUM | code / pickle / archive files next to the adapter |
| `TOKEN_CONCENTRATION` | LOW | token-level concentration of an `lm_head` / embedding update |
| `DORA_PARTIAL_ANALYSIS`, `UNRESOLVED_SCALE_PATTERN`, `CONFIG_FIELDS_IGNORED`, `TARGET_MODULES_MISMATCH`, `TRAINABLE_TOKENS` | LOW | analysis completeness and consistency notes |

## `modules[]` (opt-in)
One record per LoRA pair: `{layer, module, kind, expert, shape_a, shape_b, rank, scale, status, features, z, heads}`.

`features` (all of the effective update ΔW = s·B·A):

| Feature | Definition |
|---|---|
| `log_rms` | log(‖ΔW‖_F / √(d·k)) |
| `log_sigma1` | log σ₁ |
| `top1_e` | σ₁² / Σσ² |
| `srank_r` | (Σσ² / σ₁²) / r |
| `pr_r` | ((Σσ²)² / Σσ⁴) / r |
| `spec_h` | spectral entropy of energy shares above the bf16 noise floor, / log r |
| `excess_conc` | log(srank_null / srank), random d×r baseline |
| `kurt_dw` | Pearson kurtosis of ΔW entries (importance-sampled rows) |
| `row_hoyer`, `row_gini`, `row_top8` | concentration of row norms (tokens for lm_head / embeddings) |
| `col_hoyer` | concentration of column norms |

`z` holds the robust z-score of each feature within the module's family. `heads[]` (with
`--include-heads`) holds the same features per attention head.
