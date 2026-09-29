"""Reading adapter_config.json and computing the effective LoRA scale.

The effective update of one module is ΔW = s·B·A with
    s = alpha / r          (standard LoRA)
    s = alpha / sqrt(r)    (rsLoRA, ``use_rslora: true``)
where r and alpha may be overridden per module by ``rank_pattern`` /
``alpha_pattern``. Without the config (or without ``lora_alpha``) the PEFT
convention for alpha-less checkpoints applies: alpha = r, i.e. s = 1.

Ignoring s lets an attacker hide an update: scale B down by c and alpha up by c,
and B looks "untrained" while the model behaves the same.

Security Notes:
    - The config is untrusted. It is read only if it is a regular file of at
      most _MAX_CONFIG_BYTES; malformed fields are ignored and reported.
    - PEFT treats rank_pattern / alpha_pattern keys as regular expressions.
      Compiling attacker-supplied regexes risks catastrophic backtracking
      (Python's re has no timeout), so only literal patterns are honoured;
      a pattern containing regex metacharacters is reported as unresolved.
"""

from __future__ import annotations

import json
import logging
import math
import stat
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "adapter_config.json"
_MAX_CONFIG_BYTES = 1_000_000
_REGEX_METACHARACTERS = frozenset("\\^$*+?{}[]()|")


class AdapterConfig(BaseModel):
    """Validated subset of a PEFT adapter_config.json."""

    model_config = ConfigDict(frozen=True)

    peft_type: str | None = None
    base_model_name_or_path: str | None = None
    r: int | None = Field(default=None, ge=1)
    lora_alpha: float | None = Field(default=None, gt=0)
    use_rslora: bool = False
    use_dora: bool = False
    init_lora_weights: str | None = None
    rank_pattern: dict[str, int] = Field(default_factory=dict)
    alpha_pattern: dict[str, float] = Field(default_factory=dict)
    target_modules: list[str] = Field(default_factory=list)
    modules_to_save: list[str] = Field(default_factory=list)
    trainable_token_indices: bool = False


class ConfigReadResult(BaseModel):
    """Outcome of looking for and reading adapter_config.json."""

    model_config = ConfigDict(frozen=True)

    config: AdapterConfig | None = None
    path_name: str | None = None
    problems: list[str] = Field(default_factory=list)


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _as_number_map(value: Any, *, integer: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in value.items():
        if not isinstance(k, str) or isinstance(v, bool):
            continue
        if integer and isinstance(v, int) and v >= 1:
            out[k] = v
        elif not integer and isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            out[k] = float(v)
    return out


def parse_adapter_config(raw: dict[str, Any]) -> tuple[AdapterConfig, list[str]]:
    """Validate a decoded adapter_config.json, ignoring malformed fields.

    Returns:
        (AdapterConfig, problems) — problems lists every field that was present
        but ignored, so the caller can surface it instead of silently dropping it.
    """
    problems: list[str] = []

    def num(name: str, *, integer: bool) -> Any:
        v = raw.get(name)
        if v is None:
            return None
        ok = isinstance(v, int) if integer else isinstance(v, (int, float))
        if isinstance(v, bool) or not ok or not math.isfinite(float(v)) or v <= 0:
            problems.append(f"{name}: invalid value ignored")
            return None
        return int(v) if integer else float(v)

    init = raw.get("init_lora_weights")
    tti = raw.get("trainable_token_indices")
    config = AdapterConfig(
        peft_type=_as_str(raw.get("peft_type")),
        base_model_name_or_path=_as_str(raw.get("base_model_name_or_path")),
        r=num("r", integer=True),
        lora_alpha=num("lora_alpha", integer=False),
        use_rslora=raw.get("use_rslora") is True,
        use_dora=raw.get("use_dora") is True,
        init_lora_weights=str(init) if isinstance(init, (str, bool)) else None,
        rank_pattern=_as_number_map(raw.get("rank_pattern"), integer=True),
        alpha_pattern=_as_number_map(raw.get("alpha_pattern"), integer=False),
        target_modules=_as_str_list(raw.get("target_modules")),
        modules_to_save=_as_str_list(raw.get("modules_to_save")),
        trainable_token_indices=isinstance(tti, (list, dict)) and len(tti) > 0,
    )
    for name in ("rank_pattern", "alpha_pattern"):
        given = raw.get(name)
        if isinstance(given, dict) and len(given) != len(getattr(config, name)):
            problems.append(f"{name}: invalid entries ignored")
    return config, problems


def read_adapter_config(adapter_path: Path) -> ConfigReadResult:
    """Find and read adapter_config.json next to *adapter_path*.

    Never raises: a missing config is normal for bare .safetensors files, and a
    bad one is reported in ``problems``.
    """
    candidate = adapter_path.resolve().parent / CONFIG_FILENAME
    try:
        if not candidate.exists():
            return ConfigReadResult()
        st = candidate.stat()
        if not stat.S_ISREG(st.st_mode):
            return ConfigReadResult(problems=[f"{CONFIG_FILENAME} is not a regular file"])
        if st.st_size > _MAX_CONFIG_BYTES:
            return ConfigReadResult(problems=[f"{CONFIG_FILENAME} exceeds {_MAX_CONFIG_BYTES} bytes"])
        raw = json.loads(candidate.read_bytes())
    except (OSError, ValueError, RecursionError) as exc:
        return ConfigReadResult(problems=[f"{CONFIG_FILENAME} unreadable: {type(exc).__name__}"])
    if not isinstance(raw, dict):
        return ConfigReadResult(problems=[f"{CONFIG_FILENAME} is not a JSON object"])
    config, problems = parse_adapter_config(raw)
    return ConfigReadResult(config=config, path_name=CONFIG_FILENAME, problems=problems)


class ScaleResolution(BaseModel):
    """Effective scale of one module and how it was obtained."""

    model_config = ConfigDict(frozen=True)

    scale: float
    rank: int
    alpha: float
    source: str  # "config", "pattern", "default_alpha_equals_r"
    unresolved_pattern: bool = False


def _is_literal(pattern: str) -> bool:
    return not any(ch in _REGEX_METACHARACTERS for ch in pattern)


def _pattern_lookup(patterns: dict[str, Any], module_path: str) -> tuple[Any | None, bool]:
    """Match PEFT pattern keys against a module path without compiling them.

    PEFT matches ``(.*\\.)?(pattern)$``. For a literal pattern that is exactly
    "path equals pattern, or ends with '.' + pattern". Non-literal patterns are
    never executed; they are reported as unresolved.

    Returns:
        (value or None, any_unresolved_pattern_present)
    """
    unresolved = False
    for pattern, value in patterns.items():
        if not _is_literal(pattern):
            unresolved = True
            continue
        if module_path == pattern or module_path.endswith("." + pattern):
            return value, unresolved
    return None, unresolved


def resolve_scale(module_path: str, rank_actual: int, config: AdapterConfig | None) -> ScaleResolution:
    """Effective scale s of ΔW = s·B·A for one module.

    Args:
        module_path: module path from the tensor key (e.g. "...layers.3.self_attn.q_proj").
        rank_actual: rank measured from the tensor shapes (A.shape[0]).
        config: parsed adapter_config.json, or None if absent.

    Returns:
        ScaleResolution. The measured rank is used for the scale because it is
        what the weights actually encode; a mismatch with the declared rank is
        reported separately as a finding.
    """
    rank = max(rank_actual, 1)
    if config is None:
        return ScaleResolution(scale=1.0, rank=rank, alpha=float(rank), source="default_alpha_equals_r")

    alpha_override, unresolved_a = _pattern_lookup(config.alpha_pattern, module_path)
    _, unresolved_r = _pattern_lookup(config.rank_pattern, module_path)
    unresolved = unresolved_a or unresolved_r

    if alpha_override is not None:
        alpha, source = float(alpha_override), "pattern"
    elif config.lora_alpha is not None:
        alpha, source = config.lora_alpha, "config"
    else:
        alpha, source = float(rank), "default_alpha_equals_r"

    scale = alpha / math.sqrt(rank) if config.use_rslora else alpha / rank
    return ScaleResolution(scale=scale, rank=rank, alpha=alpha, source=source, unresolved_pattern=unresolved)
