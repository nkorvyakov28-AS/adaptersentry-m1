"""Tensor-key parsing: which LoRA pair, layer, module and expert a key belongs to.

Keys come from the untrusted safetensors header. Parsing uses fixed, linear-time
regular expressions only; nothing from the file is ever compiled as a pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# PEFT LoRA: "<base>.lora_A.weight", optionally with an adapter name segment
# ("<base>.lora_A.default.weight") as written by some PEFT versions.
_LORA_A = re.compile(r"^(?P<base>.+)\.lora_A(?:\.[A-Za-z0-9_\-]+)?\.weight$")
_LORA_B = re.compile(r"^(?P<base>.+)\.lora_B(?:\.[A-Za-z0-9_\-]+)?\.weight$")
# Embedding LoRA stores bare parameters (no ".weight").
_EMB_A = re.compile(r"^(?P<base>.+)\.lora_embedding_A(?:\.[A-Za-z0-9_\-]+)?$")
_EMB_B = re.compile(r"^(?P<base>.+)\.lora_embedding_B(?:\.[A-Za-z0-9_\-]+)?$")
_DORA = re.compile(r"^(?P<base>.+)\.lora_magnitude_vector(?:\.[A-Za-z0-9_\-]+)?(?:\.weight)?$")
_TRAINABLE_TOKENS = re.compile(r"^(?P<base>.+)\.trainable_tokens_(?:delta|original)(?:\.[A-Za-z0-9_\-]+)?$")

_LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)(?:\.|$)")
_EXPERT = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")

ModuleKind = Literal["attention", "mlp", "embedding", "lm_head", "router", "other"]
KeyRole = Literal[
    "lora_A", "lora_B", "embedding_A", "embedding_B", "dora_magnitude", "trainable_tokens", "other",
]

_ATTENTION = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query_key_value", "c_attn",
    "Wqkv", "out_proj", "query", "key", "value", "wq", "wk", "wv", "wo",
})
_MLP = frozenset({
    "gate_proj", "up_proj", "down_proj", "gate_up_proj", "fc1", "fc2", "w1", "w2", "w3",
    "c_fc", "c_proj", "dense_h_to_4h", "dense_4h_to_h",
})
_EMBEDDING = frozenset({"embed_tokens", "wte", "word_embeddings", "tok_embeddings"})
_LM_HEAD = frozenset({"lm_head", "output", "embed_out"})
_ROUTER = frozenset({"router", "gate"})  # "gate" alone is the MoE router in Qwen/Mixtral


@dataclass(frozen=True)
class ParsedKey:
    """Classification of one tensor key."""

    role: KeyRole
    base: str | None  # module path the tensor belongs to (None for role "other")


@dataclass(frozen=True)
class ModuleLocation:
    """Where a module sits in the model."""

    layer: int | None
    expert: int | None
    module: str
    kind: ModuleKind


def parse_key(key: str) -> ParsedKey:
    """Classify a safetensors key by its role in a PEFT adapter."""
    for role, pattern in (
        ("lora_A", _LORA_A),
        ("lora_B", _LORA_B),
        ("embedding_A", _EMB_A),
        ("embedding_B", _EMB_B),
        ("dora_magnitude", _DORA),
        ("trainable_tokens", _TRAINABLE_TOKENS),
    ):
        m = pattern.match(key)
        if m:
            return ParsedKey(role=role, base=m.group("base"))  # type: ignore[arg-type]
    return ParsedKey(role="other", base=None)


def module_name(base: str) -> str:
    """Last meaningful component of a module path ("...self_attn.q_proj" → "q_proj")."""
    parts = [p for p in base.split(".") if p and not p.isdigit()]
    return parts[-1] if parts else base


def locate(base: str) -> ModuleLocation:
    """Layer index, expert index, module name and kind for a module path."""
    layer_m = _LAYER.search(base)
    expert_m = _EXPERT.search(base)
    name = module_name(base)
    if name in _ATTENTION:
        kind: ModuleKind = "attention"
    elif name in _MLP:
        kind = "mlp"
    elif name in _EMBEDDING:
        kind = "embedding"
    elif name in _LM_HEAD:
        kind = "lm_head"
    elif name in _ROUTER and expert_m is None:
        kind = "router"
    else:
        kind = "other"
    return ModuleLocation(
        layer=int(layer_m.group(1)) if layer_m else None,
        expert=int(expert_m.group(1)) if expert_m else None,
        module=name,
        kind=kind,
    )
