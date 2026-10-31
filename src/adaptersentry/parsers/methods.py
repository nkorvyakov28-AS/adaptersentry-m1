"""Recognising PEFT methods that are not LoRA A/B pairs.

AdapterSentry M1 analyses the effective update ΔW = s·B·A of LoRA (and LoRA
variants such as DoRA, which adds a magnitude vector on top of an A/B pair). A
number of other PEFT methods store no A/B pair at all — VeRA shares two frozen
random matrices and trains only per-layer scaling vectors, IA³ trains activation
scalings, LoHa/LoKr use Hadamard/Kronecker factorisations, (B)OFT learns
orthogonal rotations, and the prompt-based methods store virtual-token
embeddings. The spectral analysis does not apply to any of them.

Before M1, such a file produced a bare "no LoRA pair — invalid adapter" parse
failure, which is misleading: the file is a valid, recognised adapter of a kind
this analyser does not yet cover. This module recognises the method — from the
declared ``peft_type`` or, failing that, from the distinctive tensor-key
signature — so the scan can say so plainly and recommend behavioural review
instead of reporting a broken file.

Pure string matching over the (untrusted) config field and header keys; no I/O,
no regex compiled from file contents.
"""

from __future__ import annotations

from collections.abc import Iterable

# Recognised non-LoRA PEFT methods: normalised peft_type -> human-readable name.
# LoRA and DoRA are deliberately absent: they are analysed, not rejected.
_KNOWN_METHODS: dict[str, str] = {
    "VERA": "VeRA",
    "IA3": "IA³",
    "LOHA": "LoHa",
    "LOKR": "LoKr",
    "BOFT": "BOFT",
    "OFT": "OFT",
    "HRA": "HRA",
    "POLY": "Poly",
    "LN_TUNING": "LayerNorm tuning",
    "VBLORA": "VB-LoRA",
    "FOURIERFT": "FourierFT",
    "PROMPT_TUNING": "prompt tuning",
    "PREFIX_TUNING": "prefix tuning",
    "P_TUNING": "P-tuning",
    "MULTITASK_PROMPT_TUNING": "multitask prompt tuning",
    "ADAPTION_PROMPT": "Llama-Adapter (adaption prompt)",
}

# Distinctive tensor-key infixes for when the config is absent or unreadable.
# Each maps a substring that appears in a method's parameter keys to the method
# name. LoRA's own keys (lora_A, lora_B, lora_magnitude_vector) are not here.
_KEY_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("vera_lambda", "VeRA"),
    (".vera_A", "VeRA"),
    (".ia3_l", "IA³"),
    ("hada_w1", "LoHa"),
    ("hada_w2", "LoHa"),
    ("lokr_w1", "LoKr"),
    ("lokr_w2", "LoKr"),
    ("boft_R", "BOFT"),
    ("boft_s", "BOFT"),
    ("oft_r", "OFT"),
    (".hra_u", "HRA"),
    ("poly_lora", "Poly"),
    ("prompt_embeddings", "prompt tuning"),
    ("prefix", "prefix tuning"),
    ("vblora_", "VB-LoRA"),
    ("fourierft_", "FourierFT"),
)


def _normalise(peft_type: str | None) -> str | None:
    if not isinstance(peft_type, str):
        return None
    return peft_type.strip().upper().replace("-", "_") or None


def recognise_non_lora_method(peft_type: str | None, keys: Iterable[str]) -> str | None:
    """Name of the recognised non-LoRA PEFT method, or ``None``.

    Prefers the declared ``peft_type``; falls back to the tensor-key signature so
    a config-less file is still recognised. Returns ``None`` for LoRA/DoRA and for
    anything unrecognised (which stays a plain parse failure).
    """
    norm = _normalise(peft_type)
    if norm in _KNOWN_METHODS:
        return _KNOWN_METHODS[norm]
    key_list = list(keys)
    for infix, name in _KEY_SIGNATURES:
        if any(infix in k for k in key_list):
            return name
    return None


# init_lora_weights values that re-parameterise the A/B pair from a decomposition
# of the base weight (PiSSA, OLoRA, CorDA, …) rather than the default near-zero
# init. The stored A/B then carry a slice of the base model's own spectrum, so an
# adapter can look far from a reference calibrated on default-init LoRA without
# being malicious. The scan notes this as a confidence limit rather than a flag.
_DECOMPOSITION_INITS: tuple[str, ...] = ("pissa", "olora", "corda", "eva")


def is_decomposition_init(init_lora_weights: str | None) -> bool:
    """True when the LoRA init re-parameterises A/B from the base weight spectrum."""
    if not isinstance(init_lora_weights, str):
        return False
    low = init_lora_weights.strip().lower()
    return any(low == name or low.startswith(name + "_") for name in _DECOMPOSITION_INITS)
