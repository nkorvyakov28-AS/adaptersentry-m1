"""Recognition of non-LoRA PEFT methods and decomposition inits (parsers/methods.py)."""

from __future__ import annotations

import pytest

from adaptersentry.parsers.methods import is_decomposition_init, recognise_non_lora_method


class TestRecogniseByPeftType:
    @pytest.mark.parametrize("peft_type,expected", [
        ("VERA", "VeRA"), ("vera", "VeRA"), ("IA3", "IA³"), ("LOHA", "LoHa"),
        ("LOKR", "LoKr"), ("BOFT", "BOFT"), ("OFT", "OFT"),
        ("PROMPT_TUNING", "prompt tuning"), ("prefix-tuning", "prefix tuning"),
    ])
    def test_known_methods(self, peft_type: str, expected: str) -> None:
        assert recognise_non_lora_method(peft_type, []) == expected

    def test_lora_and_dora_are_not_non_lora(self) -> None:
        # LoRA/DoRA are analysed, not rejected, so they are never "recognised" here.
        assert recognise_non_lora_method("LORA", []) is None
        assert recognise_non_lora_method("DORA", []) is None

    def test_unknown_peft_type_without_keys(self) -> None:
        assert recognise_non_lora_method("SOMETHING_NEW", []) is None
        assert recognise_non_lora_method(None, []) is None


class TestRecogniseByKeySignature:
    def test_vera_keys_without_config(self) -> None:
        keys = ["base_model.model.vera_A", "base_model.model.model.layers.0.self_attn.q_proj.vera_lambda_b"]
        assert recognise_non_lora_method(None, keys) == "VeRA"

    def test_loha_and_lokr_keys(self) -> None:
        assert recognise_non_lora_method(None, ["x.hada_w1_a", "x.hada_w2_b"]) == "LoHa"
        assert recognise_non_lora_method(None, ["x.lokr_w1", "x.lokr_w2"]) == "LoKr"

    def test_peft_type_wins_over_keys(self) -> None:
        # A declared method is preferred over a key guess.
        assert recognise_non_lora_method("IA3", ["x.vera_lambda_b"]) == "IA³"

    def test_plain_lora_keys_are_not_recognised(self) -> None:
        keys = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
        assert recognise_non_lora_method(None, keys) is None


class TestDecompositionInit:
    @pytest.mark.parametrize("init", ["pissa", "PiSSA", "pissa_niter_4", "olora", "corda", "eva"])
    def test_recognised(self, init: str) -> None:
        assert is_decomposition_init(init) is True

    @pytest.mark.parametrize("init", [None, "", "gaussian", "true", "loftq"])
    def test_not_recognised(self, init: str | None) -> None:
        assert is_decomposition_init(init) is False
