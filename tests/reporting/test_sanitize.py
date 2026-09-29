"""Regression tests: untrusted tensor names must not inject terminal escapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from adaptersentry.analyzer import scan
from adaptersentry.reporting.human_summary import _shorten_layer
from adaptersentry.reporting.sanitize import safe_text


class TestSafeText:
    def test_plain_text_unchanged(self) -> None:
        assert safe_text("model.layers.3.self_attn.q_proj") == "model.layers.3.self_attn.q_proj"

    def test_non_ascii_letters_unchanged(self) -> None:
        assert safe_text("слой.3") == "слой.3"

    def test_ansi_escape_neutralised(self) -> None:
        out = safe_text("x\x1b[2K\rRisk: LOW")
        assert "\x1b" not in out and "\r" not in out
        assert "\\x1b" in out and "\\x0d" in out

    def test_newline_and_c1_neutralised(self) -> None:
        out = safe_text("a\nb\x9bc")
        assert "\n" not in out and "\x9b" not in out

    def test_bidi_override_neutralised(self) -> None:
        out = safe_text("lora‮Agnp.")
        assert "‮" not in out and "\\u202e" in out

    def test_shorten_layer_sanitises(self) -> None:
        assert "\x1b" not in _shorten_layer("model.layers.1.\x1b]0;pwned\x07.q_proj")


class TestEndToEnd:
    def test_malicious_tensor_name_is_escaped_in_text_output(self, tmp_path: Path) -> None:
        from adaptersentry.reporters import text as text_reporter
        from adaptersentry.reporting.human_summary import render_human_summary

        rng = np.random.default_rng(0)
        evil = "model.layers.0.\x1b[2J\x1b[H.q_proj"
        tensors = {}
        for name in (evil, "model.layers.1.self_attn.q_proj", "model.layers.2.self_attn.q_proj"):
            a = (rng.standard_normal((4, 64)) * 0.01).astype(np.float32)
            if name == evil:
                a[0, :3] = 50.0  # heavy tail: the layer is flagged, so its name is printed
            tensors[f"{name}.lora_A.weight"] = a
            tensors[f"{name}.lora_B.weight"] = rng.standard_normal((64, 4)).astype(np.float32)
        path = tmp_path / "evil.safetensors"
        save_file(tensors, str(path), metadata={"r": "4"})

        report = scan(path)
        assert any(evil in f.affected_layers for f in report.findings)
        rendered = text_reporter.render(report, no_color=True) + render_human_summary(
            report, verbose=True, no_color=True,
        )
        assert "\x1b" not in rendered
        assert "\\x1b[2J" in rendered
