"""Regression tests: untrusted tensor names must not inject terminal escapes."""

from __future__ import annotations

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
