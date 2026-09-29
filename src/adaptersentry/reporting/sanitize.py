"""Neutralise untrusted strings before they reach a terminal or log line.

Tensor names, metadata values and file paths come from the adapter file
itself. Printed raw, an ANSI/OSC escape sequence or newline in a tensor name
can draw a fake "Risk: LOW" line, rewrite the terminal title or split a log
record; bidirectional-override characters can reorder what the reader sees.
JSON and SARIF output are already safe (json.dumps escapes control
characters); this module is for human-readable text.
"""

from __future__ import annotations

import unicodedata

# Bidirectional embedding/override/isolate controls (Trojan Source class).
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩‎‏")


def safe_text(value: object) -> str:
    """Return *value* as text with control and bidi characters escaped.

    Control characters (Unicode category Cc, which covers C0, DEL and C1) and
    bidi controls are replaced by a visible ``\\xNN`` / ``\\uNNNN`` escape.
    Printable text, including non-ASCII letters, is left unchanged.

    Args:
        value: Any object; it is converted with str() first.

    Returns:
        A string safe to embed in terminal output or a log message.
    """
    text = str(value)
    if text.isprintable():
        return text
    out: list[str] = []
    for ch in text:
        if ch in _BIDI_CONTROLS or unicodedata.category(ch) == "Cc":
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)
