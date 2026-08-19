"""Shared text bounds for persisted and prompt-facing emotion facts."""

from __future__ import annotations

import re

EVENT_FACT_STORAGE_CHARS = 600
EVENT_FACT_REVIEW_CHARS = 360
EVENT_FACT_INJECTION_CHARS = 360

_STRONG_BOUNDARY = re.compile(r"[。！？!?；;](?:[\"'”’」』】）)]*)")
_SOFT_BOUNDARY = re.compile(r"[，,、：:](?:[\"'”’」』】）)]*)")


def bound_complete_text(text: str, max_chars: int) -> str:
    """Bound text without silently leaving a bare mid-clause fragment."""
    clean = str(text or "").strip()
    limit = max(2, int(max_chars))
    if len(clean) <= limit:
        return clean

    window = clean[: limit - 1].rstrip()
    minimum_boundary = min(40, max(1, limit // 3))
    for pattern, strip_chars in (
        (_STRONG_BOUNDARY, ""),
        (_SOFT_BOUNDARY, "，,、：:"),
    ):
        matches = list(pattern.finditer(window))
        if matches and matches[-1].end() >= minimum_boundary:
            prefix = window[: matches[-1].end()].rstrip(strip_chars).rstrip()
            return f"{prefix}…"
    return f"{window}…"
