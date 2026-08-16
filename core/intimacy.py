"""Persona intimacy tiers that tune dynamic body-response sensitivity."""

from __future__ import annotations

PERSONA_INTIMACY_TIERS = (
    "普通亲密",
    "很亲密",
    "非常亲密",
    "对你有很强的身体吸引",
)
DEFAULT_PERSONA_INTIMACY_TIER = "很亲密"

_TIER_MULTIPLIERS = {
    "普通亲密": 0.85,
    "很亲密": 1.0,
    "非常亲密": 1.2,
    "对你有很强的身体吸引": 1.45,
}
_TIER_ALIASES = {
    "ordinary": "普通亲密",
    "close": "很亲密",
    "very_close": "非常亲密",
    "strong_physical_attraction": "对你有很强的身体吸引",
}


def normalize_persona_intimacy_tier(value: object) -> str:
    tier = str(value or "").strip()
    tier = _TIER_ALIASES.get(tier, tier)
    return tier if tier in _TIER_MULTIPLIERS else DEFAULT_PERSONA_INTIMACY_TIER


def persona_intimacy_tier_label(value: object) -> str:
    return normalize_persona_intimacy_tier(value)


def persona_intimacy_multiplier(value: object) -> float:
    return _TIER_MULTIPLIERS[normalize_persona_intimacy_tier(value)]


def body_reaction_stage(body_sensitivity: float, sexual_arousal: float) -> str:
    effective = body_sensitivity * 0.45 + sexual_arousal * 0.55
    if effective >= 0.58:
        return "open_and_receptive"
    if effective >= 0.32:
        return "warmly_receptive"
    if effective >= 0.08:
        return "slightly_aware"
    return "not_noticeable"
