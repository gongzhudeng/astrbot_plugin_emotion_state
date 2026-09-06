"""Reply-suggestion (expression guidance) gating, prompting, and formatting.

Guidance is generated asynchronously by a model only when the current state is
worth expressing; between regenerations the cached block is reused verbatim so
the expressed tone stays slow and coherent, like a real person's mood.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from .models import StateLedger, clamp
from .settlement import effective_mood, event_score, interpersonal_unresolved

GUIDANCE_BLOCK_START = "<emotion_state_guidance>"
GUIDANCE_BLOCK_END = "</emotion_state_guidance>"

NEUTRAL_LABELS = {"平静", "温和愉快"}


def should_refresh_guidance(
    *,
    has_cache: bool,
    cache_age_minutes: float,
    regime_changed: bool,
    gate_open: bool,
    min_interval_minutes: float,
    max_age_hours: float,
) -> bool:
    """Decide whether the cached reply suggestion should be regenerated.

    Edge-triggered: reaching a state worth expressing generates once and the
    result covers the whole episode. ``max_age_hours`` is a bounded freshness
    guard so a suggestion never outlives its context (e.g. a late-night note
    lingering into the next morning); ``min_interval_minutes`` only debounces
    rapid state flapping.
    """
    if not gate_open:
        return False
    if not has_cache:
        return True
    max_age_minutes = max(0.0, float(max_age_hours)) * 60.0
    if max_age_minutes > 0 and cache_age_minutes >= max_age_minutes:
        return True
    return regime_changed and cache_age_minutes >= max(0.0, float(min_interval_minutes))


def time_band(now: datetime) -> str:
    """Return a coarse human-readable time band for prompts."""
    hour = now.astimezone().hour
    if 5 <= hour < 8:
        return "清晨"
    if 8 <= hour < 12:
        return "上午"
    if 12 <= hour < 14:
        return "中午"
    if 14 <= hour < 18:
        return "下午"
    if 18 <= hour < 22:
        return "晚上"
    if 22 <= hour or hour < 2:
        return "深夜"
    return "凌晨"


def needs_guidance(
    ledger: StateLedger,
    *,
    night_hours: tuple[int, ...] | frozenset[int] = (),
    now: datetime | None = None,
) -> bool:
    """True when the current state is strong enough to deserve a suggestion block."""
    current = (now or datetime.now().astimezone()).astimezone()
    effective = effective_mood(
        ledger,
        hour=current.hour,
        night_hours=night_hours,
    )
    if effective.label not in NEUTRAL_LABELS:
        return True
    if any(event.lifecycle == "intensified" for event in ledger.events):
        return True
    if ledger.jealousy.intensity >= 0.08 and ledger.jealousy.confidence >= 0.55:
        return True
    if any(
        event.source == "night_missing" and event.lifecycle in {"active", "intensified"}
        for event in ledger.events
    ):
        return True
    return False


def guidance_regime(
    ledger: StateLedger,
    *,
    night_hours: tuple[int, ...] | frozenset[int] = (),
    now: datetime | None = None,
) -> str:
    """Compact fingerprint of the expressed state used to detect real changes."""
    current = (now or datetime.now().astimezone()).astimezone()
    effective = effective_mood(
        ledger,
        hour=current.hour,
        night_hours=night_hours,
    )
    top_event = max(
        (
            event
            for event in ledger.events
            if event.lifecycle in {"active", "intensified"}
            and event.category != "transient"
        ),
        key=event_score,
        default=None,
    )
    jealousy_tier = (
        "jealous"
        if ledger.jealousy.intensity >= 0.08 and ledger.jealousy.confidence >= 0.55
        else ""
    )
    missing = (
        "missing"
        if any(
            event.source == "night_missing"
            and event.lifecycle in {"active", "intensified"}
            for event in ledger.events
        )
        else ""
    )
    return "|".join(
        [
            effective.label,
            # No time band here on purpose: the clock alone must never trigger a
            # regeneration. Only a real state change (label, new top event,
            # jealousy/missing, temperament rollover) refreshes the suggestion.
            ledger.today_temperament.word,
            top_event.id if top_event else "",
            jealousy_tier,
            missing,
            "unresolved" if interpersonal_unresolved(ledger) else "",
        ]
    )


def build_guidance_prompt(
    ledger: StateLedger,
    *,
    now: datetime,
    night_hours: tuple[int, ...] | frozenset[int] = (),
    style_hint: str = "",
    previous: dict[str, Any] | None = None,
    max_chars: int = 200,
) -> str:
    effective = effective_mood(
        ledger,
        hour=now.astimezone().hour,
        night_hours=night_hours,
    )
    top_events = [
        {
            "fact": event.fact,
            "meaning": event.emotional_meaning,
            "valence": round(event.valence, 2),
            "intensity": round(event.intensity, 2),
        }
        for event in sorted(
            (
                item
                for item in ledger.events
                if item.lifecycle in {"active", "intensified"}
                and item.category != "transient"
            ),
            key=event_score,
            reverse=True,
        )[:3]
    ]
    payload = {
        "time_band": time_band(now),
        "baseline_label": ledger.mood.label,
        "effective_label": effective.label,
        "baseline_valence": round(ledger.mood.valence, 2),
        "short_term_offset": round(ledger.mood_offset.valence, 2),
        "today_temperament": ledger.today_temperament.word,
        "top_events": top_events,
        "jealousy_intensity": round(ledger.jealousy.intensity, 2)
        if ledger.jealousy.confidence >= 0.55
        else 0.0,
        "previous_tone": str((previous or {}).get("tone", "")),
    }
    style = str(style_hint or "").strip()[:800]
    style_line = (
        f"\n该人格表达各种情绪的方式（必须严格遵守，建议内容不得与其冲突）：{style}\n"
        if style
        else ""
    )
    return (
        "你是角色的内心声音。请根据以下内心状态材料，为角色的下一次回复生成简短的表达建议，"
        "只返回 JSON 对象，字段为 tone、can_say、avoid。"
        "\n- tone：一句话描述此刻的表达基调（语气、节奏、能量）。"
        "\n- can_say：可以自然流露什么；情绪明显偏离平静时必须点出具体原因"
        "（引用 top_events 中的某件事实），并给出表达方向（如说出来、撒娇、变安静等）。"
        "\n- avoid：此刻应避免的表达（不超过两点）。"
        "\n- 平静或温和愉快且没有强烈心事时，建议应该非常轻，甚至可以说“按人格正常聊即可”。"
        "\n- 只提供方向、原因和分寸，不要写具体台词或原话，也不要照抄材料里的句子；"
        "具体怎么说由角色的人格和说话习惯自己决定。"
        f"\n- 每个字段不超过 {max(20, int(max_chars))} 字，宁短勿长。"
        "\n- 不要编造材料里没有的事实。"
        + style_line
        + f"材料：{json.dumps(payload, ensure_ascii=False)}"
    )


def _clean_guidance_field(value: Any, max_chars: int) -> str:
    """Normalize one guidance field: accept lists, dedupe punctuation, truncate."""
    if isinstance(value, (list, tuple)):
        value = "；".join(str(item).strip() for item in value if str(item).strip())
    clean = re.sub(r"(。)\1+", r"\1", str(value or "").strip())
    return clean[: max(20, int(max_chars))]


def parse_guidance_response(text: str, max_chars: int = 200) -> dict[str, str]:
    """Parse and validate one guidance response; raises ValueError.

    Args:
        text: Raw model output (JSON object expected).
        max_chars: Configured cap applied to each field.
    """
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("guidance response is not an object")
    tone = _clean_guidance_field(data.get("tone", ""), max_chars)
    can_say = _clean_guidance_field(data.get("can_say", ""), max_chars)
    avoid = _clean_guidance_field(data.get("avoid", ""), max_chars)
    if not tone:
        raise ValueError("guidance response missing tone")
    return {"tone": tone, "can_say": can_say, "avoid": avoid}


_FALLBACK_TONES = {
    "低落": "语气放轻放慢，可以自然说出心情不好和原因，可以想要安慰",
    "紧绷难受": "语气有点绷着，不硬撑，被问到才慢慢说原因",
    "有些在意": "平静里带着一点在意，回复比平时短一点",
    "夜晚感伤": "夜深了，语气比白天软、慢，带一点黏人和低落，可以说出在想什么",
    "委屈": "有点委屈但忍着不全说，回复变短，被关心到再说原因",
    "明快开心": "语气明快活泼，话多一点，可以主动分享开心的原因",
    "雀跃": "明显雀跃，话变多，忍不住想分享好事",
    "略有不安": "语气里带一点不安，可以小声说出在担心什么",
    "温和愉快": "保持自然温和即可，偶尔流露开心",
    "平静": "按人格正常聊即可，不需要刻意表达情绪",
}


def local_guidance_fallback(ledger: StateLedger, now: datetime) -> dict[str, str]:
    """Deterministic one-line fallback used only before the first model guidance."""
    effective = effective_mood(ledger, hour=now.astimezone().hour)
    tone = _FALLBACK_TONES.get(
        effective.label, f"按当前心境（{effective.label}）自然表达"
    )
    reason_event = next(
        (
            event
            for event in ledger.events
            if event.lifecycle in {"active", "intensified"}
            and event.category != "transient"
        ),
        None,
    )
    can_say = (
        f"可以说出此刻的感受是因为：{reason_event.fact[:80]}"
        if reason_event and effective.label not in NEUTRAL_LABELS
        else ""
    )
    return {"tone": tone, "can_say": can_say, "avoid": ""}


def format_guidance_block(
    guidance: dict[str, Any] | None,
    *,
    strength: float = 1.0,
) -> str:
    """Format the cached guidance as an injection block, scaled by strength."""
    if not guidance:
        return ""
    bounded = clamp(float(strength), 0.0, 2.0)
    if bounded <= 0.0:
        return ""
    tone = str(guidance.get("tone", "")).strip()
    can_say = str(guidance.get("can_say", "")).strip()
    avoid = str(guidance.get("avoid", "")).strip()
    if not tone:
        return ""
    lines = [GUIDANCE_BLOCK_START]
    if bounded < 0.5:
        lines.append(f"当前表达基调：{tone}")
    else:
        lines.append(f"当前表达基调：{tone}。")
        if can_say:
            lines.append(f"可以自然流露：{can_say}。")
        if avoid:
            lines.append(f"避免：{avoid}。")
    lines.append(GUIDANCE_BLOCK_END)
    return "\n".join(lines)
