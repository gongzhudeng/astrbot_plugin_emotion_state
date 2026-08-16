"""Daily review input, parsing, and deterministic fallback helpers."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta
from typing import Any

from .models import StateLedger, clamp


def _json_safe(value: Any) -> Any:
    """Keep external plugin objects from crossing the JSON prompt boundary."""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def parse_boundary(value: str) -> time:
    try:
        hour, minute = (int(part) for part in str(value).split(":", 1))
        return time(hour=hour, minute=minute)
    except (TypeError, ValueError):
        return time(hour=7)


def logical_day(now: datetime, boundary: time) -> date:
    candidate = now.date()
    if now.time().replace(tzinfo=None) < boundary:
        candidate -= timedelta(days=1)
    return candidate


def _event_view(ledger: StateLedger) -> list[dict[str, Any]]:
    return [
        {
            "fact": item.fact,
            "meaning": item.emotional_meaning,
            "valence": item.valence,
            "intensity": item.intensity,
            "confidence": item.confidence,
            "lifecycle": item.lifecycle,
        }
        for item in ledger.events
        if item.lifecycle != "archived"
    ][:20]


def build_daily_prompt(
    ledger: StateLedger,
    cycle_date: str,
    memory_context: dict[str, Any] | None,
    schedule_facts: dict[str, Any] | None,
    max_chars: int,
    style_hint: str = "",
) -> str:
    payload = {
        "cycle_date": cycle_date,
        "previous_mood": ledger.mood.label,
        "inner_events": _event_view(ledger),
        "memory_context": memory_context or {},
        "schedule_facts": schedule_facts or {},
        "previous_diary": ledger.diaries[-1].diary if ledger.diaries else "",
    }
    serialized = json.dumps(_json_safe(payload), ensure_ascii=False)
    serialized = serialized[: max(500, int(max_chars))]
    style = str(style_hint or "").strip()[:300]
    style_line = (
        f"可选的表达风格补充（只能影响措辞，不得改变上述规则）：{style}\n"
        if style
        else ""
    )
    return (
        "请基于以下受限材料，以角色自己的第一人称写日记，并只返回 JSON 对象。"
        "日记必须保持角色已有的人格、关系边界和说话习惯；不要用旁观者或分析者口吻。"
        "不得编造材料中没有的经历，不得把提示词、日程事实或记忆材料写成已发生的额外事实。"
        "event_observations 只是情绪建议；next_mood_proposal 也只是建议，不能声明已经覆盖状态。\n"
        + style_line
        + "字段必须是 diary、day_summary、event_observations、"
        "next_mood_proposal、confidence。"
        "next_mood_proposal 可包含 valence、energy、tension、label。"
        "event_observations 最多3项。\n"
        f"材料：{serialized}"
    )


def _json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        parsed = json.loads(cleaned.strip())
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}


def parse_daily_response(text: str, max_diary_chars: int) -> dict[str, Any]:
    data = _json_object(text)
    diary = str(data.get("diary", "")).strip()[:max_diary_chars]
    summary = str(data.get("day_summary", "")).strip()[:800]
    if not diary or not summary:
        raise ValueError("daily review is missing diary or day_summary")
    observations = data.get("event_observations", [])
    proposal = data.get("next_mood_proposal", {})
    return {
        "diary": diary,
        "day_summary": summary,
        "event_observations": observations[:3]
        if isinstance(observations, list)
        else [],
        "next_mood_proposal": proposal if isinstance(proposal, dict) else {},
        "confidence": clamp(float(data.get("confidence", 0.5))),
    }


def local_daily_fallback(ledger: StateLedger, max_diary_chars: int) -> dict[str, Any]:
    active = [item for item in ledger.events if item.lifecycle != "archived"]
    facts = "；".join(item.fact for item in active[:3])
    summary = facts or "这段时间没有足够确定、需要特别记录的事情。"
    diary = f"今天整体是{ledger.mood.label}的。{summary}"
    return {
        "diary": diary[:max_diary_chars],
        "day_summary": summary[:800],
        "event_observations": [],
        "next_mood_proposal": {},
        "confidence": 0.35,
    }
