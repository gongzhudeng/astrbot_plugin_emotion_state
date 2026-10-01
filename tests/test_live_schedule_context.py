"""Live low-sensitivity schedule context for background correction."""

from __future__ import annotations

from datetime import datetime, timezone

from astrbot_plugin_emotion_state.core.injector import (
    InjectionOptions,
    build_live_schedule_context,
)
from astrbot_plugin_emotion_state.core.models import (
    AttentionItem,
    MoodState,
    StateLedger,
    TemperamentState,
)

NOON = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _ledger(**kwargs) -> StateLedger:
    return StateLedger(user_key="private:live", **kwargs)


def test_live_context_contains_mood_and_attention() -> None:
    ledger = _ledger(
        mood=MoodState(label="温和愉快", energy=0.8, tension=0.2),
        today_temperament=TemperamentState(word="慵懒"),
        attention_items=[
            AttentionItem(content="下午记得去天台拍照"),
        ],
    )
    result = build_live_schedule_context(ledger, 5, options=InjectionOptions(now=NOON))

    assert "温和愉快" in result["mood"]
    assert "能量较高" in result["mood"]
    assert "慵懒" in result["mood"]
    assert "下午记得去天台拍照" in result["attention"]
    assert "仍待关注" in result["attention"]


def test_live_context_never_contains_private_state() -> None:
    ledger = _ledger(
        mood=MoodState(label="温和愉快"),
        attention_items=[AttentionItem(content="下午记得去天台拍照")],
    )
    result = build_live_schedule_context(ledger, 5, options=InjectionOptions(now=NOON))
    joined = f"{result['mood']}{result['attention']}"
    for private_marker in ("亲密", "吃醋", "身体敏感", "占有欲", "性唤起"):
        assert private_marker not in joined


def test_live_context_respects_attention_limit() -> None:
    ledger = _ledger(
        mood=MoodState(label="平静"),
        attention_items=[
            AttentionItem(content=f"事项{i}") for i in range(4)
        ],
    )
    result = build_live_schedule_context(ledger, 2, options=InjectionOptions(now=NOON))
    assert result["attention"].count("- [") == 2


def test_live_context_empty_ledger_returns_empty_strings() -> None:
    result = build_live_schedule_context(
        _ledger(mood=MoodState(label="")), 5, options=InjectionOptions(now=NOON)
    )
    assert result == {"mood": "", "attention": ""}
