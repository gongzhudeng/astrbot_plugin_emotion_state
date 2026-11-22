"""重复建项：同一件事只留一条，时间只往后挪。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astrbot_plugin_emotion_state.core.attention import apply_attention_observation
from astrbot_plugin_emotion_state.core.models import AttentionObservation, StateLedger

NOW = datetime.now(timezone.utc)


def _create(**overrides) -> AttentionObservation:
    payload = {
        "action": "create",
        "content": "星期天下午提醒Mando给电动车充电",
        "kind": "remember",
        "confidence": 0.9,
        "explicit": True,
        "source": "attention_review",
        "evidence_quote": "记得星期天下午提醒我给电动车充电",
        "evidence_speaker": "user",
        "due_at": (NOW + timedelta(days=1)).replace(hour=15, minute=0, second=0).isoformat(),
    }
    payload.update(overrides)
    return AttentionObservation(**payload)


def test_same_thing_is_merged_not_duplicated() -> None:
    """2026-11-22 同一件充电事连建三条，就是要拦这个。"""
    ledger, ok, _ = apply_attention_observation(StateLedger(user_key="p"), _create())
    assert ok and len(ledger.attention_items) == 1

    # 换个措辞、换个时间，但仍是同一件事
    again, ok, reason = apply_attention_observation(
        ledger,
        _create(
            content="Mando打算今天下午吃完饭去给电动车充电，我要留意他有没有真的去充",
            due_at=(NOW + timedelta(days=1)).replace(hour=18, minute=0, second=0).isoformat(),
        ),
    )
    assert ok and reason == "merged_into_existing"
    assert len(again.attention_items) == 1


def test_later_time_overwrites() -> None:
    """用户说「待会儿再去」是推迟，时间该更新。"""
    ledger, _, _ = apply_attention_observation(StateLedger(user_key="p"), _create())
    later_due = (NOW + timedelta(days=1)).replace(hour=18, minute=0, second=0).isoformat()
    updated, ok, reason = apply_attention_observation(
        ledger, _create(due_at=later_due, content="傍晚再去给电动车充电")
    )
    assert ok and reason == "merged_into_existing"
    assert updated.attention_items[0].due_at == later_due


def test_earlier_time_does_not_overwrite() -> None:
    """模型手滑填更早的时间，不能把原本对的计划改早。"""
    ledger, _, _ = apply_attention_observation(StateLedger(user_key="p"), _create())
    original_due = ledger.attention_items[0].due_at
    updated, ok, _ = apply_attention_observation(
        ledger,
        _create(
            content="下午给电动车充电去",
            evidence_quote="下午给电动车充电去",
            due_at=(NOW + timedelta(days=1)).replace(hour=9, minute=0, second=0).isoformat(),
        ),
    )
    assert ok
    assert updated.attention_items[0].due_at == original_due


def test_different_things_stay_separate() -> None:
    ledger, _, _ = apply_attention_observation(StateLedger(user_key="p"), _create())
    other, ok, reason = apply_attention_observation(
        ledger, _create(content="提醒我给手机充电", evidence_quote="提醒我给手机充电")
    )
    assert ok and reason != "merged_into_existing"
    assert len(other.attention_items) == 2


def test_different_kind_stays_separate() -> None:
    ledger, _, _ = apply_attention_observation(StateLedger(user_key="p"), _create())
    other, ok, _ = apply_attention_observation(
        ledger,
        _create(kind="plan", content="打算下午去给电动车充电"),
    )
    assert ok
    assert len(other.attention_items) == 2


def test_other_day_stays_separate() -> None:
    ledger, _, _ = apply_attention_observation(StateLedger(user_key="p"), _create())
    other, ok, _ = apply_attention_observation(
        ledger,
        _create(
            content="下周五下午给电动车充电",
            evidence_quote="下周五下午给电动车充电",
            due_at=(NOW + timedelta(days=5)).replace(hour=15, minute=0, second=0).isoformat(),
        ),
    )
    assert ok
    assert len(other.attention_items) == 2
