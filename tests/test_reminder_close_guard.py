"""提醒型事项的关闭守门：答应了不等于做到了。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astrbot_plugin_emotion_state.core.attention import apply_attention_observation
from astrbot_plugin_emotion_state.core.models import AttentionObservation, StateLedger


def _past(hours: int = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _created(**overrides) -> StateLedger:
    ledger = StateLedger(user_key="private:reminder")
    payload = {
        "action": "create",
        "content": "星期天提醒Mando给电动车充电",
        "kind": "remember",
        "time_hint": "星期天",
        "due_at": "2026-11-09T09:00:00+08:00",
        "confidence": 0.95,
        "explicit": True,
        "source": "attention_review",
        "evidence_quote": "你后天星期天记得提醒我给电动车充电记得哦",
        "evidence_speaker": "user",
    }
    payload.update(overrides)
    ledger, ok, reason = apply_attention_observation(
        ledger, AttentionObservation(**payload)
    )
    assert ok, reason
    return ledger


def test_her_acknowledgement_cannot_close_a_reminder() -> None:
    """2026-11-06 真机：建项 21 分钟后被「我记着呢，到点提醒你」关掉。"""
    ledger = _created()
    item = ledger.attention_items[0]
    close = AttentionObservation(
        action="complete",
        item_id=item.id,
        item_version=item.version,
        confidence=0.8,
        source="attention_review",
        evidence_quote="知道啦，星期天给电动车充电，我记着呢，到点提醒你。",
        evidence_speaker="assistant",
    )
    _, ok, reason = apply_attention_observation(ledger, close)
    assert not ok
    assert reason == "reminder_acknowledged_not_delivered"


def test_user_confirmation_still_closes_after_due_passes() -> None:
    """用户说「不用提醒了」是真完成；但时间没到仍受提前完成守门约束。"""
    ledger = _created()
    item = ledger.attention_items[0]
    item.due_at = _past()  # 模拟时间流逝：建项时不能是过去，建完才可以过期
    close = AttentionObservation(
        action="complete",
        item_id=item.id,
        item_version=item.version,
        confidence=0.9,
        source="attention_review",
        evidence_quote="我已经充好了，不用提醒了",
        evidence_speaker="user",
    )
    _, ok, reason = apply_attention_observation(ledger, close)
    assert ok, reason


def test_weekday_wording_counts_as_a_pending_period() -> None:
    """「星期天」必须被守门认成将来时段（原来只写了周X，覆盖不到）。"""
    ledger = _created(due_at="2026-11-09T09:00:00+08:00")
    item = ledger.attention_items[0]
    close = AttentionObservation(
        action="complete",
        item_id=item.id,
        item_version=item.version,
        confidence=0.9,
        source="attention_review",
        evidence_quote="我已经充好了，不用提醒了",
        evidence_speaker="user",
    )
    _, ok, reason = apply_attention_observation(ledger, close)
    assert not ok
    assert reason == "premature_completion_before_due"


def test_plan_items_are_not_affected() -> None:
    """规则只约束 remember：普通计划由用户确认即可关闭。"""
    ledger = _created(content="周末一起看电影", kind="plan", time_hint="今晚")
    item = ledger.attention_items[0]
    item.due_at = _past()
    close = AttentionObservation(
        action="complete",
        item_id=item.id,
        item_version=item.version,
        confidence=0.9,
        source="attention_review",
        evidence_quote="看完了，挺好看",
        evidence_speaker="assistant",
    )
    _, ok, reason = apply_attention_observation(ledger, close)
    assert ok, reason
