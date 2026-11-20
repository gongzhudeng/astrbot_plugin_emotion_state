"""主动提醒：到点该开口的判定，以及提醒后的静默。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astrbot_plugin_emotion_state.core.attention import (
    due_reminder_candidates,
    mark_reminded,
    should_inject,
)
from astrbot_plugin_emotion_state.core.models import AttentionItem, StateLedger

NOW = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)


def _item(**overrides) -> AttentionItem:
    payload = {
        "content": "提醒我给电动车充电",
        "kind": "remember",
        "status": "open",
        "explicit": True,
        "confidence": 0.9,
        "due_at": (NOW - timedelta(minutes=1)).isoformat(),
    }
    payload.update(overrides)
    return AttentionItem(**payload)


def test_due_item_is_picked() -> None:
    items = [_item()]
    picked = due_reminder_candidates(items, NOW)
    assert [row.id for row in picked] == [items[0].id]


def test_future_item_is_not_picked() -> None:
    items = [_item(due_at=(NOW + timedelta(hours=1)).isoformat())]
    assert due_reminder_candidates(items, NOW) == []


def test_non_remember_kinds_are_excluded_by_default() -> None:
    items = [_item(kind="commitment"), _item(kind="plan"), _item(kind="care")]
    assert due_reminder_candidates(items, NOW) == []
    # 显式放开才收
    picked = due_reminder_candidates(items, NOW, kinds=("commitment", "plan"))
    assert len(picked) == 2


def test_notify_count_caps_repeats() -> None:
    items = [_item(notify_count=2)]
    assert due_reminder_candidates(items, NOW, max_count=2) == []
    # 放宽上限就还能提
    assert len(due_reminder_candidates(items, NOW, max_count=3)) == 1


def test_interval_gates_repeats() -> None:
    just_now = (NOW - timedelta(minutes=5)).isoformat()
    items = [_item(notified_at=just_now, notify_count=1)]
    assert due_reminder_candidates(items, NOW, interval_minutes=30) == []
    # 过了间隔就再提
    assert len(due_reminder_candidates(items, NOW, interval_minutes=1)) == 1


def test_done_or_unconfirmed_items_are_skipped() -> None:
    assert due_reminder_candidates([_item(status="completed")], NOW) == []
    assert due_reminder_candidates([_item(status="proposed")], NOW) == []
    assert due_reminder_candidates([_item(explicit=False)], NOW) == []
    assert due_reminder_candidates([_item(confidence=0.3)], NOW) == []


def test_mark_reminded_records_without_touching_status() -> None:
    ledger = StateLedger(user_key="private:remind")
    ledger.attention_items.append(_item())
    before = ledger.attention_items[0]

    updated, changed = mark_reminded(ledger, before.id, NOW)

    assert changed
    after = updated.attention_items[0]
    assert after.status == "open"
    assert after.notified_at
    assert after.notify_count == 1
    assert after.version == before.version + 1
    # 原账本没被改（服务层靠 to_dict 比较判断要不要落盘）
    assert ledger.attention_items[0].notify_count == 0


def test_mark_reminded_is_noop_for_unknown_id() -> None:
    ledger = StateLedger(user_key="private:remind")
    _, changed = mark_reminded(ledger, "nope", NOW)
    assert not changed


def test_just_reminded_item_stops_injecting() -> None:
    """提醒过的 30 分钟内不再挂进上下文，否则她刚说完又会说一遍。"""
    item = _item(notified_at=(NOW - timedelta(minutes=5)).isoformat(), notify_count=1)
    assert not should_inject(item, NOW)
    # 过了静默期又回到正常判断（这条已逾期，所以会注入）
    later = NOW + timedelta(minutes=40)
    assert should_inject(item, later)


def test_user_mention_overrides_reminder_silence() -> None:
    """他主动问起时必须能答——静默不能把她变成失忆。"""
    item = _item(notified_at=(NOW - timedelta(minutes=5)).isoformat(), notify_count=1)
    assert not should_inject(item, NOW)
    assert should_inject(item, NOW, user_message="电动车充上了")
