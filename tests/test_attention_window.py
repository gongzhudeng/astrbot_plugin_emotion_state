"""注入时机：窗口 / 提及 / 逾期 / care 四条通道。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astrbot_plugin_emotion_state.core.attention import (
    extract_mention_keywords,
    is_mentioned,
    select_attention_items,
    should_inject,
)
from astrbot_plugin_emotion_state.core.models import AttentionItem

NOW = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)


def _item(content: str, **kwargs) -> AttentionItem:
    defaults = {"status": "open", "explicit": True, "confidence": 0.9}
    defaults.update(kwargs)
    return AttentionItem(content=content, **defaults)


def test_keywords_extract_nouns_not_verbs_or_time() -> None:
    words = extract_mention_keywords("星期天提醒我给电动车充电")
    # 提取的是词块，不保证切成标准词；「电动车」「充电」都在同一个块里即可
    assert any("电动车" in w for w in words)
    assert any("充电" in w for w in words)
    assert all("星期天" not in w for w in words)
    assert all("提醒" not in w for w in words)

    words = extract_mention_keywords("Mando说要下影之刃零的盗版")
    assert any("影之刃零" in w for w in words)
    assert any("盗版" in w for w in words)
    assert all("Mando" not in w for w in words)


def test_relative_time_is_not_a_keyword() -> None:
    words = extract_mention_keywords("三天后提醒我还书")
    assert all("三天后" not in w for w in words)


def test_is_mentioned_matches_current_and_recent() -> None:
    item = _item("星期天提醒我给电动车充电")
    assert is_mentioned(item, user_message="我明天去给电动车充电")
    assert is_mentioned(item, user_message="在吗", recent_messages=["我买了充电线"])
    assert not is_mentioned(item, user_message="在吗")


def test_care_always_injected() -> None:
    item = _item("他现在有点头晕", kind="care")
    assert should_inject(item, NOW)


def test_overdue_always_injected() -> None:
    yesterday = (NOW - timedelta(days=1)).isoformat()
    item = _item("昨天说要发视频", due_at=yesterday)
    assert should_inject(item, NOW)


def test_mentioned_injects_regardless_of_window() -> None:
    future = (NOW + timedelta(days=2)).isoformat()
    item = _item("星期天给电动车充电", due_at=future, kind="commitment")
    # 不在窗口内（还有 2 天，窗口是 2 小时）
    assert not should_inject(item, NOW)
    # 但他提起了
    assert should_inject(item, NOW, user_message="充电那事我记着呢")


def test_dated_item_injects_the_whole_day() -> None:
    # 「星期天」只有日期没有钟点：当天 0 点就该出现
    sunday = datetime(2026, 11, 8, 0, 0, tzinfo=timezone.utc).isoformat()
    item = _item("星期天提醒我给电动车充电", due_at=sunday, kind="follow_up")
    friday = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)
    sunday_noon = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    assert not should_inject(item, friday)
    assert should_inject(item, sunday_noon)


def test_timed_item_injects_within_lead() -> None:
    at_2pm = datetime(2026, 11, 6, 22, 0, tzinfo=timezone.utc).isoformat()
    item = _item("今晚十点提醒我吃药", due_at=at_2pm, kind="commitment")
    at_6pm = datetime(2026, 11, 6, 18, 0, tzinfo=timezone.utc)
    at_9pm = datetime(2026, 11, 6, 21, 0, tzinfo=timezone.utc)
    # commitment 提前 2 小时：18 点不在，21 点在
    assert not should_inject(item, at_6pm)
    assert should_inject(item, at_9pm)


def test_no_time_item_stays_visible() -> None:
    item = _item("记得买猫粮")
    assert should_inject(item, NOW)


def test_select_filters_by_window() -> None:
    future = (NOW + timedelta(days=3)).isoformat()
    items = [
        _item("记得买猫粮"),                       # 无时间：注入
        _item("三天后提醒我还书", due_at=future),   # 未到窗口：不注入
    ]
    picked = select_attention_items(items, 4, now=NOW)
    assert [i.content for i in picked] == ["记得买猫粮"]
    # 用户提起了那条，就注入
    picked = select_attention_items(items, 4, now=NOW, user_message="我还书去了")
    assert {i.content for i in picked} == {"记得买猫粮", "三天后提醒我还书"}


def test_care_and_overdue_rank_before_window() -> None:
    """逾期与 care 不被窗口挡住——这条保证它们永远能进来。"""
    yesterday = (NOW - timedelta(hours=1)).isoformat()
    items = [
        _item("昨天说要发视频", due_at=yesterday, kind="plan"),
        _item("他现在有点累", kind="care"),
    ]
    picked = select_attention_items(items, 4, now=NOW)
    assert len(picked) == 2


def test_end_of_day_due_is_treated_as_whole_day() -> None:
    """建项没给钟点时 due_at 会兜底成当天 23:59，不能拖到 21:59 才出现。"""
    tonight = datetime(2026, 11, 6, 23, 59, tzinfo=timezone.utc).isoformat()
    item = _item("今晚只让我用声音陪着", due_at=tonight, kind="commitment")
    morning = datetime(2026, 11, 6, 9, 0, tzinfo=timezone.utc)
    assert should_inject(item, morning)
