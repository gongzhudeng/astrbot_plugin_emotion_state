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


def test_model_date_is_corrected_by_local_calendar() -> None:
    """模型把「后天星期天」算成周一；日期以本地日历为准，钟点保留模型的。"""
    from datetime import timedelta

    from astrbot_plugin_emotion_state.core.attention import reconcile_due_at

    cst = timezone(timedelta(hours=8))
    friday_night = datetime(2026, 11, 6, 20, 18, tzinfo=cst)
    fixed = reconcile_due_at("后天星期天", "2026-11-09T09:00:00+08:00", friday_night)
    assert fixed.startswith("2026-11-08T09:00"), fixed


def test_reconcile_keeps_model_time_when_date_matches() -> None:
    from datetime import timedelta

    from astrbot_plugin_emotion_state.core.attention import reconcile_due_at

    cst = timezone(timedelta(hours=8))
    friday_night = datetime(2026, 11, 6, 20, 18, tzinfo=cst)
    # 日期一致：模型给的 15:30 比本地兜底的 23:59 精确，保留
    keep = reconcile_due_at("星期天", "2026-11-08T15:30:00+08:00", friday_night)
    assert keep.startswith("2026-11-08T15:30"), keep
    # 本地解析不出（绝对日期表述）：原样保留
    raw = "2026-11-20T10:00:00+08:00"
    assert reconcile_due_at("11月20日上午十点", raw, friday_night) == raw


def test_update_due_is_also_reconciled() -> None:
    """改时间也要校准，不然只堵了 create 一半的洞。"""
    from astrbot_plugin_emotion_state.core.attention import apply_attention_observation
    from astrbot_plugin_emotion_state.core.models import AttentionObservation, StateLedger

    ledger, ok, reason = apply_attention_observation(
        StateLedger(user_key="private:upd"),
        AttentionObservation(
            action="create",
            content="星期天提醒我充电",
            kind="remember",
            time_hint="后天",
            confidence=0.9,
            explicit=True,
            source="attention_review",
            evidence_quote="记得提醒我后天充电",
            evidence_speaker="user",
        ),
    )
    assert ok, reason
    item = ledger.attention_items[0]

    updated, ok, reason = apply_attention_observation(
        ledger,
        AttentionObservation(
            action="update",
            item_id=item.id,
            item_version=item.version,
            time_hint="星期天",
            due_at="2026-11-09T09:00:00+08:00",  # 模型给了一个错日期，由本地校准
            confidence=0.9,
            source="attention_review",
            evidence_quote="那就星期天提醒吧",
            evidence_speaker="user",
        ),
    )
    assert ok, reason
    fixed = updated.attention_items[0].due_at
    # 不写死具体日期（会随运行日期失效）：只要落在周日本身就说明校准生效
    from datetime import datetime as _dt

    assert _dt.fromisoformat(fixed).weekday() == 6, fixed


def test_dawn_shift_survives_reconciliation() -> None:
    """凌晨 0-5 点说的「明天」= 睡醒后的那个白天，不是日历上的明天。

    这是人话不是数据：01:30 说「明天提醒我开会」，指的是今天上午，
    不是明天上午。本地 resolve 有 dawn_shift，校准必须沿用同一条规则，
    否则会把模型给的日历日期「纠正」成错的那天。
    """
    from datetime import timedelta

    from astrbot_plugin_emotion_state.core.attention import reconcile_due_at, resolve_due_at

    cst = timezone(timedelta(hours=8))
    dawn = datetime(2026, 11, 6, 1, 30, tzinfo=cst)

    # 本地解析：明天 -> 当天白天
    assert resolve_due_at("明天", dawn).startswith("2026-11-06")
    # 模型按日历给 11-07，校准后应回到 11-06（保留它给的 09:00）
    fixed = reconcile_due_at("明天", "2026-11-07T09:00:00+08:00", dawn)
    assert fixed.startswith("2026-11-06T09:00"), fixed
    # 白天说「明天」就是日历明天，不受影响
    noon = datetime(2026, 11, 6, 12, 0, tzinfo=cst)
    assert reconcile_due_at("明天", "2026-11-07T09:00:00+08:00", noon).startswith("2026-11-07T09:00")
