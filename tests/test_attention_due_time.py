"""due 的形态：睡前跨天、整点浮动。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from astrbot_plugin_emotion_state.core.attention import (
    configure_due_jitter,
    jitter_due_at,
    parse_time,
    resolve_due_at,
)

CST = timezone(timedelta(hours=8))


def test_bedtime_rolls_past_midnight() -> None:
    """「睡前」是当晚跨过零点之后，不是 23:59。"""
    anchor = datetime(2026, 11, 22, 22, 0, tzinfo=CST)
    assert resolve_due_at("睡前", anchor).startswith("2026-11-23T00:30")
    # 「明天睡前」= 明天夜里 → 后天 00:30
    assert resolve_due_at("明天睡前", anchor).startswith("2026-11-24T00:30")


def test_other_periods_unchanged() -> None:
    anchor = datetime(2026, 11, 22, 12, 0, tzinfo=CST)
    assert resolve_due_at("晚上", anchor).startswith("2026-11-22T23:59")


def test_jitter_only_moves_round_times() -> None:
    """整点/半点往后错几分钟；其他分钟原样返回。"""
    configure_due_jitter(10)
    try:
        moved = jitter_due_at("2026-11-22T15:00:00+08:00")
        minute = parse_time(moved).minute
        assert 1 <= minute <= 10, moved
        assert parse_time(moved).hour == 15

        half = jitter_due_at("2026-11-22T15:30:00+08:00")
        assert 31 <= parse_time(half).minute <= 40, half

        # 已经是奇数的分钟不动（否则每次保存都会再挪一次）
        assert jitter_due_at("2026-11-22T15:07:00+08:00") == "2026-11-22T15:07:00+08:00"
    finally:
        configure_due_jitter(0)


def test_jitter_never_crosses_midnight() -> None:
    configure_due_jitter(10)
    try:
        late = "2026-11-22T23:30:00+08:00"
        assert parse_time(jitter_due_at(late)).date() == parse_time(late).date()
    finally:
        configure_due_jitter(0)


def test_jitter_off_is_identity() -> None:
    configure_due_jitter(0)
    value = "2026-11-22T15:00:00+08:00"
    assert jitter_due_at(value) == value
