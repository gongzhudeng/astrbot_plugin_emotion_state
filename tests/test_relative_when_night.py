"""跨零点不翻页：凌晨说的事仍然算「今晚」。

对照 core/guidance.py 的 relative_when 与 NIGHT_CUTOFF_HOUR。
"""
from datetime import datetime, timedelta

from astrbot_plugin_emotion_state.core.guidance import relative_when


def _local(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi).astimezone()


def test_after_midnight_still_tonight():
    """01:00 回看 23:00，人还没睡，应该说今晚。"""
    now = _local(2026, 10, 23, 1, 0)
    event = _local(2026, 10, 22, 23, 0)
    label = relative_when(now, event)
    assert label.startswith("今晚"), label


def test_just_past_midnight_is_tonight():
    now = _local(2026, 10, 23, 1, 0)
    event = _local(2026, 10, 23, 0, 30)
    assert relative_when(now, event).startswith("今晚")


def test_old_event_still_yesterday():
    """凌晨回看昨天下午，超过 8 小时，保持昨天。"""
    now = _local(2026, 10, 23, 1, 0)
    event = _local(2026, 10, 22, 15, 0)
    assert relative_when(now, event).startswith("昨天")


def test_daytime_does_not_use_tonight():
    """下午看当天凌晨，不该说今晚。"""
    now = _local(2026, 10, 23, 14, 0)
    event = _local(2026, 10, 23, 0, 30)
    assert relative_when(now, event).startswith("今天")


def test_multi_day_keeps_day_count():
    now = _local(2026, 10, 23, 1, 0)
    event = _local(2026, 10, 20, 20, 0)
    assert relative_when(now, event).startswith("3天前")


def test_iso_string_input():
    now = _local(2026, 10, 23, 2, 0)
    event = now - timedelta(hours=3)
    label = relative_when(now, event.isoformat())
    assert label.startswith("今晚")
