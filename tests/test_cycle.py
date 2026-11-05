from __future__ import annotations

import json
from datetime import date

from astrbot_plugin_emotion_state.core.cycle import (
    FORBIDDEN_WORDS,
    PHASE_FOLLICULAR,
    PHASE_LUTEAL,
    PHASE_OVULATION,
    PHASE_PERIOD,
    CycleSettings,
    body_line,
    load_settings,
    mood_delta,
    phase_of,
    schedule_hint,
    update_role,
)
from astrbot_plugin_emotion_state.core.injector import build_snapshot
from astrbot_plugin_emotion_state.core.models import StateLedger


def _settings(start: str = "2026-11-03", **kwargs) -> CycleSettings:
    return CycleSettings(last_period_start=start, **kwargs)


def test_phase_boundaries_use_start_day_as_day_one() -> None:
    settings = _settings()
    cases = {
        0: (PHASE_PERIOD, 1),
        1: (PHASE_PERIOD, 2),
        4: (PHASE_PERIOD, 5),
        5: (PHASE_FOLLICULAR, 6),
        12: (PHASE_FOLLICULAR, 13),
        13: (PHASE_OVULATION, 14),
        15: (PHASE_OVULATION, 16),
        16: (PHASE_LUTEAL, 17),
        27: (PHASE_LUTEAL, 28),
    }
    for offset, expected in cases.items():
        today = date.fromordinal(date(2026, 11, 3).toordinal() + offset)
        assert phase_of(today, settings) == expected, today


def test_cycle_wraps_after_length() -> None:
    settings = _settings()
    assert phase_of(date(2026, 12, 1), settings) == (PHASE_PERIOD, 1)
    assert phase_of(date(2026, 12, 2), settings) == (PHASE_PERIOD, 2)


def test_unconfigured_and_future_and_bad_start_are_silent() -> None:
    today = date(2026, 11, 5)
    assert phase_of(today, None) is None
    assert phase_of(today, CycleSettings()) is None
    assert phase_of(today, _settings("2026-12-01")) is None
    assert phase_of(today, _settings("不是日期")) is None
    assert body_line(phase_of(today, _settings("不是日期"))) == ""
    assert mood_delta(None) == {"energy": 0.0, "tension": 0.0, "neg_mul": 1.0}
    assert schedule_hint(None) == ""


def test_short_cycle_and_long_cycle_boundaries() -> None:
    short = _settings(cycle_length=21, period_length=3)
    assert phase_of(date(2026, 11, 5), short) == (PHASE_PERIOD, 3)
    assert phase_of(date(2026, 11, 6), short) == (PHASE_FOLLICULAR, 4)
    assert phase_of(date(2026, 11, 24), short) == (PHASE_PERIOD, 1)

    long = _settings(cycle_length=35)
    assert phase_of(date(2026, 11, 30), long) == (PHASE_LUTEAL, 28)
    assert body_line(phase_of(date(2026, 11, 30), long), long) == ""
    assert body_line(phase_of(date(2026, 12, 1), long), long) != ""


def test_body_line_only_when_it_matters() -> None:
    settings = _settings()
    assert "小腹坠着" in body_line((PHASE_PERIOD, 1), settings)
    assert "有点乏" in body_line((PHASE_PERIOD, 4), settings)
    assert body_line((PHASE_FOLLICULAR, 8), settings) == ""
    assert body_line((PHASE_OVULATION, 14), settings) == ""
    assert body_line((PHASE_LUTEAL, 18), settings) == ""
    assert "容易烦" in body_line((PHASE_LUTEAL, 25), settings)


def test_mood_delta_is_small_and_zero_when_irrelevant() -> None:
    settings = _settings()
    early = mood_delta((PHASE_PERIOD, 1), settings)
    assert early["energy"] == -0.10
    assert early["neg_mul"] == 1.10
    late = mood_delta((PHASE_PERIOD, 4), settings)
    assert late["energy"] == -0.05
    assert late["neg_mul"] == 1.05
    luteal = mood_delta((PHASE_LUTEAL, 25), settings)
    assert luteal["tension"] == 0.10
    assert luteal["neg_mul"] == 1.15
    assert mood_delta((PHASE_FOLLICULAR, 8), settings) == {
        "energy": 0.0,
        "tension": 0.0,
        "neg_mul": 1.0,
    }


def test_schedule_hint_levels_and_fallback() -> None:
    settings = _settings()
    assert "饮食可以暖一点" in schedule_hint((PHASE_PERIOD, 2), "light", settings)
    assert "减少外出" in schedule_hint((PHASE_PERIOD, 2), "medium", settings)
    assert "以休息为主" in schedule_hint((PHASE_PERIOD, 2), "heavy", settings)
    # 非法档位退回最轻的一档，而不是抛异常或返回空。
    assert "饮食可以暖一点" in schedule_hint((PHASE_PERIOD, 2), "???", settings)
    assert schedule_hint((PHASE_FOLLICULAR, 8), "heavy", settings) == ""
    assert "容易烦" in schedule_hint((PHASE_LUTEAL, 25), "light", settings)


def test_copy_never_uses_medical_terms() -> None:
    settings = _settings()
    samples = []
    for day in range(1, 29):
        phase_day = phase_of(date.fromordinal(date(2026, 11, 3).toordinal() + day - 1), settings)
        samples.append(body_line(phase_day, settings))
        samples.append(schedule_hint(phase_day, "medium", settings))
    for text in samples:
        for word in FORBIDDEN_WORDS:
            assert word not in text, text


def test_settings_roundtrip_and_bad_file(tmp_path) -> None:
    path = tmp_path / "cycle.json"
    assert load_settings(path) == {}
    saved = update_role(path, "bot-a", _settings("2026-11-01"))
    assert saved.last_period_start == "2026-11-01"
    roles = load_settings(path)
    assert roles["bot-a"].cycle_length == 28
    assert path.exists()
    assert not (tmp_path / "cycle.json.tmp").exists()

    path.write_text("这不是 json", encoding="utf-8")
    assert load_settings(path) == {}

    path.write_text(json.dumps({"roles": {"bot-a": "坏值"}}), encoding="utf-8")
    assert load_settings(path)["bot-a"].cycle_length == 28


def test_settings_values_are_clamped() -> None:
    raw = CycleSettings(last_period_start="2026-11-03", cycle_length=365, period_length=99)
    normalized = raw.normalized()
    assert normalized.cycle_length == 35
    assert normalized.period_length == 7
    assert normalized.last_period_start == "2026-11-03"


def test_cycle_line_goes_before_body_reaction() -> None:
    ledger = StateLedger(user_key="private:cycle")
    prompt = build_snapshot(ledger, cycle_line="经期第2天，小腹坠着")
    assert "经期第2天" in prompt
    assert prompt.index("经期第2天") < prompt.index("当前身体反应档位")


def test_empty_cycle_line_leaves_block_byte_identical() -> None:
    ledger = StateLedger(user_key="private:cycle")
    assert build_snapshot(ledger) == build_snapshot(ledger, cycle_line="")
    assert build_snapshot(ledger) == build_snapshot(ledger, cycle_line="   ")


def test_view_reports_upcoming_start(tmp_path) -> None:
    """提前设置（明天开始）是正常需求，不该报「异常」。"""
    import json as _json
    from datetime import timedelta

    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    plugin = object.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.cycle_file = tmp_path / "cycle.json"
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    (tmp_path / "cycle.json").write_text(
        _json.dumps({"version": 1, "roles": {"default": {"last_period_start": tomorrow}}}),
        encoding="utf-8",
    )
    view = plugin._cycle_view()
    assert view["configured"] is True
    assert view["phase"] == ""
    assert view["starts_in_days"] == 1
    assert plugin._cycle_line() == ""


def test_raw_args_come_from_the_message_not_param_slots() -> None:
    """宿主按空格逐个分配参数：handler 只有两个形参时，日期会被丢掉。

    这条是 2026-11-05 的真机故障：`/生理期 开始 2026-11-06` 回「日期看不懂」。
    所以参数一律从 message_str 里剥前缀取，不依赖形参个数。
    """
    from types import SimpleNamespace

    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    raw = EmotionStatePlugin._cycle_raw_args
    assert raw(SimpleNamespace(message_str="/生理期 开始 2026-11-06")) == "开始 2026-11-06"
    assert raw(SimpleNamespace(message_str="生理期 开始 2026-11-06")) == "开始 2026-11-06"
    assert raw(SimpleNamespace(message_str="/生理期")) == ""
    # 拿不到原文时退回形参，至少不丢功能。
    assert raw(SimpleNamespace(message_str=""), "开始", "2026-11-06") == "开始 2026-11-06"


def test_parse_cycle_args_keeps_the_date() -> None:
    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    assert EmotionStatePlugin._parse_cycle_args("开始 2026-11-06") == ("开始", "2026-11-06")
    assert EmotionStatePlugin._parse_cycle_args("") == ("", "")


def test_image_line_variants() -> None:
    """情绪查看图片里那一行：只在经期/经前出现，未来日期说「N 天后开始」。"""
    from astrbot_plugin_emotion_state.core.image_renderer import EmotionStateImageRenderer

    line = EmotionStateImageRenderer._cycle_line
    assert line(None) == ""
    assert line({"configured": False}) == ""
    assert line({"configured": True, "phase": "", "starts_in_days": 1}) == "1 天后开始"
    assert (
        line(
            {
                "configured": True,
                "phase": "period",
                "phase_label": "经期",
                "day": 2,
                "next_period_in_days": 27,
            }
        )
        == "经期第 2 天 · 距下次约 27 天"
    )
    # 经后/排卵期没有值得说的，图片上不该多一行。
    assert line({"configured": True, "phase": "follicular", "phase_label": "", "day": 8}) == ""
