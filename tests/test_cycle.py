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
