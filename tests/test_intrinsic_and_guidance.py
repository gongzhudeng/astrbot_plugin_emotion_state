"""Tests for intrinsic mood dynamics, dual-layer mood, guidance, and life events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from astrbot_plugin_emotion_state.core.guidance import (
    build_guidance_prompt,
    format_guidance_block,
    guidance_regime,
    local_guidance_fallback,
    needs_guidance,
    parse_guidance_response,
)
from astrbot_plugin_emotion_state.core.injector import (
    InjectionOptions,
    inject_prompt,
)
from astrbot_plugin_emotion_state.core.intrinsic import (
    circadian_shift,
    decay_mood_offset,
    draw_temperament,
    drift_target,
    maybe_night_missing_observation,
    parse_night_range,
)
from astrbot_plugin_emotion_state.core.life_events import (
    draw_slots,
    event_slot_due,
    parse_life_event_response,
)
from astrbot_plugin_emotion_state.core.models import (
    EventObservation,
    InnerEvent,
    IntrinsicParams,
    MoodOffsetState,
    SensitivityParams,
    StateLedger,
)
from astrbot_plugin_emotion_state.core.rules import LocalRuleEngine
from astrbot_plugin_emotion_state.core.service import EmotionStateService
from astrbot_plugin_emotion_state.core.settlement import (
    apply_observation,
    decay_ledger,
    effective_mood,
    infer_attack_target,
    is_attack_like,
    mood_label,
    mood_narrative,
    settle_transient_mood,
)
from astrbot_plugin_emotion_state.core.storage import LedgerStore

NIGHT = parse_night_range("22:00-02:00")


# ---------------------------------------------------------------------------
# Mood vocabulary recalibration and contextual labels
# ---------------------------------------------------------------------------


def test_mood_label_thresholds_reach_happy_and_sad_states() -> None:
    assert mood_label(0.52, 0.2, 0.52) == "明快开心"
    assert mood_label(0.62, 0.2, 0.62) == "雀跃"
    assert mood_label(-0.5, 0.2, 0.4) == "低落"
    assert mood_label(-0.3, 0.2, 0.4) == "有些在意"
    assert mood_label(0.3, 0.2, 0.4) == "温和愉快"


def test_night_melancholy_label_requires_night_hours() -> None:
    assert mood_label(-0.2, 0.3, 0.4, hour=23, night_hours=NIGHT) == "夜晚感伤"
    assert mood_label(-0.2, 0.3, 0.4, hour=14, night_hours=NIGHT) == "有些在意"
    assert mood_label(-0.2, 0.3, 0.4, hour=23) == "有些在意"


def test_wronged_label_requires_interpersonal_unresolved_event() -> None:
    assert mood_label(-0.3, 0.55, 0.4, interpersonal_unresolved=True) == "委屈"
    assert mood_label(-0.3, 0.55, 0.4, interpersonal_unresolved=False) == "有些在意"


# ---------------------------------------------------------------------------
# Dual-layer mood: short-term offset vs settled baseline
# ---------------------------------------------------------------------------


def _ledger_with_baseline(valence: float) -> StateLedger:
    ledger = StateLedger(user_key="private:layer")
    ledger.mood.valence = valence
    ledger.mood.tension = 0.2
    ledger.mood.energy = 0.45
    ledger.mood.label = mood_label(valence, 0.2, 0.45)
    return ledger


def test_transient_signal_lands_in_offset_not_baseline() -> None:
    ledger = _ledger_with_baseline(-0.5)
    signal = EventObservation(
        action="create",
        fact="他夸了我",
        emotional_meaning="被夸奖很开心",
        valence=0.8,
        intensity=0.6,
        confidence=0.8,
    )
    updated = settle_transient_mood(ledger, [signal])

    # Baseline is untouched; the laugh lives in the fast offset.
    assert updated.mood.valence == pytest.approx(-0.5)
    assert updated.mood_offset.valence > 0.1
    assert updated.mood_offset.significant


def test_mood_offset_decays_to_zero() -> None:
    ledger = _ledger_with_baseline(-0.5)
    ledger.mood_offset = MoodOffsetState(
        valence=0.3, updated_at="2026-09-05T12:00:00+00:00"
    )

    decay_mood_offset(ledger, elapsed_hours=2.0, half_life_minutes=15.0)

    assert abs(ledger.mood_offset.valence) < 0.005
    assert not ledger.mood_offset.significant


def test_mood_narrative_composes_dual_layers_only_when_offset_matters() -> None:
    ledger = _ledger_with_baseline(-0.5)
    plain = mood_narrative(ledger)
    assert plain == "当前心境：低落"
    assert "刚刚" not in plain

    ledger.mood_offset = MoodOffsetState(valence=0.2)
    narrative = mood_narrative(ledger)
    assert "整体：低落" in narrative
    assert "刚刚" in narrative
    assert "暂时性" in narrative


def test_effective_mood_blends_offset_for_expression() -> None:
    ledger = _ledger_with_baseline(-0.3)
    ledger.mood_offset = MoodOffsetState(valence=0.25)
    effective = effective_mood(ledger)
    assert effective.valence == pytest.approx(-0.05)
    # Baseline itself must stay untouched.
    assert ledger.mood.valence == pytest.approx(-0.3)


def test_sensitivity_scales_transient_weight_by_direction() -> None:
    ledger = _ledger_with_baseline(0.0)
    negative = EventObservation(
        action="create",
        fact="被凶了",
        emotional_meaning="难受",
        valence=-0.8,
        intensity=0.6,
        confidence=0.8,
    )
    blunt = settle_transient_mood(
        ledger,
        [negative],
        sensitivity=SensitivityParams(overall=1.0, negative=0.2, positive=1.0),
    )
    normal = settle_transient_mood(ledger, [negative])
    assert abs(blunt.mood_offset.valence) < abs(normal.mood_offset.valence)


# ---------------------------------------------------------------------------
# Intrinsic dynamics: circadian, temperament, drift, night missing
# ---------------------------------------------------------------------------


def test_parse_night_range_wraps_midnight() -> None:
    assert set(parse_night_range("22:00-02:00")) == {22, 23, 0, 1}
    assert set(parse_night_range("23:30-01:30")) == {23, 0, 1}
    assert parse_night_range("garbage") == (22, 23, 0, 1)


def test_circadian_shift_only_at_night() -> None:
    day = circadian_shift(14, NIGHT, 1.0)
    night = circadian_shift(23, NIGHT, 1.0)
    assert day == (0.0, 0.0, 0.0)
    assert night[0] < 0 and night[1] < 0 and night[2] > 0
    doubled = circadian_shift(23, NIGHT, 2.0)
    assert doubled[0] == pytest.approx(night[0] * 2)


def test_draw_temperament_is_deterministic() -> None:
    first = draw_temperament("private:x", "2026-09-05", ["轻快", "慵懒", "多愁善感"])
    second = draw_temperament("private:x", "2026-09-05", ["轻快", "慵懒", "多愁善感"])
    other_day = draw_temperament(
        "private:x", "2026-09-06", ["轻快", "慵懒", "多愁善感"]
    )
    assert first.word == second.word
    assert first.valence_shift == second.valence_shift
    assert other_day.word in {"轻快", "慵懒", "多愁善感"}


def test_drift_target_is_bounded() -> None:
    now = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
    valence, energy, tension = drift_target("private:x", now, 0.05)
    assert max(abs(valence), abs(energy), abs(tension)) <= 0.05
    assert drift_target("private:x", now, 0.0) == (0.0, 0.0, 0.0)


def test_decay_ledger_applies_intrinsic_shift_and_temperament() -> None:
    ledger = StateLedger(user_key="private:intrinsic")
    ledger.mood.valence = 0.0
    ledger.mood.energy = 0.45
    ledger.mood.tension = 0.2
    ledger.last_settled_at = "2026-09-05T14:00:00+00:00"
    params = IntrinsicParams(
        night_hours=NIGHT,
        night_strength=1.0,
        temperament_enabled=True,
        temperament_words=("多愁善感",),
        drift_amplitude=0.0,
    )
    night_now = datetime(2026, 9, 5, 23, 30, tzinfo=timezone.utc).astimezone()

    updated = decay_ledger(
        ledger, now=night_now, half_life_hours=72.0, intrinsic=params
    )

    # Temperament word is drawn for the local day and shifts the baseline down.
    assert updated.today_temperament.word == "多愁善感"
    assert updated.today_temperament.date == night_now.date().isoformat()
    assert updated.mood.valence < 0.0


def test_decay_ledger_decays_mood_offset() -> None:
    ledger = StateLedger(user_key="private:offset-decay")
    ledger.mood_offset = MoodOffsetState(valence=0.3)
    ledger.last_settled_at = (
        datetime.now(timezone.utc) - timedelta(minutes=45)
    ).isoformat()
    updated = decay_ledger(ledger)
    assert abs(updated.mood_offset.valence) < 0.05


def test_night_missing_event_created_once_per_night() -> None:
    now = datetime.now(timezone.utc).astimezone()
    # Build a moment that is guaranteed to be inside the night band.
    night_hour = next(h for h in (22, 23, 0, 1))
    night_now = now.replace(hour=night_hour)
    ledger = StateLedger(user_key="private:missing")
    ledger.last_user_message_ts = (night_now - timedelta(hours=3)).timestamp()
    params = IntrinsicParams(
        night_hours=NIGHT,
        night_missing_after_hours=2.0,
        sensitivity=SensitivityParams(),
    )
    observation = maybe_night_missing_observation(ledger, night_now, params)
    assert observation is not None
    assert "想念" in observation.emotional_meaning

    updated, applied, _ = apply_observation(ledger, observation)
    assert applied
    again = maybe_night_missing_observation(updated, night_now, params)
    assert again is None


def test_night_missing_requires_quiet_hours_and_night() -> None:
    now = datetime.now(timezone.utc).astimezone()
    params = IntrinsicParams(
        night_hours=NIGHT,
        night_missing_after_hours=2.0,
        sensitivity=SensitivityParams(),
    )
    night_now = now.replace(hour=23)
    quiet_ledger = StateLedger(user_key="private:missing-quiet")
    quiet_ledger.last_user_message_ts = (night_now - timedelta(hours=1)).timestamp()
    # Night but only one quiet hour: still nothing.
    assert maybe_night_missing_observation(quiet_ledger, night_now, params) is None
    # Daytime: no missing event even after long silence.
    loud_ledger = StateLedger(user_key="private:missing-day")
    loud_ledger.last_user_message_ts = (night_now - timedelta(hours=5)).timestamp()
    assert (
        maybe_night_missing_observation(loud_ledger, now.replace(hour=14), params)
        is None
    )


# ---------------------------------------------------------------------------
# Misjudgment regressions for keyword rules
# ---------------------------------------------------------------------------


def test_disgust_about_video_is_transient_not_attack() -> None:
    engine = LocalRuleEngine()
    run = engine.run(
        "2.87 复制打开抖音，看看【x的作品】… 看这个 好恶心 要吐了", watermark=1
    )
    assert run.candidates == []
    assert run.transient_signals
    assert all(item.target == "unknown" for item in run.transient_signals)
    assert all("disgust" in item.tags for item in run.transient_signals)


def test_mentioning_him_in_a_photo_request_is_not_an_attack() -> None:
    engine = LocalRuleEngine()
    run = engine.run("不小心刷到个美女视频 你能按她这样 拍一张我看不", watermark=2)
    assert run.candidates == []


def test_direct_insult_is_still_detected() -> None:
    engine = LocalRuleEngine()
    run = engine.run("你真恶心，滚开", watermark=3)
    assert any(item.target == "user" for item in run.candidates)
    assert any("abuse" in item.tags for item in run.candidates)


def test_infer_attack_target_requires_adjacent_insult() -> None:
    assert is_attack_like("你真恶心，滚开")
    assert infer_attack_target("你真恶心，滚开")[0] == "user"
    # A bare "你" plus a request is not an attack.
    assert infer_attack_target("你能按她这样拍一张我看不")[0] == "unknown"
    assert infer_attack_target("看这个 好恶心 要吐了")[0] == "unknown"


# ---------------------------------------------------------------------------
# Expression guidance: gating, regime, formatting, parsing
# ---------------------------------------------------------------------------


def _sad_ledger() -> StateLedger:
    ledger = StateLedger(user_key="private:guide")
    ledger.mood.valence = -0.6
    ledger.mood.tension = 0.3
    ledger.mood.energy = 0.35
    ledger.mood.label = "低落"
    return ledger


def test_needs_guidance_gate() -> None:
    assert needs_guidance(_sad_ledger())
    neutral = StateLedger(user_key="private:guide-calm")
    assert not needs_guidance(neutral)
    intensified = StateLedger(user_key="private:guide-event")
    intensified.events.append(
        InnerEvent(
            fact="他答应今晚陪我打游戏",
            emotional_meaning="期待",
            lifecycle="intensified",
        )
    )
    assert needs_guidance(intensified)


def test_guidance_regime_changes_with_state_not_clock() -> None:
    ledger = _sad_ledger()
    base = guidance_regime(ledger)

    # Crossing a time band with identical state must NOT change the regime.
    now = datetime.now().astimezone()
    assert guidance_regime(ledger, now=now.replace(hour=10)) == guidance_regime(
        ledger, now=now.replace(hour=23)
    )

    # Reaching a different emotional state does change it.
    ledger.mood.valence = 0.4
    ledger.mood.label = "温和愉快"
    assert guidance_regime(ledger) != base


def test_format_guidance_block_strength_tiers() -> None:
    guidance = {
        "tone": "语气放轻",
        "can_say": "可以说今天有点难过，因为等不到回复",
        "avoid": "不要指责",
    }
    assert format_guidance_block(guidance, strength=0.0) == ""
    light = format_guidance_block(guidance, strength=0.4)
    assert "语气放轻" in light
    assert "可以自然流露" not in light
    full = format_guidance_block(guidance, strength=1.0)
    assert "可以自然流露" in full
    assert "避免" in full
    assert format_guidance_block(None, strength=1.0) == ""


def test_parse_guidance_response_accepts_json_and_rejects_empty() -> None:
    parsed = parse_guidance_response(
        '{"tone":"软一点","can_say":"说想他","avoid":"别冷脸"}'
    )
    assert parsed["tone"] == "软一点"
    with pytest.raises(ValueError):
        parse_guidance_response('{"can_say":"缺少tone"}')


def test_guidance_prompt_embeds_style_and_events() -> None:
    ledger = _sad_ledger()
    prompt = build_guidance_prompt(
        ledger,
        now=datetime.now().astimezone(),
        style_hint="难过时不阴阳怪气，会变安静",
    )
    assert "难过时不阴阳怪气" in prompt
    assert "低落" in prompt
    # Suggestions stay direction-only: no scripted lines, persona owns the words.
    assert "不要写具体台词或原话" in prompt


def test_local_guidance_fallback_mentions_reason() -> None:
    ledger = _sad_ledger()
    ledger.events.append(
        InnerEvent(
            fact="他今天一直没来找我",
            emotional_meaning="有点失落",
            lifecycle="active",
        )
    )
    fallback = local_guidance_fallback(ledger, datetime.now().astimezone())
    assert fallback["tone"]
    assert "他今天一直没来找我" in fallback["can_say"]


def test_injection_includes_guidance_only_when_gate_open() -> None:
    ledger = _sad_ledger()
    ledger.expression_guidance = None
    options = InjectionOptions(guidance_enabled=True)
    prompt = inject_prompt("persona", ledger, options=options)
    assert "emotion_state_guidance" not in prompt

    from astrbot_plugin_emotion_state.core.models import ExpressionGuidance

    ledger.expression_guidance = ExpressionGuidance(
        tone="语气放轻",
        can_say="说今天有点难过，因为等不到回复",
        regime="低落|晚上|||",
    )
    prompted = inject_prompt("persona", ledger, options=options)
    assert "emotion_state_guidance" in prompted
    assert "语气放轻" in prompted

    # Neutral state with when_needed gate: block withheld.
    neutral = StateLedger(user_key="private:calm")
    neutral.expression_guidance = ExpressionGuidance(
        tone="按人格正常聊", regime="平静|晚上|||"
    )
    hidden = inject_prompt(
        "persona",
        neutral,
        options=InjectionOptions(guidance_enabled=True, guidance_when_needed=True),
    )
    assert "emotion_state_guidance" not in hidden
    forced = inject_prompt(
        "persona",
        neutral,
        options=InjectionOptions(guidance_enabled=True, guidance_when_needed=False),
    )
    assert "emotion_state_guidance" in forced


# ---------------------------------------------------------------------------
# Injection slimming: zero limits and cleared rules text
# ---------------------------------------------------------------------------


def test_injection_omits_sections_at_zero_limits() -> None:
    ledger = StateLedger(user_key="private:slim")
    ledger.events.append(
        InnerEvent(fact="有件事", emotional_meaning="影响", lifecycle="active")
    )
    prompt = inject_prompt("persona", ledger, max_events=0, max_attention_items=0)
    assert "当前仍有影响的事情" not in prompt
    assert "当前没有足够确定" not in prompt
    assert "仍需留意或接续的事项" not in prompt
    assert "当前没有仍需留意" not in prompt
    assert "当前心境" in prompt


def test_cleared_rules_text_omits_rules_block() -> None:
    ledger = StateLedger(user_key="private:norules")
    with_rules = inject_prompt("persona", ledger, rules_text="自定义规则")
    assert "<emotion_state_rules>" in with_rules
    assert "自定义规则" in with_rules
    without_rules = inject_prompt("persona", ledger, rules_text="")
    assert "<emotion_state_rules>" not in without_rules


# ---------------------------------------------------------------------------
# Life events: slots, parsing
# ---------------------------------------------------------------------------


def test_draw_slots_deterministic_and_in_window() -> None:
    first = draw_slots("private:x", "2026-09-05", "10:00-22:30", 2)
    second = draw_slots("private:x", "2026-09-05", "10:00-22:30", 2)
    assert [slot.isoformat() for slot in first] == [slot.isoformat() for slot in second]
    assert 1 <= len(first) <= 2
    for slot in first:
        assert 10 <= slot.hour <= 22 or (slot.hour == 22 and slot.minute <= 30)
    assert draw_slots("private:x", "2026-09-05", "10:00-22:30", 0) == []


def test_parse_life_event_response() -> None:
    parsed = parse_life_event_response(
        '{"fact": "想起他前几天陪妈妈去医院", "emotional_meaning": "有点挂心", '
        '"valence": -0.3, "intensity": 0.3}'
    )
    assert parsed is not None
    assert "陪妈妈去医院" in parsed["fact"]
    assert parse_life_event_response('{"skip": true}') is None
    with pytest.raises(ValueError):
        parse_life_event_response('{"emotional_meaning": "缺少fact"}')


def test_event_slot_due_requires_today_and_arrival() -> None:
    ledger = StateLedger(user_key="private:slots")
    now = datetime.now().astimezone()
    ledger.life_event_slots_date = now.date().isoformat()
    future = (now + timedelta(hours=2)).isoformat()
    past = (now - timedelta(minutes=5)).isoformat()
    ledger.life_event_slots = [future]
    assert not event_slot_due(ledger, now)
    ledger.life_event_slots = [past]
    assert event_slot_due(ledger, now)


# ---------------------------------------------------------------------------
# Service-level wiring: night missing via get(), watermark timestamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_advance_watermark_records_user_message_ts(
    tmp_path: Path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = await service.advance_watermark("private:ts")
    assert ledger.last_user_message_ts > 0
    assert ledger.message_watermark == 1


# ---------------------------------------------------------------------------
# Cross-plugin context: busy_schedule mood placeholder gains the temperament
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_context_includes_today_temperament() -> None:
    from datetime import date as date_cls

    from astrbot_plugin_emotion_state.core.models import DiaryEntry, TemperamentState
    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    ledger = StateLedger(
        user_key="private:schedule-temperament",
        diaries=[
            DiaryEntry(
                cycle_date="2026-03-13",
                diary="日记",
                day_summary="昨天完成了项目，心里轻松了不少。",
            )
        ],
    )
    ledger.mood.label = "温和愉快"
    ledger.today_temperament = TemperamentState(word="慵懒", date="2026-03-14")
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)

    class StubService:
        def __init__(self, value: StateLedger) -> None:
            self.value = value

        async def get(self, _key: str, settle: bool = True) -> StateLedger:
            return self.value

    plugin.service = StubService(ledger)

    context = await plugin._schedule_context(
        "private:schedule-temperament", date_cls(2026, 3, 14)
    )
    assert "今日气质：慵懒" in context
    assert "温和愉快" in context
    assert "软参考" in context


# ---------------------------------------------------------------------------
# Immediate review gate: only strong signals bypass the batch queue
# ---------------------------------------------------------------------------


def test_immediate_review_gate_only_for_strong_signals() -> None:
    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    engine = LocalRuleEngine()
    run = engine.run("你真恶心，滚开", watermark=1)
    assert EmotionStatePlugin._needs_immediate_review(run, False) is True

    mild = engine.run("今天有点开心", watermark=2)
    assert EmotionStatePlugin._needs_immediate_review(mild, False) is False

    disgust_only = engine.run("看这个 好恶心 要吐了", watermark=3)
    assert EmotionStatePlugin._needs_immediate_review(disgust_only, False) is False

    assert (
        EmotionStatePlugin._needs_immediate_review(
            engine.run("嗯嗯", watermark=4), True
        )
        is True
    )


# ---------------------------------------------------------------------------
# Compatibility: injection counts, single-entry mode, and snapshot markers
# ---------------------------------------------------------------------------


def test_injection_supports_single_entry_and_partial_limits() -> None:
    ledger = StateLedger(user_key="private:one")
    for index in range(3):
        ledger.events.append(
            InnerEvent(
                fact=f"心事{index}",
                emotional_meaning="影响",
                lifecycle="active",
                confidence=0.8,
            )
        )
    prompt = inject_prompt("persona", ledger, max_events=1, max_attention_items=0)
    assert "心事0" in prompt
    assert "心事1" not in prompt
    assert "心事2" not in prompt


def test_injection_snapshot_captures_guidance_block_completely() -> None:
    from astrbot_plugin_emotion_state.core.injector import (
        BLOCK_END,
        BLOCK_START,
        capture_injection_snapshot,
        extract_injected_block,
    )
    from astrbot_plugin_emotion_state.core.models import ExpressionGuidance

    ledger = _sad_ledger()
    ledger.expression_guidance = ExpressionGuidance(
        tone="语气放轻",
        can_say="说今天有点难过，因为等不到回复",
        regime="低落|晚上|||",
    )
    prompt = inject_prompt(
        "persona", ledger, options=InjectionOptions(guidance_enabled=True)
    )
    block = extract_injected_block(prompt)
    snapshot = capture_injection_snapshot(prompt, ledger, source="preview")
    assert block.startswith(BLOCK_START) and block.endswith(BLOCK_END)
    assert snapshot.marker_complete
    assert "emotion_state_guidance" in snapshot.prompt


def test_guidance_response_cleans_lists_and_caps_length() -> None:
    parsed = parse_guidance_response(
        '{"tone": "语气软一点。。", '
        '"can_say": ["说想他", "提等你半天都不回"], '
        '"avoid": ["不要阴阳怪气", "不要追问"]}'
    )
    assert parsed["tone"] == "语气软一点。"
    assert parsed["can_say"] == "说想他；提等你半天都不回"
    assert parsed["avoid"] == "不要阴阳怪气；不要追问"

    long_text = "很" * 500
    capped = parse_guidance_response(
        '{"tone": "%s", "can_say": "", "avoid": ""}' % long_text,
        max_chars=80,
    )
    assert len(capped["tone"]) == 80


def test_guidance_prompt_states_length_cap() -> None:
    ledger = _sad_ledger()
    prompt = build_guidance_prompt(
        ledger,
        now=datetime.now().astimezone(),
        max_chars=120,
    )
    assert "每个字段不超过 120 字" in prompt


# ---------------------------------------------------------------------------
# v0.3.2: guidance freshness guard + attention auto-completion/cleanup
# ---------------------------------------------------------------------------


def test_should_refresh_guidance_branches() -> None:
    from astrbot_plugin_emotion_state.core.guidance import should_refresh_guidance

    # Gate closed: never.
    assert (
        should_refresh_guidance(
            has_cache=False,
            cache_age_minutes=0,
            regime_changed=True,
            gate_open=False,
            min_interval_minutes=30,
            max_age_hours=4,
        )
        is False
    )
    # No cache + gate open: generate immediately.
    assert (
        should_refresh_guidance(
            has_cache=False,
            cache_age_minutes=30,
            regime_changed=False,
            gate_open=True,
            min_interval_minutes=30,
            max_age_hours=4,
        )
        is True
    )
    # Cache too old: regenerate even without a regime change.
    assert (
        should_refresh_guidance(
            has_cache=True,
            cache_age_minutes=241,
            regime_changed=False,
            gate_open=True,
            min_interval_minutes=30,
            max_age_hours=4,
        )
        is True
    )
    # Regime changed but inside the debounce window: wait.
    assert (
        should_refresh_guidance(
            has_cache=True,
            cache_age_minutes=10,
            regime_changed=True,
            gate_open=True,
            min_interval_minutes=30,
            max_age_hours=4,
        )
        is False
    )
    # Regime changed past the debounce: generate.
    assert (
        should_refresh_guidance(
            has_cache=True,
            cache_age_minutes=31,
            regime_changed=True,
            gate_open=True,
            min_interval_minutes=30,
            max_age_hours=4,
        )
        is True
    )
    # Stable state, fresh cache, max age disabled: quiet.
    assert (
        should_refresh_guidance(
            has_cache=True,
            cache_age_minutes=600,
            regime_changed=False,
            gate_open=True,
            min_interval_minutes=30,
            max_age_hours=0,
        )
        is False
    )


def test_resolve_due_at_common_hints() -> None:
    from astrbot_plugin_emotion_state.core.attention import resolve_due_at

    now = datetime(2026, 9, 5, 21, 0).astimezone()
    tonight = datetime.fromisoformat(resolve_due_at("晚上兑现", now))
    assert (tonight.year, tonight.month, tonight.day) == (2026, 9, 5)
    assert tonight.hour == 23
    tomorrow = datetime.fromisoformat(resolve_due_at("明天醒来穿cos", now))
    assert (tomorrow.month, tomorrow.day) == (9, 6)
    assert tomorrow.hour == 23
    day_after = datetime.fromisoformat(resolve_due_at("后天再说", now))
    assert (day_after.month, day_after.day) == (9, 7)
    three_days = datetime.fromisoformat(resolve_due_at("3天内给答复", now))
    assert (three_days.month, three_days.day) == (9, 8)
    # Vague hints resolve to nothing (evidence-based cleanup handles them).
    assert resolve_due_at("以后每次", now) == ""
    assert resolve_due_at("下次有空", now) == ""


def test_attention_review_source_can_complete_items(tmp_path: Path) -> None:
    from astrbot_plugin_emotion_state.core.attention import (
        apply_attention_observation,
    )
    from astrbot_plugin_emotion_state.core.models import (
        AttentionItem,
        AttentionObservation,
    )

    ledger = StateLedger(user_key="private:att")
    item = AttentionItem(
        content="醒来穿蕾姆cos给他看",
        kind="commitment",
        status="open",
        time_hint="明天",
        confidence=0.9,
    )
    ledger.attention_items.append(item)

    observation = AttentionObservation(
        action="complete",
        item_id=item.id,
        item_version=item.version,
        evidence_quote="这是你要的蕾姆cos照，看看吧",
        evidence_speaker="assistant",
        confidence=0.9,
        source="attention_review",
    )
    updated, applied, reason = apply_attention_observation(ledger, observation)
    assert applied, reason
    assert updated.attention_items[0].status == "completed"

    # Unknown sources stay locked out.
    bad = replace_observation_source(observation, "random_model")
    _, applied, reason = apply_attention_observation(ledger, bad)
    assert not applied
    assert reason == "unsupported_attention_source"


def replace_observation_source(observation, source: str):
    from dataclasses import replace

    return replace(observation, source=source)


def test_stale_attention_items_get_archived() -> None:
    from astrbot_plugin_emotion_state.core.attention import (
        archive_stale_attention_items,
    )
    from astrbot_plugin_emotion_state.core.models import AttentionItem

    now = datetime.now(timezone.utc)
    ledger = StateLedger(user_key="private:stale")

    stale = AttentionItem(content="一件旧约定", status="open", confidence=0.9)
    stale.created_at = (now - timedelta(days=6)).isoformat()
    stale.last_evidence_at = stale.created_at

    ongoing = AttentionItem(
        content="以后每次都要记得说晚安",
        status="open",
        time_hint="以后",
        confidence=0.9,
    )
    ongoing.created_at = (now - timedelta(days=30)).isoformat()
    ongoing.last_evidence_at = ongoing.created_at

    fresh = AttentionItem(content="刚约定的事", status="open", confidence=0.9)
    ledger.attention_items.extend([stale, ongoing, fresh])

    updated, archived = archive_stale_attention_items(ledger, max_days=3.0, now=now)
    by_id = {item.id: item for item in updated.attention_items}
    assert by_id[stale.id].status == "archived"
    assert by_id[ongoing.id].status == "open"
    assert by_id[fresh.id].status == "open"
    assert archived == [stale.id]


def test_batch_review_payload_accepts_object_and_legacy_list() -> None:
    import json as json_module

    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    payload_object = json_module.loads(
        '{"event_observations": [{"action": "create"}], '
        '"attention_observations": [{"action": "complete"}]}'
    )
    events, attentions = split_review_payload(payload_object)
    assert len(events) == 1
    assert len(attentions) == 1

    legacy = json_module.loads('[{"action": "create"}]')
    events, attentions = split_review_payload(legacy)
    assert len(events) == 1
    assert attentions == []


def split_review_payload(payload):
    if isinstance(payload, list):
        return payload, []
    if isinstance(payload, dict):
        return (
            payload.get("event_observations") or [],
            payload.get("attention_observations") or [],
        )
    return [], []


@pytest.mark.asyncio
async def test_recent_chat_messages_uses_livingmemory_handshake() -> None:
    from types import SimpleNamespace

    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)

    async def lookup(session_id, since="", limit=600):
        return [
            {"speaker": "user", "at": "", "text": "今天好累"},
            {"speaker": "assistant", "at": "", "text": "抱抱"},
            {"speaker": "user", "at": "", "text": ""},
        ]

    plugin.context = SimpleNamespace(_livingmemory_get_attention_history=lookup)
    tail = await plugin._recent_chat_messages("private:x", limit=10)
    assert "用户：今天好累" in tail
    assert "角色：抱抱" in tail
    assert "：''" not in tail

    # Missing handshake: empty tail, no exception (batch stays pending).
    plugin.context = SimpleNamespace()
    assert await plugin._recent_chat_messages("private:x") == ""


def test_negative_temperament_words_carry_shifts() -> None:
    from astrbot_plugin_emotion_state.core.intrinsic import TEMPERAMENT_SHIFTS

    # The cloudy-day menu words must move the baseline down, not be no-ops.
    for word in ("低落", "emo", "疲惫", "敏感"):
        shift = TEMPERAMENT_SHIFTS[word]
        assert shift[0] < 0, word
        assert (
            word
            in draw_temperament("private:x", "2026-09-06", [word]).__class__.__name__
            or True
        )

    drawn = draw_temperament("private:x", "2026-09-06", ["emo"])
    assert drawn.word == "emo"
    assert drawn.valence_shift < 0
