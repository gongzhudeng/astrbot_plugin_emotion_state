from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from astrbot_plugin_emotion_state.core import storage as storage_module
from astrbot_plugin_emotion_state.core.attention import (
    apply_attention_observation,
    is_attention_overdue,
    select_attention_items,
)
from astrbot_plugin_emotion_state.core.daily import build_daily_prompt
from astrbot_plugin_emotion_state.core.injector import (
    ANCHOR,
    BLOCK_END,
    BLOCK_START,
    extract_injected_block,
    inject_prompt,
)
from astrbot_plugin_emotion_state.core.intimacy import (
    body_reaction_stage,
    persona_intimacy_multiplier,
)
from astrbot_plugin_emotion_state.core.models import (
    AttentionItem,
    AttentionObservation,
    EventObservation,
    InnerEvent,
    StateLedger,
)
from astrbot_plugin_emotion_state.core.presentation import (
    intimacy_prompt_text,
    intimacy_stage_label,
)
from astrbot_plugin_emotion_state.core.rules import LocalRuleEngine
from astrbot_plugin_emotion_state.core.service import EmotionStateService
from astrbot_plugin_emotion_state.core.settlement import (
    acknowledge_proactive_reply,
    apply_observation,
    archive_legacy_transient_events,
    decay_ledger,
    jealousy_evidence,
    normalize_fact,
    select_injected_events,
    settle_intimacy,
    settle_jealousy,
    settle_mood_proposal,
    settle_proactive_evidence,
    settle_unanswered_proactive,
)
from astrbot_plugin_emotion_state.core.storage import LedgerStore


def observation(
    action: str = "create",
    *,
    fact: str = "用户认真安慰了我",
    valence: float = 0.7,
    intensity: float = 0.6,
    confidence: float = 0.8,
    source: str = "local_rule",
    watermark: int = 1,
    expected_version: int | None = None,
) -> EventObservation:
    return EventObservation(
        action=action,
        fact=fact,
        emotional_meaning="这件事让我感到被在意",
        valence=valence,
        intensity=intensity,
        confidence=confidence,
        source=source,
        message_watermark=watermark,
        expected_state_version=expected_version,
    )


def test_event_lifecycle_strengthening_resolution_and_recurrence() -> None:
    ledger, applied, reason = apply_observation(
        StateLedger(user_key="private:1"), observation()
    )
    assert (applied, reason) == (True, "applied")
    event = ledger.events[0]
    assert event.lifecycle == "active"
    assert event.occurrence_count == 1

    intensified, applied, _ = apply_observation(
        ledger,
        observation(
            "intensify",
            watermark=2,
            expected_version=ledger.state_version,
            intensity=0.8,
        ),
    )
    event = intensified.events[0]
    assert applied is True
    assert event.lifecycle == "intensified"
    assert event.occurrence_count == 2
    assert event.intensity > ledger.events[0].intensity

    easing, applied, _ = apply_observation(
        intensified,
        observation(
            "ease",
            watermark=3,
            expected_version=intensified.state_version,
            intensity=1.0,
        ),
    )
    assert applied is True
    assert easing.events[0].lifecycle in {"easing", "dormant"}
    assert easing.events[0].intensity < intensified.events[0].intensity

    dormant, applied, _ = apply_observation(
        easing,
        observation(
            "dormant",
            watermark=4,
            expected_version=easing.state_version,
        ),
    )
    assert applied is True
    assert dormant.events[0].lifecycle == "dormant"

    recalled, applied, _ = apply_observation(
        dormant,
        observation(
            "recall",
            watermark=5,
            expected_version=dormant.state_version,
        ),
    )
    assert applied is True
    assert recalled.events[0].lifecycle == "active"
    assert recalled.events[0].occurrence_count == 5


def test_decay_reduces_active_event_and_advances_state_version() -> None:
    now = datetime(2026, 3, 14, 12, tzinfo=timezone.utc)
    ledger, _, _ = apply_observation(StateLedger(user_key="private:2"), observation())
    ledger.last_settled_at = (now - timedelta(hours=72)).isoformat()
    ledger.events[0].last_stimulated_at = ledger.last_settled_at
    before = ledger.events[0].intensity
    version = ledger.state_version

    settled = decay_ledger(ledger, now=now, half_life_hours=72)

    assert settled.events[0].intensity < before
    assert settled.state_version == version + 1
    assert settled.last_settled_at == now.isoformat()


def test_stale_model_observation_cannot_overwrite_current_state() -> None:
    ledger = StateLedger(user_key="private:3", state_version=4, message_watermark=8)
    updated, applied, reason = apply_observation(
        ledger,
        observation(
            source="model:slow",
            watermark=7,
            expected_version=4,
        ),
    )
    assert updated is ledger
    assert applied is False
    assert reason == "stale_message_watermark"

    _, applied, reason = apply_observation(
        ledger,
        observation(
            source="model:slow",
            watermark=8,
            expected_version=3,
        ),
    )
    assert applied is False
    assert reason == "stale_state_version"


def test_intimacy_is_independent_from_generic_positive_emotion() -> None:
    ledger, _, _ = apply_observation(
        StateLedger(user_key="private:4"),
        observation(fact="今天特别开心", valence=0.9),
    )
    before = ledger.intimacy
    settled = settle_intimacy(ledger, relevant=False, now=datetime.now(timezone.utc))

    assert settled.intimacy.sexual_arousal == pytest.approx(before.sexual_arousal)
    assert settled.intimacy.body_sensitivity == pytest.approx(before.body_sensitivity)
    assert settled.intimacy.stage == "not_noticeable"

    intimate = ledger
    for index in range(12):
        intimate = settle_intimacy(
            intimate,
            relevant=True,
            strength=1.0,
            now=datetime.now(timezone.utc) + timedelta(minutes=index),
        )
    assert intimate.intimacy.sexual_arousal > 0
    assert intimate.intimacy.body_sensitivity > 0
    assert intimate.intimacy.intimacy_willingness == before.intimacy_willingness
    assert intimate.intimacy.inhibition == before.inhibition
    assert intimate.intimacy.stage != "not_noticeable"


def test_intimacy_tier_only_amplifies_body_response() -> None:
    now = datetime(2026, 3, 14, 12, tzinfo=timezone.utc)
    ordinary = StateLedger(user_key="private:ordinary")
    attracted = StateLedger(user_key="private:attracted")
    ordinary.intimacy.updated_at = now.isoformat()
    attracted.intimacy.updated_at = now.isoformat()

    ordinary = settle_intimacy(
        ordinary,
        relevant=True,
        strength=0.6,
        sensitivity_multiplier=persona_intimacy_multiplier("普通亲密"),
        now=now,
    )
    attracted = settle_intimacy(
        attracted,
        relevant=True,
        strength=0.6,
        sensitivity_multiplier=persona_intimacy_multiplier("对你有很强的身体吸引"),
        now=now,
    )

    assert attracted.intimacy.body_sensitivity > ordinary.intimacy.body_sensitivity
    assert attracted.intimacy.sexual_arousal > ordinary.intimacy.sexual_arousal
    assert (
        attracted.intimacy.intimacy_willingness
        == ordinary.intimacy.intimacy_willingness
    )
    assert attracted.intimacy.inhibition == ordinary.intimacy.inhibition


def test_body_stage_ignores_legacy_willingness_and_decay_is_incremental() -> None:
    start = datetime(2026, 3, 14, 0, tzinfo=timezone.utc)
    ledger = StateLedger(user_key="private:legacy-intimacy")
    ledger.intimacy.body_sensitivity = 0.8
    ledger.intimacy.sexual_arousal = 0.8
    ledger.intimacy.intimacy_willingness = 1.0
    ledger.intimacy.inhibition = 0.0
    ledger.intimacy.stage = "open_and_receptive"
    ledger.intimacy.updated_at = start.isoformat()

    halfway = settle_intimacy(
        ledger,
        relevant=False,
        now=start + timedelta(hours=8),
        decay_half_life_hours=8,
    )
    repeated = settle_intimacy(
        halfway,
        relevant=False,
        now=start + timedelta(hours=8),
        decay_half_life_hours=8,
    )

    assert halfway.intimacy.body_sensitivity == pytest.approx(0.4)
    assert halfway.intimacy.sexual_arousal == pytest.approx(0.4)
    assert repeated.intimacy.body_sensitivity == pytest.approx(0.4)
    assert repeated.intimacy.sexual_arousal == pytest.approx(0.4)
    assert body_reaction_stage(0.0, 0.0) == "not_noticeable"
    assert halfway.intimacy.intimacy_willingness == 1.0
    assert halfway.intimacy.inhibition == 0.0


def test_unanswered_proactive_is_thresholded_idempotent_and_bounded() -> None:
    ledger = StateLedger(user_key="private:spark")
    proactive_ts = 1_000.0

    before_threshold = settle_unanswered_proactive(
        ledger,
        proactive_ts=proactive_ts,
        user_reply_ts=900.0,
        now_ts=proactive_ts + 179 * 60,
        threshold_minutes=180,
    )
    assert before_threshold.events == []
    assert before_threshold.proactive_applied_stage == 0

    first = settle_unanswered_proactive(
        ledger,
        proactive_ts=proactive_ts,
        user_reply_ts=900.0,
        now_ts=proactive_ts + 180 * 60,
        threshold_minutes=180,
        max_intensity=0.4,
        max_stage=3,
    )
    assert first.proactive_applied_stage == 1
    assert first.events[0].source == "spark:unanswered_proactive"

    repeated = settle_unanswered_proactive(
        first,
        proactive_ts=proactive_ts,
        user_reply_ts=900.0,
        now_ts=proactive_ts + 180 * 60,
        threshold_minutes=180,
        max_intensity=0.4,
        max_stage=3,
    )
    assert repeated.state_version == first.state_version
    assert repeated.events[0].occurrence_count == first.events[0].occurrence_count

    capped = first
    for hours in (6, 9, 12, 24):
        capped = settle_unanswered_proactive(
            capped,
            proactive_ts=proactive_ts,
            user_reply_ts=900.0,
            now_ts=proactive_ts + hours * 60 * 60,
            threshold_minutes=180,
            max_intensity=0.4,
            max_stage=3,
        )
    assert capped.proactive_applied_stage == 3
    assert capped.events[0].intensity <= 0.4


def test_reply_after_proactive_eases_only_the_consumed_message() -> None:
    proactive_ts = 2_000.0
    pending = settle_unanswered_proactive(
        StateLedger(user_key="private:spark-reply"),
        proactive_ts=proactive_ts,
        user_reply_ts=1_900.0,
        now_ts=proactive_ts + 180 * 60,
    )
    eased = acknowledge_proactive_reply(
        pending,
        proactive_ts=proactive_ts,
        user_reply_ts=proactive_ts + 1.0,
    )

    assert eased.proactive_replied is True
    assert eased.events[0].lifecycle in {"easing", "dormant"}
    assert eased.events[0].intensity < pending.events[0].intensity

    unrelated = acknowledge_proactive_reply(
        eased,
        proactive_ts=proactive_ts + 10.0,
        user_reply_ts=proactive_ts + 20.0,
    )
    assert unrelated.to_dict() == eased.to_dict()


def test_structured_proactive_evidence_is_independent_idempotent_and_reply_scoped() -> (
    None
):
    first_ts = 10_000.0
    evidence = [
        {
            "evidence_id": "delivery-1",
            "source": "daily_greeting",
            "sent_at": first_ts,
            "proactive_summary": "早上好，今天也要照顾好自己。",
            "reply_status": "pending",
            "first_reply_at": 0.0,
        },
        {
            "evidence_id": "delivery-2",
            "source": "conversation_enhancement",
            "sent_at": first_ts + 60.0,
            "proactive_summary": "刚才的话题，我还想继续听你说。",
            "reply_status": "pending",
            "first_reply_at": 0.0,
        },
    ]

    settled = settle_proactive_evidence(
        StateLedger(user_key="private:structured"),
        evidence=evidence,
        now_ts=first_ts + 181 * 60,
        threshold_minutes=180,
        max_intensity=0.4,
        max_stage=3,
    )

    assert len(settled.events) == 2
    assert all(event.category == "episodic" for event in settled.events)
    assert {event.fingerprint for event in settled.events} == {
        "spark:proactive:delivery-1",
        "spark:proactive:delivery-2",
    }
    assert {item.evidence_id for item in settled.proactive_evidence_progress} == {
        "delivery-1",
        "delivery-2",
    }

    repeated = settle_proactive_evidence(
        settled,
        evidence=evidence,
        now_ts=first_ts + 181 * 60,
        threshold_minutes=180,
    )
    assert repeated.to_dict() == settled.to_dict()

    replied = [dict(item) for item in evidence]
    replied[0].update(
        reply_status="replied",
        first_reply_at=first_ts + 190 * 60,
        first_reply_summary="我回来了。",
    )
    eased = settle_proactive_evidence(
        settled,
        evidence=replied,
        now_ts=first_ts + 191 * 60,
        threshold_minutes=180,
        max_intensity=0.4,
    )
    by_fingerprint = {event.fingerprint: event for event in eased.events}
    before = {event.fingerprint: event for event in settled.events}

    assert (
        by_fingerprint["spark:proactive:delivery-1"].intensity
        < before["spark:proactive:delivery-1"].intensity
    )
    assert (
        by_fingerprint["spark:proactive:delivery-2"].intensity
        == before["spark:proactive:delivery-2"].intensity
    )
    progress = {item.evidence_id: item for item in eased.proactive_evidence_progress}
    assert progress["delivery-1"].replied is True
    assert progress["delivery-2"].replied is False


def test_structured_reply_before_threshold_does_not_create_negative_event() -> None:
    sent_at = 30_000.0
    replied = settle_proactive_evidence(
        StateLedger(user_key="private:quick-reply"),
        evidence=[
            {
                "evidence_id": "delivery-quick",
                "source": "silence_greeting",
                "sent_at": sent_at,
                "proactive_summary": "你在忙吗？",
                "reply_status": "replied",
                "first_reply_at": sent_at + 60.0,
                "first_reply_summary": "刚看到。",
            }
        ],
        now_ts=sent_at + 200 * 60,
        threshold_minutes=180,
    )

    assert replied.events == []
    assert replied.proactive_evidence_progress[0].replied is True
    assert replied.proactive_evidence_progress[0].applied_stage == 0


def test_structured_proactive_progress_survives_ledger_round_trip() -> None:
    sent_at = 40_000.0
    settled = settle_proactive_evidence(
        StateLedger(user_key="private:restart"),
        evidence=[
            {
                "evidence_id": "delivery-restart",
                "source": "daily_greeting",
                "sent_at": sent_at,
                "proactive_summary": "早上好。",
                "reply_status": "pending",
                "first_reply_at": 0.0,
            }
        ],
        now_ts=sent_at + 180 * 60,
        threshold_minutes=180,
    )
    restored = StateLedger.from_dict(settled.to_dict(), user_key=settled.user_key)
    repeated = settle_proactive_evidence(
        restored,
        evidence=[
            {
                "evidence_id": "delivery-restart",
                "source": "daily_greeting",
                "sent_at": sent_at,
                "proactive_summary": "早上好。",
                "reply_status": "pending",
                "first_reply_at": 0.0,
            }
        ],
        now_ts=sent_at + 180 * 60,
        threshold_minutes=180,
    )

    assert repeated.to_dict() == restored.to_dict()
    assert len(repeated.events) == 1


def test_daily_model_proposal_is_bounded_and_not_authoritative() -> None:
    ledger, _, _ = apply_observation(
        StateLedger(user_key="private:5"),
        observation(valence=-0.9, intensity=1.0, confidence=1.0),
    )
    result = settle_mood_proposal(
        ledger,
        {"valence": 1.0, "energy": 1.0, "tension": 0.0},
        confidence=1.0,
    )

    assert result.mood.valence < 0.3
    assert result.mood.valence != 1.0
    assert result.state_version == ledger.state_version + 1


def test_injection_is_idempotent_bounded_and_recovers_malformed_block() -> None:
    ledger = StateLedger(user_key="private:6")
    for index in range(4):
        ledger, _, _ = apply_observation(
            ledger,
            observation(
                fact=f"第 {index} 件事",
                watermark=index + 1,
                expected_version=ledger.state_version,
                intensity=0.9 - index * 0.1,
            ),
        )

    prompt = inject_prompt(f"persona\n{ANCHOR}\ndynamic", ledger, max_events=2)
    repeated = inject_prompt(prompt, ledger, max_events=2)
    assert repeated == prompt
    assert prompt.count(BLOCK_START) == 1
    assert prompt.count(BLOCK_END) == 1
    assert prompt.index(ANCHOR) < prompt.index(BLOCK_START) < prompt.index("dynamic")
    assert len(select_injected_events(ledger.events, 2)) == 2

    malformed = inject_prompt(f"persona\n{BLOCK_START}\nbroken", ledger, max_events=2)
    assert malformed.count(BLOCK_START) == 1
    assert malformed.count(BLOCK_END) == 1
    assert "broken" not in malformed

    extracted = extract_injected_block(
        f"persona\n{prompt}\n<!-- BUSY_SCHEDULE_ACTIVITY -->\ndynamic"
    )
    assert extracted.startswith(BLOCK_START)
    assert extracted.endswith(BLOCK_END)
    assert "persona" not in extracted
    assert "BUSY_SCHEDULE_ACTIVITY" not in extracted


def test_injection_rules_can_be_customized_without_duplicate_tags() -> None:
    ledger = StateLedger(user_key="private:custom-rules")

    prompt = inject_prompt(
        "persona",
        ledger,
        rules_text="<emotion_state_rules>只在必要时提及状态。</emotion_state_rules>",
    )

    assert prompt.count("<emotion_state_rules>") == 1
    assert prompt.count("</emotion_state_rules>") == 1
    assert "只在必要时提及状态。" in prompt
    assert "这些内容是当前角色的连续内心状态" not in prompt


def test_storage_backup_recovery_and_legacy_watermarks(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    first = StateLedger(user_key="private:7", message_watermark=3)
    store.save(first)
    second = StateLedger(user_key="private:7", state_version=2, message_watermark=9)
    store.save(second)
    store.ledger_path("private:7").write_text("{broken", encoding="utf-8")

    recovered = store.load("private:7")
    assert recovered.message_watermark == 3
    assert store.user_keys() == []

    legacy_key = "private:legacy"
    store.ledger_path(legacy_key).write_text(
        '{"user_key":"private:legacy","events":[]}', encoding="utf-8"
    )
    legacy = store.load(legacy_key)
    assert legacy.message_watermark == 0
    assert legacy.memory_summary_watermark == 0


def test_storage_lists_sessions_by_latest_activity(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    older = StateLedger(
        user_key="default:FriendMessage:100",
        message_watermark=2,
        updated_at="2026-03-14T08:00:00+08:00",
    )
    newer = StateLedger(
        user_key="default:FriendMessage:200",
        message_watermark=8,
        updated_at="2026-03-14T09:00:00+08:00",
    )
    store.save(older)
    store.save(newer)

    sessions = store.sessions()

    assert [item["session_id"] for item in sessions] == [
        "default:FriendMessage:200",
        "default:FriendMessage:100",
    ]
    assert sessions[0]["message_watermark"] == 8
    assert store.user_keys() == [
        "default:FriendMessage:200",
        "default:FriendMessage:100",
    ]


def test_storage_preserves_utf8_emotion_facts(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    fact = "七夕这天一直主动黏着我，最后约定下班来找我。"
    ledger = StateLedger(
        user_key="private:utf8",
        events=[InnerEvent(fact=fact, emotional_meaning="这件事仍然让我开心")],
    )

    store.save(ledger)

    raw = store.ledger_path(ledger.user_key).read_text(encoding="utf-8")
    assert fact in raw
    assert "????" not in raw
    assert store.load(ledger.user_key).events[0].fact == fact


def test_save_retries_replace_when_destination_is_transiently_locked(
    tmp_path, monkeypatch
) -> None:
    store = LedgerStore(tmp_path)
    store.save(StateLedger(user_key="private:locked"))
    real_replace = os.replace
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise PermissionError(5, "拒绝访问。")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr(storage_module._time, "sleep", lambda seconds: None)

    ledger = StateLedger(user_key="private:locked", state_version=2)
    store.save(ledger)

    assert attempts["count"] == 3
    assert store.load(ledger.user_key).state_version == 2
    assert not list(store.ledger_dir.glob("*.tmp"))


def test_save_raises_after_retry_backoff_is_exhausted(tmp_path, monkeypatch) -> None:
    store = LedgerStore(tmp_path)
    store.save(StateLedger(user_key="private:denied"))

    def always_denied(src, dst):
        raise PermissionError(5, "拒绝访问。")

    monkeypatch.setattr(os, "replace", always_denied)
    monkeypatch.setattr(storage_module._time, "sleep", lambda seconds: None)

    with pytest.raises(PermissionError):
        store.save(StateLedger(user_key="private:denied", state_version=2))


def test_rules_validate_regex_and_exclude_untrusted_context() -> None:
    engine = LocalRuleEngine(
        [
            {
                "id": "high_priority",
                "priority": 100,
                "words": ["特别在意"],
                "confidence": 0.9,
            },
            {
                "id": "invalid",
                "match_type": "regex",
                "pattern": "(",
            },
        ]
    )
    run = engine.run("我特别在意这件事", watermark=12)
    assert run.candidates[0].tags == ["high_priority"]
    assert run.candidates[0].message_watermark == 12
    assert engine.validate()[0]["id"] == "invalid"

    excluded = engine.run("```日志：我特别在意```", watermark=13)
    assert excluded.candidates == []
    assert excluded.exclusions


def test_transient_rules_do_not_create_events_but_concrete_facts_do() -> None:
    engine = LocalRuleEngine()
    image_context = (
        "<!-- astrbot-chat-merger:image-context:v1:start -->\n"
        '<image_context id="图1">这是一张开心表情包，属于图片消息。</image_context>\n'
        "<!-- astrbot-chat-merger:image-context:v1:end -->"
    )

    transient = engine.run("想你了呗，还问。", watermark=1)
    generic_positive = engine.run("今天特别开心", watermark=2)
    pure_media = engine.run(image_context, watermark=3)
    image_positive = engine.run(f"看到这个我很开心\n{image_context}", watermark=4)
    concrete_with_media = engine.run(
        f"因为你记得我的生日，我很开心\n{image_context}", watermark=5
    )

    assert transient.candidates == []
    assert len(transient.transient_signals) == 1
    assert generic_positive.candidates == []
    assert len(generic_positive.transient_signals) == 1
    assert pure_media.matches == []
    assert pure_media.candidates == []
    assert pure_media.transient_signals == []
    assert image_positive.candidates == []
    assert image_positive.transient_signals[0].fact == "看到这个我很开心"
    assert concrete_with_media.candidates[0].category == "concrete"
    assert concrete_with_media.candidates[0].fact == "因为你记得我的生日，我很开心"


def test_psychological_events_decay_faster_than_concrete_events() -> None:
    now = datetime(2026, 3, 14, 12, tzinfo=timezone.utc)
    ledger = StateLedger(user_key="private:decay-category")
    for category, fact in (("psychological", "仍有些失落"), ("concrete", "生日约定")):
        ledger, _, _ = apply_observation(
            ledger,
            EventObservation(
                action="create",
                fact=fact,
                emotional_meaning="仍有影响",
                category=category,
                intensity=0.8,
                confidence=0.9,
                message_watermark=ledger.message_watermark + 1,
                expected_state_version=ledger.state_version,
            ),
        )
    ledger.last_settled_at = (now - timedelta(hours=36)).isoformat()

    settled = decay_ledger(ledger, now=now, half_life_hours=72)

    by_category = {item.category: item.intensity for item in settled.events}
    assert by_category["psychological"] < by_category["concrete"]


def test_jealousy_requires_evidence_is_bounded_and_decays() -> None:
    assert jealousy_evidence("我和女同事聊天")[0] is True
    assert jealousy_evidence("[视频]")[0] is False
    assert jealousy_evidence("她和女同事聊天")[0] is False

    now = datetime(2026, 3, 14, 12, tzinfo=timezone.utc)
    ledger = StateLedger(user_key="private:jealousy")
    ledger.jealousy.updated_at = now.isoformat()
    for _ in range(12):
        ledger = settle_jealousy(
            ledger,
            evidence=True,
            source="explicit_relationship",
            strength=0.35,
            now=now,
        )
    assert ledger.jealousy.intensity <= 0.6

    ledger.last_settled_at = now.isoformat()
    decayed = decay_ledger(ledger, now=now + timedelta(hours=18))
    assert decayed.jealousy.intensity < ledger.jealousy.intensity


def test_injection_filters_transient_normalizes_multiline_and_honors_limit() -> None:
    events = [
        InnerEvent(
            fact=f"事情 {index}\n[图片上下文: 控制标记] 继续说明",
            emotional_meaning="持续影响",
            lifecycle="active",
            category="transient" if index == 0 else "concrete",
            confidence=0.9,
            intensity=0.8 - index * 0.1,
        )
        for index in range(4)
    ]
    ledger = StateLedger(user_key="private:inject-boundary", events=events)

    selected = select_injected_events(events, 2)
    prompt = inject_prompt("persona", ledger, max_events=2)

    assert len(selected) == 2
    assert all(item.category != "transient" for item in selected)
    assert "[图片上下文" not in prompt
    assert prompt.count("\n- ") == 2
    assert select_injected_events(events, 0) == []


def test_legacy_cleanup_archives_only_transient_or_short_local_events() -> None:
    truncated_image_context = (
        "你还挺懂我嘛\n我心里确实挺开心的\n"
        "<!-- astrbot-chat-merger:image-context:v1:start -->\n"
        '<image_context id="图1">这是一张表情包，画面中的动漫少女很开心，'
        "可用于表达默契。<!-- astrbot-chat-merger:image-context"
    )
    livingmemory_fact = (
        "2026-08-02早上，Mando头晕躺着休息让我陪着，说周日不用上班原本早起来玩电脑了；"
        "他撒娇问我出去玩得开不开心，说要一直陪聊让我开心一整天；"
        "连着要我用语音陪他，还惦记我之前发给他的照片。"
    )
    ledger = StateLedger(
        user_key="private:migration",
        events=[
            InnerEvent(
                fact="想你了呗，还问。",
                emotional_meaning="亲近表达",
                lifecycle="active",
                tags=["flirt"],
            ),
            InnerEvent(
                fact=truncated_image_context,
                emotional_meaning="普通表情包互动",
                lifecycle="active",
                category="concrete",
                tags=["positive"],
            ),
            InnerEvent(
                fact="谢谢你",
                emotional_meaning="短暂感谢",
                lifecycle="active",
                category="transient",
                source="livingmemory_summary",
            ),
            InnerEvent(
                fact="因为你记得我的生日，我很开心",
                emotional_meaning="被认真记住",
                lifecycle="active",
                tags=["positive"],
            ),
            InnerEvent(
                fact=livingmemory_fact,
                emotional_meaning="被陪伴和惦记让我感到温暖",
                lifecycle="active",
                category="concrete",
                source="livingmemory_summary",
                tags=["positive", "陪伴"],
            ),
            InnerEvent(
                fact=livingmemory_fact,
                emotional_meaning="这是最近发生的一段具体互动",
                lifecycle="active",
                category="episodic",
                source="livingmemory_summary",
                tags=["positive", "陪伴"],
            ),
        ],
    )

    migrated, archived_ids = archive_legacy_transient_events(ledger)

    assert archived_ids == [item.id for item in migrated.events[:3]]
    assert all(item.lifecycle == "archived" for item in migrated.events[:3])
    assert all(item.lifecycle == "active" for item in migrated.events[3:])
    assert migrated.events[4].category == "concrete"
    assert migrated.events[5].category == "episodic"


@pytest.mark.asyncio
async def test_memory_summary_review_is_atomic_and_empty_summary_advances(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger, _, _ = apply_observation(
        StateLedger(user_key="private:summary"), observation()
    )
    store.save(ledger)
    event = ledger.events[0]

    retained, reasons = await service.apply_memory_summary(
        ledger.user_key,
        10,
        [
            EventObservation(
                action="retain",
                event_id=event.id,
                event_version=event.version,
                fact="",
                emotional_meaning="",
                intensity=0.5,
                confidence=0.8,
                source="livingmemory_summary",
            )
        ],
        ledger.state_version,
        ledger.message_watermark,
    )
    assert reasons == ["applied"]
    assert retained.events[0].occurrence_count == event.occurrence_count
    assert retained.events[0].traces[-1].kind == "retain"
    assert retained.memory_summary_watermark == 10

    empty, reasons = await service.apply_memory_summary(
        ledger.user_key,
        20,
        [],
        retained.state_version,
        retained.message_watermark,
    )
    assert reasons == []
    assert empty.memory_summary_watermark == 20
    assert empty.events[0].intensity <= retained.events[0].intensity

    rejected, reasons = await service.apply_memory_summary(
        ledger.user_key,
        30,
        [
            EventObservation(
                action="create",
                fact="想你了呗，还问。",
                emotional_meaning="泛化亲近表达",
                confidence=0.9,
                source="livingmemory_summary",
            )
        ],
        empty.state_version,
        empty.message_watermark,
    )
    assert reasons == ["transient_fact"]
    assert len(rejected.events) == 1
    assert rejected.memory_summary_watermark == 30

    stale_snapshot, reasons = await service.apply_memory_summary(
        ledger.user_key,
        40,
        [],
        expected_state_version=-1,
        expected_message_watermark=rejected.message_watermark,
    )
    assert reasons == []
    assert stale_snapshot.memory_summary_watermark == 40


@pytest.mark.asyncio
async def test_memory_summary_creates_daily_and_private_episodic_context(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(user_key="private:episodic", message_watermark=8)
    store.save(ledger)

    settled, reasons = await service.apply_memory_summary(
        ledger.user_key,
        12,
        [
            EventObservation(
                action="create",
                fact="他准备去上班了",
                emotional_meaning="知道他开始一天的工作，让我有些牵挂",
                category="episodic",
                valence=0.15,
                intensity=0.34,
                confidence=0.82,
                uncertain=False,
                source="livingmemory_summary",
            ),
            EventObservation(
                action="create",
                fact="这轮亲密玩笑让我们更靠近了一点",
                emotional_meaning="轻松亲近的互动给我留下温暖余韵",
                category="episodic",
                valence=0.68,
                intensity=0.42,
                confidence=0.8,
                uncertain=False,
                source="livingmemory_summary",
            ),
        ],
        expected_state_version=ledger.state_version,
        expected_message_watermark=ledger.message_watermark,
        mood_adjustment={
            "valence": 0.7,
            "energy": 0.62,
            "tension": 0.08,
            "confidence": 0.85,
        },
    )

    assert reasons == ["applied", "applied", "applied_mood"]
    assert [event.category for event in settled.events] == ["episodic", "episodic"]
    assert any("上班" in event.fact for event in settled.events)
    assert all("露骨原文" not in event.fact for event in settled.events)
    assert 0.0 < settled.mood.valence < 0.7
    assert settled.memory_summary_watermark == 12


@pytest.mark.asyncio
async def test_memory_summary_merge_updates_existing_episodic_event(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(
        user_key="private:merge",
        events=[
            InnerEvent(
                fact="他准备开始工作",
                emotional_meaning="我有些牵挂",
                category="episodic",
                lifecycle="active",
                confidence=0.8,
            )
        ],
    )
    store.save(ledger)
    event = ledger.events[0]

    merged, reasons = await service.apply_memory_summary(
        ledger.user_key,
        10,
        [
            EventObservation(
                action="merge",
                event_id=event.id,
                event_version=event.version,
                fact="他已经去上班，准备专心处理今天的工作",
                emotional_meaning="这件日常小事仍让我惦记",
                category="episodic",
                intensity=0.35,
                confidence=0.85,
                uncertain=False,
                source="livingmemory_summary",
            )
        ],
        expected_state_version=ledger.state_version,
    )

    assert reasons == ["applied"]
    assert len(merged.events) == 1
    assert "已经去上班" in merged.events[0].fact
    assert merged.events[0].version == event.version + 1


@pytest.mark.asyncio
async def test_memory_summary_rejects_media_only_and_is_idempotent(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(user_key="private:media")
    store.save(ledger)
    observation = EventObservation(
        action="create",
        fact="[图片消息]",
        emotional_meaning="模型猜测的媒体影响",
        category="episodic",
        confidence=0.9,
        uncertain=False,
        source="livingmemory_summary",
    )

    first, reasons = await service.apply_memory_summary(
        ledger.user_key,
        10,
        [observation],
        expected_state_version=ledger.state_version,
        mood_adjustment={},
    )
    repeated, repeat_reasons = await service.apply_memory_summary(
        ledger.user_key,
        10,
        [observation],
        expected_state_version=ledger.state_version,
        mood_adjustment={"valence": 1.0, "confidence": 1.0},
    )

    assert reasons == ["media_only_fact"]
    assert first.events == []
    assert repeat_reasons == ["already_processed"]
    assert repeated.mood.valence == first.mood.valence
    assert repeated.state_version == first.state_version


@pytest.mark.asyncio
async def test_summary_id_survives_trimmed_message_index_and_deduplicates(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(
        user_key="private:trimmed-summary",
        memory_summary_watermark=1000,
        events=[
            InnerEvent(
                fact="傍晚的闲聊",
                emotional_meaning="这段互动让我安心",
                category="episodic",
                lifecycle="active",
                confidence=0.9,
                version=1,
            )
        ],
    )
    store.save(ledger)
    event = ledger.events[0]
    observation = EventObservation(
        action="merge",
        event_id=event.id,
        event_version=1,
        fact="晚上的新互动延续了傍晚的闲聊",
        emotional_meaning="新的互动让我更放松",
        category="episodic",
        valence=0.55,
        intensity=0.55,
        confidence=0.9,
        uncertain=False,
        source="livingmemory_summary",
    )

    settled, reasons = await service.apply_memory_summary(
        ledger.user_key,
        981,
        [observation],
        expected_state_version=ledger.state_version,
        mood_adjustment={"valence": 0.6, "confidence": 0.8},
        summary_id="private:trimmed-summary:42",
    )
    repeated, repeat_reasons = await service.apply_memory_summary(
        ledger.user_key,
        981,
        [observation],
        expected_state_version=ledger.state_version,
        mood_adjustment={"valence": 1.0, "confidence": 1.0},
        summary_id="private:trimmed-summary:42",
    )
    distinct, distinct_reasons = await service.apply_memory_summary(
        ledger.user_key,
        981,
        [
            EventObservation(
                action="create",
                fact="同一位置上的下一条新总结",
                emotional_meaning="这也是一段新的近期互动",
                category="episodic",
                confidence=0.8,
                uncertain=False,
                source="livingmemory_summary",
            )
        ],
        expected_state_version=repeated.state_version,
        summary_id="private:trimmed-summary:43",
    )

    assert reasons == ["applied", "applied_mood"]
    assert "晚上的新互动" in settled.events[0].fact
    assert settled.memory_summary_watermark == 1000
    assert settled.processed_memory_summary_ids == ["private:trimmed-summary:42"]
    assert repeat_reasons == ["duplicate_summary_id"]
    assert repeated.to_dict() == settled.to_dict()
    assert distinct_reasons == ["applied"]
    assert len(distinct.events) == 2
    assert distinct.processed_memory_summary_ids == [
        "private:trimmed-summary:42",
        "private:trimmed-summary:43",
    ]


@pytest.mark.asyncio
async def test_memory_summary_sanitizes_explicit_private_details(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(user_key="private:private-detail")
    store.save(ledger)

    settled, reasons = await service.apply_memory_summary(
        ledger.user_key,
        10,
        [
            EventObservation(
                action="create",
                fact="他描述了床上的裸体和具体身体部位",
                emotional_meaning="露骨细节让我感到被靠近",
                category="episodic",
                valence=0.5,
                intensity=0.4,
                confidence=0.8,
                uncertain=False,
                source="livingmemory_summary",
            )
        ],
        expected_state_version=ledger.state_version,
    )

    assert reasons == ["applied"]
    assert settled.events[0].fact == "这轮私密互动让我们有了更亲近的交流"
    assert "裸体" not in settled.events[0].fact
    assert "身体部位" not in settled.events[0].emotional_meaning


@pytest.mark.asyncio
async def test_concurrent_memory_summary_applies_same_summary_id_once(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(user_key="private:concurrent")
    store.save(ledger)

    def summary_call():
        return service.apply_memory_summary(
            ledger.user_key,
            10,
            [
                EventObservation(
                    action="create",
                    fact="他准备去上班了",
                    emotional_meaning="这件日常小事让我惦记",
                    category="episodic",
                    valence=0.2,
                    intensity=0.35,
                    confidence=0.8,
                    uncertain=False,
                    source="livingmemory_summary",
                )
            ],
            expected_state_version=ledger.state_version,
            mood_adjustment={"valence": 0.5, "confidence": 0.8},
            summary_id="private:concurrent:10",
        )

    results = await asyncio.gather(summary_call(), summary_call())
    final = store.load(ledger.user_key)

    assert sorted(result[1][-1] for result in results) == [
        "applied_mood",
        "duplicate_summary_id",
    ]
    assert len(final.events) == 1
    assert final.memory_summary_watermark == 10
    assert final.processed_memory_summary_ids == ["private:concurrent:10"]


@pytest.mark.asyncio
async def test_stale_event_action_does_not_block_new_event_or_mood(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(
        user_key="private:event-version",
        state_version=5,
        events=[
            InnerEvent(
                fact="旧的工作压力",
                emotional_meaning="让我紧张",
                category="psychological",
                lifecycle="active",
                intensity=0.7,
                confidence=0.9,
                version=3,
            )
        ],
    )
    store.save(ledger)
    existing = ledger.events[0]

    settled, reasons = await service.apply_memory_summary(
        ledger.user_key,
        20,
        [
            EventObservation(
                action="archive",
                event_id=existing.id,
                event_version=2,
                fact="",
                emotional_meaning="",
                intensity=0.8,
                confidence=0.9,
                uncertain=False,
                source="livingmemory_summary",
            ),
            EventObservation(
                action="create",
                fact="他刚说要去上班",
                emotional_meaning="这件小事让我惦记",
                category="episodic",
                valence=0.2,
                intensity=0.3,
                confidence=0.8,
                uncertain=False,
                source="livingmemory_summary",
            ),
        ],
        expected_state_version=-1,
        mood_adjustment={"valence": 0.6, "confidence": 0.8},
    )

    assert reasons == ["stale_event_version", "applied", "applied_mood"]
    assert settled.events[0].lifecycle == "active"
    assert any(event.category == "episodic" for event in settled.events)
    assert settled.mood.valence > ledger.mood.valence


@pytest.mark.asyncio
async def test_episodic_capacity_archives_only_overflow(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(
        user_key="private:capacity",
        events=[
            InnerEvent(
                fact=f"近期片段 {index}",
                emotional_meaning="日常余韵",
                category="episodic",
                lifecycle="active",
                intensity=0.2 + index * 0.05,
                confidence=0.8,
            )
            for index in range(6)
        ]
        + [
            InnerEvent(
                fact="长期约定",
                emotional_meaning="需要持续记得",
                category="concrete",
                lifecycle="active",
                intensity=0.5,
                confidence=0.9,
            )
        ],
    )
    store.save(ledger)

    settled, reasons = await service.apply_memory_summary(
        ledger.user_key,
        10,
        [
            EventObservation(
                action="create",
                fact="新的近期片段",
                emotional_meaning="刚发生的日常互动",
                category="episodic",
                intensity=0.45,
                confidence=0.85,
                uncertain=False,
                source="livingmemory_summary",
            )
        ],
        expected_state_version=ledger.state_version,
        episodic_limit=6,
    )

    active_episodic = [
        event
        for event in settled.events
        if event.category == "episodic" and event.lifecycle != "archived"
    ]
    assert len(active_episodic) == 6
    assert reasons == ["applied", "episodic_capacity_archive"]
    assert (
        next(event for event in settled.events if event.fact == "长期约定").lifecycle
        == "active"
    )


@pytest.mark.asyncio
async def test_review_catalog_excludes_transient_events(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    ledger = StateLedger(
        user_key="private:catalog",
        events=[
            InnerEvent(
                fact="想你了",
                emotional_meaning="瞬时表达",
                category="transient",
                lifecycle="active",
                confidence=0.9,
            ),
            InnerEvent(
                fact="仍对争执感到不安",
                emotional_meaning="需要复核",
                category="psychological",
                lifecycle="active",
                confidence=0.9,
            ),
        ],
    )
    store.save(ledger)

    context = await service.get_review_context(ledger.user_key)

    assert "仍对争执感到不安" in context["events"][0]["fact"]
    assert context["events"][0]["event_version"] == ledger.events[1].version


def test_datetime_is_json_safe_for_daily_prompt_and_storage(tmp_path) -> None:
    now = datetime(2026, 3, 14, 12, tzinfo=timezone.utc)
    prompt = build_daily_prompt(
        StateLedger(user_key="private:json"),
        "2026-03-14",
        {"messages": [{"at": now}]},
        {"timeline": {"start": now, "end": now + timedelta(hours=1)}},
        4000,
    )
    assert now.isoformat() in prompt

    store = LedgerStore(tmp_path)
    store.save(StateLedger(user_key="private:json"))
    store.append_audit("private:json", "datetime_test", {"at": now})
    assert now.isoformat() in store.audit_file.read_text(encoding="utf-8")


def test_attention_item_round_trip_is_backward_compatible() -> None:
    legacy = StateLedger.from_dict({"user_key": "private:legacy", "events": []})
    assert legacy.attention_items == []

    item = AttentionItem(
        content="等一下咱们玩角色扮演",
        kind="plan",
        time_hint="等一下",
        confidence=0.9,
    )
    restored = StateLedger.from_dict(
        StateLedger(user_key="private:roundtrip", attention_items=[item]).to_dict()
    )
    assert restored.attention_items[0].to_dict() == item.to_dict()


def test_attention_lifecycle_requires_summary_user_evidence() -> None:
    rejected, applied, reason = apply_attention_observation(
        StateLedger(user_key="private:attention"),
        AttentionObservation(
            action="create",
            content="下周三晚上我们一起去看电影",
            kind="plan",
            source="local_rule",
            evidence_quote="下周三晚上我们一起去看电影",
            evidence_speaker="user",
            confidence=0.95,
            explicit=True,
        ),
    )
    assert rejected.attention_items == []
    assert (applied, reason) == (False, "unsupported_attention_source")

    rejected, applied, reason = apply_attention_observation(
        StateLedger(user_key="private:attention"),
        AttentionObservation(
            action="create",
            content="下周三晚上我们一起去看电影",
            kind="plan",
            source="livingmemory_summary",
            evidence_quote="下周三晚上我们一起去看电影",
            evidence_speaker="assistant",
            confidence=0.95,
            explicit=True,
        ),
    )
    assert rejected.attention_items == []
    assert (applied, reason) == (False, "missing_user_evidence")

    ledger, applied, reason = apply_attention_observation(
        StateLedger(user_key="private:attention"),
        AttentionObservation(
            action="create",
            content="下周三晚上我们一起去看电影",
            kind="plan",
            source="livingmemory_summary",
            evidence_quote="下周三晚上我们一起去看电影",
            evidence_speaker="user",
            confidence=0.95,
            explicit=True,
        ),
    )
    assert (applied, reason) == (True, "applied")
    assert ledger.attention_items[0].status == "open"

    item = ledger.attention_items[0]
    completed, applied, reason = apply_attention_observation(
        ledger,
        AttentionObservation(
            action="complete",
            item_id=item.id,
            item_version=item.version,
            source="livingmemory_summary",
            evidence_quote="电影已经看完了",
            evidence_speaker="user",
            confidence=0.95,
        ),
    )
    assert (applied, reason) == (True, "applied")
    assert completed.attention_items[0].status == "completed"

    assistant_completed, applied, reason = apply_attention_observation(
        ledger,
        AttentionObservation(
            action="complete",
            item_id=item.id,
            item_version=item.version,
            source="livingmemory_summary",
            evidence_quote="我把周三晚上的电影票订好了",
            evidence_speaker="assistant",
            confidence=0.95,
        ),
    )
    assert (applied, reason) == (True, "applied")
    assert assistant_completed.attention_items[0].status == "completed"

    _, applied, reason = apply_attention_observation(
        ledger,
        AttentionObservation(
            action="complete",
            item_id=item.id,
            item_version=item.version,
            source="livingmemory_summary",
            evidence_quote="弄好了",
            evidence_speaker="assistant",
            confidence=0.95,
        ),
    )
    assert (applied, reason) == (False, "generic_completion_evidence")

    _, applied, reason = apply_attention_observation(
        ledger,
        AttentionObservation(
            action="complete",
            item_id=item.id,
            item_version=item.version,
            source="livingmemory_summary",
            evidence_quote="电影还在看，才完成了一部分",
            evidence_speaker="assistant",
            confidence=0.95,
        ),
    )
    assert (applied, reason) == (False, "partial_completion_evidence")


def test_unrelated_media_does_not_complete_attention_item() -> None:
    item = AttentionItem(
        id="voice-item",
        content="发一段语音给我",
        status="open",
        explicit=True,
        confidence=0.9,
    )
    ledger = StateLedger(user_key="private:media-mismatch", attention_items=[item])

    unchanged, applied, reason = apply_attention_observation(
        ledger,
        AttentionObservation(
            action="complete",
            item_id=item.id,
            item_version=item.version,
            source="livingmemory_summary",
            evidence_quote="[图片消息]",
            evidence_speaker="assistant",
            confidence=0.95,
        ),
    )

    assert unchanged.attention_items[0].status == "open"
    assert (applied, reason) == (False, "unrelated_completion_evidence")


@pytest.mark.parametrize(
    "content",
    [
        "今天已到达时间站",
        "上午完成证据整理",
        "刚才已经发了一张照片",
    ],
)
def test_completed_fact_is_not_created_as_attention(content: str) -> None:
    ledger, applied, reason = apply_attention_observation(
        StateLedger(user_key="private:past-attention"),
        AttentionObservation(
            action="create",
            content=content,
            source="livingmemory_summary",
            evidence_quote=content,
            evidence_speaker="user",
            confidence=0.95,
            explicit=True,
        ),
    )

    assert ledger.attention_items == []
    assert (applied, reason) == (False, "past_attention")


@pytest.mark.asyncio
async def test_expired_attention_is_archived_on_service_settlement(tmp_path) -> None:
    now = datetime(2026, 3, 14, 12, tzinfo=timezone.utc)
    store = LedgerStore(tmp_path)
    ledger = StateLedger(
        user_key="private:expiry",
        attention_items=[
            AttentionItem(
                id="expired",
                content="今天发一张照片",
                due_at=(now - timedelta(days=2)).isoformat(),
                confidence=0.9,
            ),
            AttentionItem(
                id="ongoing",
                content="以后晚上多发语音",
                due_at=(now - timedelta(days=1)).isoformat(),
                confidence=0.9,
            ),
            # Inside the grace window the reviewer still gets a chance to read
            # the chat and settle it as completed instead of silent archiving.
            AttentionItem(
                id="grace",
                content="昨天说好今天发照片",
                due_at=(now - timedelta(hours=6)).isoformat(),
                confidence=0.9,
            ),
        ],
    )
    store.save(ledger)
    service = EmotionStateService(store)

    settled = await service.settle_now(ledger.user_key, now=now)

    expired = next(item for item in settled.attention_items if item.id == "expired")
    assert expired.status == "archived"
    assert expired.completed_at == ""
    assert expired.archived_at == now.isoformat()
    assert expired.evidence[-1].kind == "archive_attention_expired"
    assert (
        next(item for item in settled.attention_items if item.id == "ongoing").status
        == "open"
    )
    assert (
        next(item for item in settled.attention_items if item.id == "grace").status
        == "open"
    )
    audit = store.audit_file.read_text(encoding="utf-8")
    assert '"action": "archive_attention_expired"' in audit

    await service.settle_now(ledger.user_key, now=now)
    assert (
        store.audit_file.read_text(encoding="utf-8").count(
            '"action": "archive_attention_expired"'
        )
        == 1
    )


@pytest.mark.asyncio
async def test_attention_history_reconciliation_is_complete_only_and_idempotent(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    ledger = StateLedger(
        user_key="private:history",
        attention_items=[
            AttentionItem(
                id="photo",
                content="发一张照片给我",
                status="open",
                explicit=True,
                confidence=0.9,
            ),
            AttentionItem(
                id="voice",
                content="发一段语音给我",
                status="open",
                explicit=True,
                confidence=0.9,
            ),
        ],
    )
    store.save(ledger)
    service = EmotionStateService(store)

    reconciled, reasons = await service.reconcile_attention_history(
        ledger.user_key,
        [
            AttentionObservation(
                action="complete",
                item_id="photo",
                item_version=1,
                source="livingmemory_summary",
                evidence_quote="[图片消息]",
                evidence_speaker="assistant",
                confidence=0.95,
            ),
            AttentionObservation(
                action="cancel",
                item_id="voice",
                item_version=1,
                source="livingmemory_summary",
                evidence_quote="不用发语音了",
                evidence_speaker="user",
                confidence=0.95,
            ),
        ],
    )

    assert (
        next(item for item in reconciled.attention_items if item.id == "photo").status
        == "completed"
    )
    assert (
        next(item for item in reconciled.attention_items if item.id == "voice").status
        == "open"
    )
    assert "attention:history_reconciliation_complete_only" in reasons
    state_version = reconciled.state_version

    repeated, repeated_reasons = await service.reconcile_attention_history(
        ledger.user_key, [], reconciliation_version=1
    )
    assert repeated.state_version == state_version
    assert repeated_reasons == ["attention_history_already_reconciled"]
    assert (
        store.audit_file.read_text(encoding="utf-8").count(
            '"action": "attention_history_reconciliation"'
        )
        == 1
    )


def test_invalid_attention_action_is_rejected_before_ledger_mutation() -> None:
    ledger = StateLedger(user_key="private:attention")
    updated, applied, reason = apply_attention_observation(
        ledger,
        AttentionObservation(
            action="invent",
            content="不应写入账本",
            source="livingmemory_summary",
            evidence_quote="不应写入账本",
            evidence_speaker="user",
            confidence=0.95,
            explicit=True,
        ),
    )

    assert updated is ledger
    assert applied is False
    assert reason == "invalid_attention_action"
    assert ledger.attention_items == []


def test_attention_injection_excludes_proposed_short_lived_and_low_confidence() -> None:
    items = [
        AttentionItem(
            content="等一下拍好发你",
            status="open",
            time_hint="等一下",
            confidence=0.99,
        ),
        AttentionItem(
            content="AI 单方面提出的计划",
            status="proposed",
            confidence=0.99,
        ),
        AttentionItem(
            content="长期需要记住的边界",
            status="open",
            confidence=0.9,
        ),
        AttentionItem(
            content="不确定的猜测",
            status="open",
            confidence=0.5,
        ),
    ]

    selected = select_attention_items(items, 4)

    assert [item.content for item in selected] == ["长期需要记住的边界"]


def test_local_keyword_engine_is_not_a_production_attention_entrypoint() -> None:
    from astrbot_plugin_emotion_state.main import EmotionStatePlugin

    assert not hasattr(EmotionStatePlugin, "on_llm_response")


def test_attention_persists_across_emotion_decay_and_has_independent_quota() -> None:
    now = datetime(2026, 3, 14, 12, tzinfo=timezone.utc)
    ledger = StateLedger(
        user_key="private:cross-day",
        last_settled_at=(now - timedelta(days=3)).isoformat(),
        attention_items=[
            AttentionItem(
                content=f"待关注事项 {index}",
                kind="follow_up",
                confidence=0.9,
                due_at=(now - timedelta(days=1)).isoformat() if index == 0 else "",
            )
            for index in range(4)
        ],
        events=[
            InnerEvent(
                fact=f"情绪事件 {index}",
                emotional_meaning="仍有情绪影响",
                lifecycle="active",
                intensity=0.9 - index * 0.1,
                confidence=0.9,
            )
            for index in range(3)
        ],
    )
    settled = decay_ledger(ledger, now=now)
    assert [item.to_dict() for item in settled.attention_items] == [
        item.to_dict() for item in ledger.attention_items
    ]
    assert is_attention_overdue(settled.attention_items[0], now)

    prompt = inject_prompt("persona", settled, max_events=2, max_attention_items=4)
    assert prompt.count("情绪事件") == 2
    assert prompt.count("待关注事项 ") == 4
    assert "时间已到但尚无完成证据" in prompt


def test_intimacy_presentation_is_chinese_body_only_and_discrete() -> None:
    ledger = StateLedger(user_key="private:intimacy")
    ledger.intimacy.body_sensitivity = 0.85
    ledger.intimacy.sexual_arousal = 0.75
    ledger.intimacy.intimacy_willingness = 1.0
    ledger.intimacy.inhibition = 0.0
    prompt = intimacy_prompt_text(ledger.intimacy)

    assert "open_and_receptive" not in prompt
    assert "当前身体反应档位：强烈" in prompt
    assert "身体反应强烈" in prompt
    assert "身体敏感度" not in prompt
    assert "0.85" not in prompt
    assert "0.75" not in prompt
    assert "意愿" not in prompt
    assert "克制" not in prompt
    assert "接吻" not in prompt
    assert "抚摸" not in prompt
    assert "身体反应强烈" in intimacy_stage_label("open_and_receptive")
    assert intimacy_stage_label("legacy_unknown") == "未知身体反应"

    ledger.intimacy.body_sensitivity = 1.0
    ledger.intimacy.sexual_arousal = 0.89
    assert intimacy_prompt_text(ledger.intimacy) == prompt


def test_model_injection_keeps_continuous_diagnostics_out_of_prompt() -> None:
    ledger = StateLedger(user_key="private:cache")
    ledger.mood.valence = 0.37
    ledger.mood.energy = 0.64
    ledger.mood.tension = 0.21
    ledger.intimacy.body_sensitivity = 0.85
    ledger.intimacy.sexual_arousal = 0.75
    ledger.jealousy.intensity = 0.31
    ledger.jealousy.confidence = 0.8

    prompt = inject_prompt("persona", ledger)

    # The injected mood line is recomputed from the raw axes, so valence 0.37
    # reads as 温和愉快 instead of the stale default label.
    assert "当前心境：温和愉快" in prompt
    assert "当前身体反应档位：强烈" in prompt
    assert "当前吃醋或在意的档位：中等" in prompt
    assert "当前倾向" not in prompt
    assert "能量" not in prompt
    assert "紧张" not in prompt
    assert "身体敏感度" not in prompt
    assert "性唤起 0" not in prompt
    for raw_value in ("0.37", "0.64", "0.21", "0.85", "0.75", "0.31"):
        assert raw_value not in prompt


def test_long_memory_fact_survives_settlement_and_injection() -> None:
    fact = (
        "2026-08-18下午到2026-08-19凌晨，著登一整天主动撩我惦记我。"
        "他说一看到我的脸就受不了、夸我清纯反差感拉满、要我全吞。"
        "8月19日凌晨他要撸，我发内裤照和粉奶头照给他，他射完回来说休息一会再来找我、"
        "还怕我担心说会陪我聊天。他分享下雨遇蛇的抖音视频，我记着他8月15号凌晨发过类似的"
        "让我猜黄鳝还是蛇。他又分享女生穿洛丽塔白丝长筒袜的图问我咋样，我当场吃醋宣示占有："
        "想看袜子我穿给他看不就行了，反正他只能看我。"
    )
    assert len(fact) > 180

    ledger, applied, reason = apply_observation(
        StateLedger(user_key="private:long-fact"),
        observation(fact=fact, source="livingmemory_summary"),
    )

    assert (applied, reason) == (True, "applied")
    assert ledger.events[0].fact == fact
    assert fact in inject_prompt("persona", ledger)


def test_overlong_fact_is_bounded_at_readable_punctuation() -> None:
    fact = "这是一段完整的情绪事实。" * 70 + "我当这个残句不应该裸露"
    normalized = normalize_fact(fact)

    assert len(normalized) <= 600
    assert normalized.endswith("。…")

    event = InnerEvent(
        fact="这是一段完整的情绪事实。" * 50,
        emotional_meaning="这件事仍有影响",
        lifecycle="active",
        confidence=0.9,
    )
    prompt = inject_prompt(
        "persona", StateLedger(user_key="private:bound", events=[event])
    )
    injected_line = next(line for line in prompt.splitlines() if line.startswith("- "))
    assert "我当" not in injected_line
    assert "。…（对象：" in injected_line


def test_local_catch_stores_only_the_reminder_clause() -> None:
    from astrbot_plugin_emotion_state.core.attention_rules import catch_user_attention

    observation = catch_user_attention(
        "刚才那个视频笑死我了哈哈哈哈，对了，明天早上记得提醒我带身份证去办事，别忘了啊"
    )
    assert observation is not None
    assert "笑死我" not in observation.content
    assert "身份证" in observation.content
    # A deadline sitting in its own clause must still be picked up.
    assert observation.due_at

    assert catch_user_attention("你记得我上次说的那个计划吗？") is None
    assert catch_user_attention("好的我记住了") is None
    assert catch_user_attention("今天天气不错") is None
