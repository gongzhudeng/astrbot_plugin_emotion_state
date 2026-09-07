from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from astrbot_plugin_emotion_state.core.models import (
    AttentionItem,
    InnerEvent,
    StateLedger,
)
from astrbot_plugin_emotion_state.core.rules import LocalRuleEngine
from astrbot_plugin_emotion_state.core.service import EmotionStateService
from astrbot_plugin_emotion_state.core.settlement import (
    event_attribution_rejection,
    select_injected_events_with_reasons,
)
from astrbot_plugin_emotion_state.core.storage import LedgerStore


@pytest.mark.asyncio
async def test_long_term_capacity_counts_concrete_and_all_visible_lifecycles(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(
        store,
        psychological_limit=6,
        attention_limit=8,
        episodic_limit=6,
    )
    events = [
        InnerEvent(
            id=f"psych-{index}",
            fact=f"长期心事 {index}",
            emotional_meaning="持续影响",
            category="concrete" if index >= 5 else "psychological",
            lifecycle="active",
            intensity=0.3 + index * 0.05,
            confidence=0.7 + index * 0.02,
            updated_at=f"2026-03-14T0{index}:00:00+00:00",
            last_stimulated_at=f"2026-03-14T0{index}:00:00+00:00",
        )
        for index in range(7)
    ]
    events.extend(
        [
            InnerEvent(
                id="psych-candidate",
                fact="候选心事",
                emotional_meaning="尚未确认",
                category="psychological",
                lifecycle="candidate",
            ),
            InnerEvent(
                id="psych-dormant",
                fact="沉寂心事",
                emotional_meaning="已经沉寂",
                category="psychological",
                lifecycle="dormant",
            ),
            InnerEvent(
                id="psych-archived",
                fact="历史心事",
                emotional_meaning="历史记录",
                category="psychological",
                lifecycle="archived",
            ),
            InnerEvent(
                id="episodic-1",
                fact="近期片段",
                emotional_meaning="日常影响",
                category="episodic",
                lifecycle="active",
            ),
        ]
    )
    ledger = StateLedger(user_key="private:psych-capacity", events=events)
    store.save(ledger)

    settled = await service.get(ledger.user_key, settle=False)
    active = [
        event
        for event in settled.events
        if event.category in {"psychological", "concrete"}
        and event.lifecycle != "archived"
    ]

    assert len(active) == 6
    assert len(settled.events) == len(events)
    archived = next(event for event in settled.events if event.id == "psych-0")
    assert archived.lifecycle == "archived"
    assert archived.unresolved is False
    assert archived.traces[-1].kind == "archive_psychological_capacity"
    assert "超过 6" in archived.traces[-1].note
    assert (
        next(
            event for event in settled.events if event.id == "psych-candidate"
        ).lifecycle
        == "archived"
    )
    assert (
        next(event for event in settled.events if event.id == "psych-dormant").lifecycle
        == "archived"
    )
    assert (
        next(event for event in settled.events if event.id == "psych-archived").traces
        == []
    )

    repeated = await service.get(ledger.user_key, settle=False)
    assert repeated.to_dict() == settled.to_dict()
    audit = store.audit_file.read_text(encoding="utf-8")
    assert audit.count("psychological_capacity_archive") == 3


@pytest.mark.asyncio
async def test_attention_capacity_keeps_terminal_history_and_audits_overflow(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store, psychological_limit=6, attention_limit=8)
    items = [
        AttentionItem(
            id=f"attention-{index}",
            content=f"待关注 {index}",
            status="open",
            confidence=0.4 + index * 0.05,
            explicit=index >= 3,
            # Recent evidence: the stale-attention janitor only clears items
            # that have had no new evidence for attention_auto_archive_days.
            last_evidence_at=(
                datetime.now(timezone.utc) - timedelta(hours=index + 2)
            ).isoformat(),
        )
        for index in range(9)
    ]
    items.extend(
        [
            AttentionItem(
                id="attention-completed",
                content="已完成历史",
                status="completed",
            ),
            AttentionItem(
                id="attention-cancelled",
                content="已取消历史",
                status="cancelled",
            ),
            AttentionItem(
                id="attention-archived",
                content="已有归档",
                status="archived",
            ),
        ]
    )
    ledger = StateLedger(user_key="private:attention-capacity", attention_items=items)
    store.save(ledger)

    settled = await service.get(ledger.user_key, settle=False)
    unfinished = [
        item for item in settled.attention_items if item.status in {"proposed", "open"}
    ]

    assert len(unfinished) == 8
    assert len(settled.attention_items) == len(items)
    archived = next(
        item for item in settled.attention_items if item.id == "attention-0"
    )
    assert archived.status == "archived"
    assert archived.completed_at == ""
    assert archived.archived_at
    assert archived.evidence[-1].kind == "archive_attention_capacity"
    assert (
        next(
            item for item in settled.attention_items if item.id == "attention-completed"
        ).status
        == "completed"
    )
    assert (
        next(
            item for item in settled.attention_items if item.id == "attention-cancelled"
        ).status
        == "cancelled"
    )
    assert (
        next(
            item for item in settled.attention_items if item.id == "attention-archived"
        ).status
        == "archived"
    )
    assert "attention_capacity_archive" in store.audit_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_manual_event_archive_preserves_history_and_recomputes_mood(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    event = InnerEvent(
        id="event-delete",
        fact="仍在影响我的负面事情",
        emotional_meaning="让我持续紧张",
        category="psychological",
        lifecycle="active",
        valence=-0.9,
        intensity=0.9,
        confidence=1.0,
        version=4,
    )
    ledger = StateLedger(
        user_key="private:manual-event",
        state_version=7,
        events=[event],
    )
    ledger.mood.valence = -0.8
    ledger.mood.tension = 0.8
    store.save(ledger)

    updated, changed, reason = await service.delete_item(
        ledger.user_key,
        "event",
        event.id,
    )

    assert changed is True
    assert reason == "archived"
    assert updated.state_version == 8
    assert len(updated.events) == 1
    archived = updated.events[0]
    assert archived.id == event.id
    assert archived.lifecycle == "archived"
    assert archived.unresolved is False
    assert archived.version == 5
    assert archived.traces[-1].kind == "manual_archive"
    assert archived.traces[-1].source == "webui"
    assert "保留历史审计" in archived.traces[-1].note
    assert updated.mood.valence > ledger.mood.valence
    assert updated.mood.tension < ledger.mood.tension

    persisted = store.load(ledger.user_key)
    assert persisted.events[0].to_dict() == archived.to_dict()
    audit = [
        json.loads(line)
        for line in store.audit_file.read_text(encoding="utf-8").splitlines()
    ][-1]
    assert audit["action"] == "webui_manual_archive"
    assert audit["detail"]["reason"] == "manual_archive"
    assert audit["detail"]["previous_state_version"] == 7
    assert audit["detail"]["previous_item_version"] == 4


@pytest.mark.asyncio
async def test_manual_attention_archive_preserves_record_and_terminal_meaning(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    item = AttentionItem(
        id="attention-delete",
        content="以后提醒我处理这件事",
        status="open",
        version=3,
    )
    ledger = StateLedger(
        user_key="private:manual-attention",
        state_version=5,
        attention_items=[item],
    )
    store.save(ledger)

    updated, changed, reason = await service.delete_item(
        ledger.user_key,
        "attention",
        item.id,
    )

    assert changed is True
    assert reason == "archived"
    assert len(updated.attention_items) == 1
    archived = updated.attention_items[0]
    assert archived.status == "archived"
    assert archived.completed_at == ""
    assert archived.archived_at
    assert archived.version == 4
    assert archived.evidence[-1].kind == "manual_archive"
    assert archived.evidence[-1].source == "webui"
    assert (
        store.load(ledger.user_key).attention_items[0].to_dict() == archived.to_dict()
    )


@pytest.mark.asyncio
async def test_manual_archive_allows_backend_state_changes_between_click_and_delete(
    tmp_path,
) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    event = InnerEvent(
        id="changing-event",
        fact="后台仍在衰减的事情",
        emotional_meaning="连续删除测试",
        lifecycle="active",
        version=2,
    )
    ledger = StateLedger(user_key="private:changing", state_version=9, events=[event])
    store.save(ledger)

    changed_ledger = StateLedger.from_dict(ledger.to_dict(), user_key=ledger.user_key)
    changed_ledger.state_version = 100
    changed_ledger.events[0].version = 8
    store.save(changed_ledger)

    updated, changed, reason = await service.delete_item(
        ledger.user_key,
        "event",
        event.id,
    )

    assert changed is True
    assert reason == "archived"
    assert updated.events[0].lifecycle == "archived"
    assert updated.events[0].version == 9


def test_injection_limit_still_returns_two_or_only_one_qualified_event() -> None:
    events = [
        InnerEvent(
            id=f"inject-{index}",
            fact=f"合格事件 {index}",
            emotional_meaning="持续影响",
            category="psychological",
            lifecycle="active",
            intensity=0.8 - index * 0.1,
            confidence=0.9,
        )
        for index in range(2)
    ]
    events.extend(
        [
            InnerEvent(
                id="inject-candidate",
                fact="候选事件",
                emotional_meaning="未确认",
                category="psychological",
                lifecycle="candidate",
                intensity=1.0,
                confidence=1.0,
            ),
            InnerEvent(
                id="inject-transient",
                fact="瞬时事件",
                emotional_meaning="短时情绪",
                category="transient",
                lifecycle="active",
                intensity=1.0,
                confidence=1.0,
            ),
        ]
    )
    selected, exclusions = select_injected_events_with_reasons(events, 2)
    assert [event.id for event in selected] == ["inject-0", "inject-1"]
    assert exclusions == {
        "inject-candidate": "lifecycle:candidate",
        "inject-transient": "transient_category",
    }

    single, single_exclusions = select_injected_events_with_reasons(events[1:], 2)
    assert [event.id for event in single] == ["inject-1"]
    assert single_exclusions == {
        "inject-candidate": "lifecycle:candidate",
        "inject-transient": "transient_category",
    }


def test_attack_attribution_requires_exact_direct_evidence() -> None:
    assert (
        event_attribution_rejection(
            "用户说恶心死了",
            "unknown",
            ["abuse"],
            "恶心死了",
            "user",
        )
        == "non_user_attack_target"
    )
    assert (
        event_attribution_rejection(
            "模型重写的攻击事实：你真恶心",
            "user",
            ["abuse"],
            "你",
            "user",
        )
        == "unverified_user_attack_target"
    )
    assert (
        event_attribution_rejection(
            "地铁站老头身上臭烘烘的，恶心死了",
            "user",
            ["abuse"],
            "地铁站老头身上臭烘烘的，恶心死了",
            "user",
        )
        == "unverified_user_attack_target"
    )
    assert (
        event_attribution_rejection(
            "你真恶心，滚开",
            "user",
            ["abuse"],
            "你真恶心，滚开",
            "user",
        )
        is None
    )


def test_local_attack_rules_keep_ambiguous_and_third_party_text_transient() -> None:
    engine = LocalRuleEngine()
    ambiguous = engine.run("恶心死了", watermark=1)
    third_party = engine.run("地铁站老头身上臭烘烘的，恶心死了", watermark=2)
    direct = engine.run("你真恶心，滚开", watermark=3)

    assert ambiguous.candidates == []
    assert third_party.candidates == []
    assert all(item.target == "unknown" for item in ambiguous.transient_signals)
    assert all(item.target == "third_party" for item in third_party.transient_signals)
    assert any(item.target == "user" for item in direct.candidates)
    assert all(
        item.target_basis == "explicit_user_subject" for item in direct.candidates
    )
