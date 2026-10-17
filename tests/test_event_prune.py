"""Retention tests for moving stale archived events out of the hot ledger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from astrbot_plugin_emotion_state.core.models import InnerEvent, StateLedger
from astrbot_plugin_emotion_state.core.service import EmotionStateService
from astrbot_plugin_emotion_state.core.storage import LedgerStore


def _archived_event(event_id: str, age_days: int) -> InnerEvent:
    stamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    return InnerEvent(
        fact=f"归档测试 {event_id}",
        emotional_meaning="测试用",
        category="episodic",
        lifecycle="archived",
        id=event_id,
        created_at=stamp,
        updated_at=stamp,
        last_stimulated_at=stamp,
    )


@pytest.mark.asyncio
async def test_prune_archived_events_moves_only_stale_rows(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    user_key = "private:prune"
    ledger = StateLedger(user_key=user_key)
    ledger.events = [
        _archived_event("old-1", 30),
        _archived_event("old-2", 20),
        _archived_event("fresh", 1),
        InnerEvent(
            fact="仍然活跃",
            emotional_meaning="测试用",
            category="episodic",
            lifecycle="active",
            id="active-1",
        ),
    ]
    store.save(ledger)

    assert await service.prune_archived_events(user_key, 14) == 2

    reloaded = store.load(user_key)
    assert {event.id for event in reloaded.events} == {"fresh", "active-1"}

    archive = tmp_path / "events_archive.jsonl"
    rows = [
        json.loads(line)
        for line in archive.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert {row["event"]["id"] for row in rows} == {"old-1", "old-2"}

    # Second pass is a no-op: retired rows are gone, nothing is archived twice.
    assert await service.prune_archived_events(user_key, 14) == 0
    assert len(archive.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.asyncio
async def test_prune_disabled_when_keep_days_is_zero(tmp_path) -> None:
    store = LedgerStore(tmp_path)
    service = EmotionStateService(store)
    user_key = "private:prune-off"
    ledger = StateLedger(user_key=user_key)
    ledger.events = [_archived_event("old-1", 30)]
    store.save(ledger)

    assert await service.prune_archived_events(user_key, 0) == 0
    assert len(store.load(user_key).events) == 1
    assert not (tmp_path / "events_archive.jsonl").exists()
