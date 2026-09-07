"""Serialized application service for per-private-session state."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any

from .attention import (
    apply_attention_observation,
    archive_expired_attention_items,
    archive_stale_attention_items,
    attention_observation_rejection,
    enforce_attention_capacity,
)
from .intrinsic import maybe_night_missing_observation
from .models import (
    AttentionObservation,
    EventObservation,
    EventTrace,
    IntrinsicParams,
    StateLedger,
    clamp,
    iso_now,
    utc_now,
)
from .settlement import (
    apply_observation,
    decay_ledger,
    derive_mood,
    enforce_episodic_capacity,
    enforce_psychological_capacity,
    is_generic_transient_fact,
    is_media_only_fact,
    jealousy_evidence,
    normalize_fact,
    sanitize_summary_event_text,
    settle_jealousy,
    settle_summary_mood,
)
from .storage import LedgerStore


class EmotionStateService:
    def __init__(
        self,
        store: LedgerStore,
        half_life_hours: float = 72.0,
        psychological_limit: int = 6,
        attention_limit: int = 8,
        episodic_limit: int = 6,
        intrinsic_factory: Callable[[], IntrinsicParams | None] | None = None,
        offset_half_life_minutes: float = 15.0,
        attention_auto_archive_days: float = 3.0,
        negative_bias: float = 2.5,
    ) -> None:
        self.store = store
        self.half_life_hours = half_life_hours
        self.psychological_limit = max(1, int(psychological_limit))
        self.attention_limit = max(1, int(attention_limit))
        self.episodic_limit = max(1, int(episodic_limit))
        # Optional factory producing the current intrinsic-dynamics snapshot;
        # None keeps the legacy purely reactive behaviour (used by old tests).
        self.intrinsic_factory = intrinsic_factory
        self.offset_half_life_minutes = max(1.0, float(offset_half_life_minutes))
        # 0 disables the stale-attention janitor entirely.
        self.attention_auto_archive_days = max(0.0, float(attention_auto_archive_days))
        # Negativity bias for user-directed hurtful events (clamped in settlement).
        self.negative_bias = float(negative_bias)
        self._locks: dict[str, asyncio.Lock] = {}

    def _intrinsic(self) -> IntrinsicParams | None:
        if self.intrinsic_factory is None:
            return None
        try:
            return self.intrinsic_factory()
        except Exception:
            return None

    def _enforce_capacities(
        self,
        ledger: StateLedger,
        *,
        episodic_limit: int | None = None,
        now: datetime | None = None,
    ) -> tuple[StateLedger, list[str], list[str]]:
        updated = ledger
        reasons: list[str] = []
        updated, expired_attention_ids = archive_expired_attention_items(updated, now)
        reasons.extend("attention_expired_archive" for _ in expired_attention_ids)
        if self.attention_auto_archive_days > 0:
            updated, stale_ids = archive_stale_attention_items(
                updated,
                max_days=self.attention_auto_archive_days,
                now=now,
            )
            expired_attention_ids.extend(stale_ids)
            reasons.extend("attention_stale_archive" for _ in stale_ids)
        if episodic_limit is None:
            episodic_limit = self.episodic_limit
        updated, episodic_ids = enforce_episodic_capacity(
            updated, episodic_limit, negative_bias=self.negative_bias
        )
        reasons.extend("episodic_capacity_archive" for _ in episodic_ids)
        updated, psychological_ids = enforce_psychological_capacity(
            updated, self.psychological_limit, negative_bias=self.negative_bias
        )
        reasons.extend("psychological_capacity_archive" for _ in psychological_ids)
        updated, attention_ids = enforce_attention_capacity(
            updated, self.attention_limit
        )
        reasons.extend("attention_capacity_archive" for _ in attention_ids)
        return updated, reasons, expired_attention_ids

    async def _append_expiry_audit(
        self,
        user_key: str,
        ledger: StateLedger,
        expired_attention_ids: list[str],
    ) -> None:
        if not expired_attention_ids:
            return
        await asyncio.to_thread(
            self.store.append_audit,
            user_key,
            "archive_attention_expired",
            {
                "item_ids": expired_attention_ids,
                "state_version": ledger.state_version,
            },
        )

    def _lock_for(self, user_key: str) -> asyncio.Lock:
        return self._locks.setdefault(user_key, asyncio.Lock())

    async def get(self, user_key: str, settle: bool = True) -> StateLedger:
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            updated = ledger
            intrinsic = self._intrinsic()
            if settle:
                updated = decay_ledger(
                    updated,
                    half_life_hours=self.half_life_hours,
                    negative_bias=self.negative_bias,
                    intrinsic=intrinsic,
                    offset_half_life_minutes=self.offset_half_life_minutes,
                )
                if intrinsic is not None:
                    observation = maybe_night_missing_observation(
                        updated, utc_now(), intrinsic
                    )
                    if observation is not None:
                        updated, applied, _ = apply_observation(
                            updated,
                            observation,
                            negative_bias=self.negative_bias,
                        )
                        if applied:
                            await asyncio.to_thread(
                                self.store.append_audit,
                                user_key,
                                "night_missing_event",
                                {
                                    "state_version": updated.state_version,
                                    "note": observation.note,
                                },
                            )
            updated, capacity_reasons, expired_attention_ids = self._enforce_capacities(
                updated
            )
            if updated.to_dict() != ledger.to_dict():
                await asyncio.to_thread(self.store.save, updated)
            if capacity_reasons:
                await asyncio.to_thread(
                    self.store.append_audit,
                    user_key,
                    "capacity_governance",
                    {
                        "reasons": capacity_reasons,
                        "state_version": updated.state_version,
                    },
                )
            await self._append_expiry_audit(user_key, updated, expired_attention_ids)
            return updated

    async def get_review_context(self, user_key: str) -> dict[str, Any]:
        """Return a bounded, read-only view for LivingMemory review."""
        ledger = await self.get(user_key)
        reviewable = [
            event
            for event in ledger.events
            if event.category != "transient"
            and event.lifecycle not in {"archived", "candidate"}
        ]
        reviewable.sort(
            key=lambda event: (
                event.category == "psychological",
                event.intensity,
                event.last_stimulated_at,
            ),
            reverse=True,
        )
        return {
            "state_version": ledger.state_version,
            "message_watermark": ledger.message_watermark,
            "events": [event.review_view() for event in reviewable[:12]],
            "attention_items": [
                item.review_view()
                for item in ledger.attention_items
                if item.status in {"proposed", "open"}
            ][-12:],
        }

    async def apply_memory_summary(
        self,
        user_key: str,
        end_index: int,
        observations: list[EventObservation],
        expected_state_version: int,
        expected_message_watermark: int | None = None,
        attention_observations: list[AttentionObservation] | None = None,
        mood_adjustment: dict[str, Any] | None = None,
        episodic_limit: int = 6,
        summary_id: str | None = None,
    ) -> tuple[StateLedger, list[str]]:
        """Apply one LivingMemory settlement atomically and idempotently."""
        normalized_summary_id = str(summary_id or "").strip()[:160]
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            if normalized_summary_id:
                if normalized_summary_id in ledger.processed_memory_summary_ids:
                    await asyncio.to_thread(
                        self.store.append_audit,
                        user_key,
                        "livingmemory_summary_duplicate",
                        {
                            "summary_id": normalized_summary_id,
                            "end_index": end_index,
                            "reason": "duplicate_summary_id",
                            "state_version": ledger.state_version,
                        },
                    )
                    return ledger, ["duplicate_summary_id"]
            elif end_index <= ledger.memory_summary_watermark:
                await asyncio.to_thread(
                    self.store.append_audit,
                    user_key,
                    "livingmemory_summary_duplicate",
                    {
                        "end_index": end_index,
                        "reason": "already_processed",
                        "state_version": ledger.state_version,
                    },
                )
                return ledger, ["already_processed"]

            snapshot_changed = ledger.state_version != expected_state_version or (
                expected_message_watermark is not None
                and ledger.message_watermark != expected_message_watermark
            )
            loaded_event_versions = {event.id: event.version for event in ledger.events}
            ledger = decay_ledger(
                ledger,
                half_life_hours=self.half_life_hours,
                negative_bias=self.negative_bias,
            )
            updated = StateLedger.from_dict(ledger.to_dict(), user_key=user_key)
            reasons: list[str] = []
            applied_count = 0
            attention_applied_count = 0
            create_count = 0
            review_count = 0
            for original_observation in observations[:6]:
                observation = replace(original_observation)
                if observation.action == "create":
                    create_count += 1
                    if create_count > 3:
                        reasons.append("create_limit")
                        continue
                else:
                    review_count += 1
                    if review_count > 3:
                        reasons.append("review_limit")
                        continue
                if "jealousy" in observation.tags:
                    has_evidence, source, strength = jealousy_evidence(observation.fact)
                    if (
                        observation.action != "create"
                        or not has_evidence
                        or observation.confidence < 0.62
                        or observation.uncertain
                    ):
                        reasons.append("invalid_jealousy_evidence")
                        continue
                    updated = settle_jealousy(
                        updated,
                        evidence=True,
                        source=f"summary:{source}",
                        strength=min(strength, observation.intensity),
                    )
                    reasons.append("applied_jealousy")
                    applied_count += 1
                    continue
                if observation.action == "create":
                    fact = normalize_fact(observation.fact)
                    if not fact:
                        reasons.append("empty_fact")
                        continue
                    if is_media_only_fact(fact):
                        reasons.append("media_only_fact")
                        continue
                    observation.fact, observation.emotional_meaning = (
                        sanitize_summary_event_text(
                            observation.fact, observation.emotional_meaning
                        )
                    )
                    if observation.category != "episodic" and is_generic_transient_fact(
                        fact
                    ):
                        reasons.append("transient_fact")
                        continue
                elif observation.confidence < 0.55 or observation.uncertain:
                    reasons.append("insufficient_review_confidence")
                    continue

                if observation.action != "create" and observation.event_version is None:
                    reasons.append("missing_event_version")
                    continue
                if observation.event_id:
                    matched = next(
                        (
                            event
                            for event in updated.events
                            if event.id == observation.event_id
                            and event.lifecycle not in {"archived", "candidate"}
                        ),
                        None,
                    )
                    if matched is None:
                        reasons.append("event_not_found")
                        continue
                    if (
                        observation.event_version is not None
                        and observation.event_version
                        != loaded_event_versions.get(matched.id)
                    ):
                        reasons.append("stale_event_version")
                        continue
                    observation.event_version = matched.version
                    if observation.action == "merge":
                        observation.fact, observation.emotional_meaning = (
                            sanitize_summary_event_text(
                                observation.fact, observation.emotional_meaning
                            )
                        )

                observation.expected_state_version = None
                observation.message_watermark = max(
                    observation.message_watermark, updated.message_watermark
                )
                updated, applied, reason = apply_observation(
                    updated, observation, negative_bias=self.negative_bias
                )
                reasons.append(reason)
                applied_count += int(applied)

            for original_attention in (attention_observations or [])[:6]:
                attention = replace(original_attention, expected_state_version=None)
                rejection = attention_observation_rejection(attention)
                if rejection:
                    reasons.append(f"attention:{rejection}")
                    continue
                updated, applied, reason = apply_attention_observation(
                    updated, attention
                )
                reasons.append(f"attention:{reason}")
                attention_applied_count += int(applied)

            updated, capacity_reasons, expired_attention_ids = self._enforce_capacities(
                updated, episodic_limit=episodic_limit
            )
            reasons.extend(capacity_reasons)
            if mood_adjustment:
                before_mood = (
                    updated.mood.to_dict()
                    if hasattr(updated.mood, "to_dict")
                    else (
                        updated.mood.valence,
                        updated.mood.energy,
                        updated.mood.tension,
                        updated.mood.updated_at,
                    )
                )
                updated = settle_summary_mood(
                    updated, mood_adjustment, negative_bias=self.negative_bias
                )
                after_mood = (
                    updated.mood.valence,
                    updated.mood.energy,
                    updated.mood.tension,
                    updated.mood.updated_at,
                )
                if before_mood != after_mood:
                    reasons.append("applied_mood")

            updated.memory_summary_watermark = max(
                updated.memory_summary_watermark, end_index
            )
            if normalized_summary_id:
                updated.processed_memory_summary_ids = [
                    item
                    for item in updated.processed_memory_summary_ids
                    if item != normalized_summary_id
                ][-31:] + [normalized_summary_id]
            updated.state_version = max(updated.state_version, ledger.state_version + 1)
            updated.updated_at = datetime.now().astimezone().isoformat()
            await asyncio.to_thread(self.store.save, updated)
            await asyncio.to_thread(
                self.store.append_audit,
                user_key,
                "livingmemory_summary",
                {
                    "summary_id": normalized_summary_id or None,
                    "end_index": end_index,
                    "observations": len(observations[:6]),
                    "applied": applied_count,
                    "attention_observations": len((attention_observations or [])[:6]),
                    "attention_applied": attention_applied_count,
                    "mood_applied": "applied_mood" in reasons,
                    "snapshot_changed": snapshot_changed,
                    "reasons": reasons,
                    "state_version": updated.state_version,
                },
            )
            await self._append_expiry_audit(user_key, updated, expired_attention_ids)
            return updated, reasons

    async def reconcile_attention_history(
        self,
        user_key: str,
        observations: list[AttentionObservation],
        *,
        reconciliation_version: int = 1,
    ) -> tuple[StateLedger, list[str]]:
        """Apply one bounded, idempotent review of pre-existing attention items."""
        version = max(1, int(reconciliation_version))
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            if ledger.attention_reconciliation_version >= version:
                return ledger, ["attention_history_already_reconciled"]

            updated = StateLedger.from_dict(ledger.to_dict(), user_key=user_key)
            reasons: list[str] = []
            applied_count = 0
            for original in observations[:12]:
                observation = replace(original, expected_state_version=None)
                if observation.action != "complete":
                    reasons.append("attention:history_reconciliation_complete_only")
                    continue
                rejection = attention_observation_rejection(observation)
                if rejection:
                    reasons.append(f"attention:{rejection}")
                    continue
                updated, applied, reason = apply_attention_observation(
                    updated, observation
                )
                reasons.append(f"attention:{reason}")
                applied_count += int(applied)

            (
                updated,
                capacity_reasons,
                expired_attention_ids,
            ) = self._enforce_capacities(updated)
            reasons.extend(capacity_reasons)
            updated.attention_reconciliation_version = version
            updated.state_version = max(updated.state_version, ledger.state_version + 1)
            updated.updated_at = datetime.now().astimezone().isoformat()
            await asyncio.to_thread(self.store.save, updated)
            await asyncio.to_thread(
                self.store.append_audit,
                user_key,
                "attention_history_reconciliation",
                {
                    "reconciliation_version": version,
                    "attention_observations": len(observations[:12]),
                    "attention_applied": applied_count,
                    "reasons": reasons,
                    "state_version": updated.state_version,
                },
            )
            await self._append_expiry_audit(user_key, updated, expired_attention_ids)
            return updated, reasons

    async def advance_watermark(self, user_key: str) -> StateLedger:
        """Persist one private-message position even when no rule matches."""
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            ledger.message_watermark += 1
            ledger.last_user_message_ts = utc_now().timestamp()
            ledger, capacity_reasons, expired_attention_ids = self._enforce_capacities(
                ledger
            )
            ledger.updated_at = datetime.now().astimezone().isoformat()
            await asyncio.to_thread(self.store.save, ledger)
            if capacity_reasons:
                await asyncio.to_thread(
                    self.store.append_audit,
                    user_key,
                    "capacity_governance",
                    {
                        "reasons": capacity_reasons,
                        "state_version": ledger.state_version,
                    },
                )
            await self._append_expiry_audit(user_key, ledger, expired_attention_ids)
            return ledger

    async def delete_item(
        self,
        user_key: str,
        kind: str,
        item_id: str,
    ) -> tuple[StateLedger, bool, str]:
        """Archive one explicitly selected item by its stable ID."""
        normalized_kind = str(kind or "").strip().lower()
        normalized_id = str(item_id or "").strip()[:96]
        if normalized_kind not in {"event", "attention"}:
            ledger = await self.get(user_key, settle=False)
            return ledger, False, "invalid_item_kind"
        if not normalized_id:
            ledger = await self.get(user_key, settle=False)
            return ledger, False, "invalid_item_reference"

        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            updated = StateLedger.from_dict(ledger.to_dict(), user_key=user_key)
            now = iso_now()
            if normalized_kind == "event":
                item = next(
                    (event for event in updated.events if event.id == normalized_id),
                    None,
                )
                if item is None:
                    return ledger, False, "item_not_found"
                if item.lifecycle == "archived":
                    return ledger, False, "item_already_archived"
                previous_item_version = item.version
                item.lifecycle = "archived"
                item.unresolved = False
                item.intensity = min(item.intensity, 0.1)
                item.updated_at = now
                item.version += 1
                item.traces.append(
                    EventTrace(
                        at=now,
                        kind="manual_archive",
                        amount=0.0,
                        note="WebUI 手动删除，保留历史审计",
                        source="webui",
                    )
                )
                item.traces = item.traces[-50:]
                updated.mood = derive_mood(updated.events, previous=updated.mood)
                archived_status = item.lifecycle
            else:
                item = next(
                    (
                        attention
                        for attention in updated.attention_items
                        if attention.id == normalized_id
                    ),
                    None,
                )
                if item is None:
                    return ledger, False, "item_not_found"
                if item.status == "archived":
                    return ledger, False, "item_already_archived"
                previous_item_version = item.version
                item.status = "archived"
                item.archived_at = now
                item.updated_at = now
                item.version += 1
                item.evidence.append(
                    EventTrace(
                        at=now,
                        kind="manual_archive",
                        amount=0.0,
                        note="WebUI 手动删除，保留历史审计",
                        source="webui",
                    )
                )
                item.evidence = item.evidence[-30:]
                archived_status = item.status

            updated.state_version = ledger.state_version + 1
            updated.updated_at = now
            await asyncio.to_thread(self.store.save, updated)
            await asyncio.to_thread(
                self.store.append_audit,
                user_key,
                "webui_manual_archive",
                {
                    "reason": "manual_archive",
                    "kind": normalized_kind,
                    "item_id": normalized_id,
                    "previous_item_version": previous_item_version,
                    "previous_state_version": ledger.state_version,
                    "state_version": updated.state_version,
                },
            )
            return updated, True, archived_status

    async def observe(
        self, user_key: str, observation: EventObservation
    ) -> tuple[StateLedger, bool, str]:
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            intrinsic = self._intrinsic()
            ledger = decay_ledger(
                ledger,
                half_life_hours=self.half_life_hours,
                intrinsic=intrinsic,
                offset_half_life_minutes=self.offset_half_life_minutes,
                negative_bias=self.negative_bias,
            )
            if intrinsic is not None:
                # Sensitivity multipliers scale how hard one piece of evidence hits.
                sensitivity = intrinsic.sensitivity
                scaled = clamp(
                    observation.intensity
                    * sensitivity.direction_factor(observation.valence)
                    * sensitivity.overall
                )
                if abs(scaled - observation.intensity) > 1e-6:
                    observation = replace(observation, intensity=clamp(scaled))
            updated, applied, reason = apply_observation(
                ledger, observation, negative_bias=self.negative_bias
            )
            capacity_reasons: list[str] = []
            if applied:
                (
                    updated,
                    capacity_reasons,
                    expired_attention_ids,
                ) = self._enforce_capacities(updated)
                await asyncio.to_thread(self.store.save, updated)
                await asyncio.to_thread(
                    self.store.append_audit,
                    user_key,
                    "event_observation",
                    {
                        "action": observation.action,
                        "source": observation.source,
                        "state_version": updated.state_version,
                        "message_watermark": observation.message_watermark,
                        "capacity_reasons": capacity_reasons,
                    },
                )
                await self._append_expiry_audit(
                    user_key, updated, expired_attention_ids
                )
            else:
                # Rejections used to be silent, which made misjudgments hard to audit.
                await asyncio.to_thread(
                    self.store.append_audit,
                    user_key,
                    "event_observation_rejected",
                    {
                        "action": observation.action,
                        "source": observation.source,
                        "reason": reason,
                        "state_version": ledger.state_version,
                    },
                )
            return updated, applied, reason

    async def observe_attention(
        self, user_key: str, observation: AttentionObservation
    ) -> tuple[StateLedger, bool, str]:
        rejection = attention_observation_rejection(observation)
        if rejection:
            ledger = await self.get(user_key, settle=False)
            return ledger, False, rejection
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            updated, applied, reason = apply_attention_observation(ledger, observation)
            capacity_reasons: list[str] = []
            if applied:
                (
                    updated,
                    capacity_reasons,
                    expired_attention_ids,
                ) = self._enforce_capacities(updated)
                await asyncio.to_thread(self.store.save, updated)
                await asyncio.to_thread(
                    self.store.append_audit,
                    user_key,
                    "attention_observation",
                    {
                        "action": observation.action,
                        "kind": observation.kind,
                        "source": observation.source,
                        "state_version": updated.state_version,
                        "capacity_reasons": capacity_reasons,
                    },
                )
                await self._append_expiry_audit(
                    user_key, updated, expired_attention_ids
                )
            return updated, applied, reason

    async def mutate(
        self,
        user_key: str,
        action: str,
        mutation: Callable[[StateLedger], StateLedger],
        audit_detail: dict[str, Any] | None = None,
    ) -> StateLedger:
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            updated = mutation(
                StateLedger.from_dict(ledger.to_dict(), user_key=user_key)
            )
            updated, capacity_reasons, expired_attention_ids = self._enforce_capacities(
                updated
            )
            updated.state_version = max(updated.state_version, ledger.state_version + 1)
            updated.updated_at = datetime.now().astimezone().isoformat()
            await asyncio.to_thread(self.store.save, updated)
            detail = dict(audit_detail or {})
            if capacity_reasons:
                detail["capacity_reasons"] = capacity_reasons
            await asyncio.to_thread(
                self.store.append_audit,
                user_key,
                action,
                detail or {"state_version": updated.state_version},
            )
            await self._append_expiry_audit(user_key, updated, expired_attention_ids)
            return updated

    async def mutate_if_changed(
        self,
        user_key: str,
        action: str,
        mutation: Callable[[StateLedger], StateLedger],
        audit_detail: dict[str, Any] | None = None,
    ) -> tuple[StateLedger, bool]:
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            updated = mutation(
                StateLedger.from_dict(ledger.to_dict(), user_key=user_key)
            )
            updated, capacity_reasons, expired_attention_ids = self._enforce_capacities(
                updated
            )
            if updated.to_dict() == ledger.to_dict():
                return ledger, False
            updated.state_version = max(updated.state_version, ledger.state_version + 1)
            updated.updated_at = datetime.now().astimezone().isoformat()
            await asyncio.to_thread(self.store.save, updated)
            detail = dict(audit_detail or {})
            if capacity_reasons:
                detail["capacity_reasons"] = capacity_reasons
            await asyncio.to_thread(
                self.store.append_audit,
                user_key,
                action,
                detail or {"state_version": updated.state_version},
            )
            await self._append_expiry_audit(user_key, updated, expired_attention_ids)
            return updated, True

    async def settle_now(
        self, user_key: str, now: datetime | None = None
    ) -> StateLedger:
        async with self._lock_for(user_key):
            ledger = await asyncio.to_thread(self.store.load, user_key)
            updated = decay_ledger(
                ledger,
                now=now,
                half_life_hours=self.half_life_hours,
                intrinsic=self._intrinsic(),
                offset_half_life_minutes=self.offset_half_life_minutes,
                negative_bias=self.negative_bias,
            )
            updated, capacity_reasons, expired_attention_ids = self._enforce_capacities(
                updated, now=now
            )
            await asyncio.to_thread(self.store.save, updated)
            if capacity_reasons:
                await asyncio.to_thread(
                    self.store.append_audit,
                    user_key,
                    "capacity_governance",
                    {
                        "reasons": capacity_reasons,
                        "state_version": updated.state_version,
                    },
                )
            await self._append_expiry_audit(user_key, updated, expired_attention_ids)
            return updated
