"""Versioned domain models for private emotional state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from .text_limits import (
    EVENT_FACT_REVIEW_CHARS,
    EVENT_FACT_STORAGE_CHARS,
    bound_complete_text,
)

Lifecycle = Literal[
    "candidate", "active", "intensified", "easing", "dormant", "archived"
]
EventCategory = Literal["transient", "episodic", "psychological", "concrete"]
ObservationAction = Literal[
    "create",
    "merge",
    "intensify",
    "ease",
    "recall",
    "dormant",
    "archive",
    "retain",
]
AttentionKind = Literal["commitment", "plan", "remember", "follow_up"]
AttentionStatus = Literal[
    "proposed", "open", "completed", "cancelled", "superseded", "archived"
]
AttentionAction = Literal[
    "create", "confirm", "update", "complete", "cancel", "supersede"
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def parse_time(value: str | None) -> datetime:
    if not value:
        return utc_now()
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class EventTrace:
    at: str
    kind: str
    amount: float
    note: str = ""
    source: str = "local"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventTrace:
        return cls(
            at=str(data.get("at") or iso_now()),
            kind=str(data.get("kind") or "evidence"),
            amount=float(data.get("amount", 0.0)),
            note=str(data.get("note") or ""),
            source=str(data.get("source") or "local"),
        )


@dataclass(slots=True)
class InnerEvent:
    fact: str
    emotional_meaning: str
    target: str = "unknown"
    target_basis: str = ""
    evidence_quote: str = ""
    evidence_speaker: str = ""
    category: EventCategory = "concrete"
    valence: float = 0.0
    intensity: float = 0.35
    confidence: float = 0.5
    source: str = "local_rule"
    lifecycle: Lifecycle = "candidate"
    id: str = field(default_factory=lambda: uuid4().hex)
    fingerprint: str = ""
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)
    last_stimulated_at: str = field(default_factory=iso_now)
    occurrence_count: int = 1
    unresolved: bool = True
    version: int = 1
    tags: list[str] = field(default_factory=list)
    traces: list[EventTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fact = bound_complete_text(self.fact, EVENT_FACT_STORAGE_CHARS)
        self.emotional_meaning = self.emotional_meaning.strip()
        self.target = str(self.target or "unknown").strip()[:80] or "unknown"
        self.target_basis = str(self.target_basis or "").strip()[:120]
        self.evidence_quote = str(self.evidence_quote or "").strip()[:240]
        self.evidence_speaker = str(self.evidence_speaker or "").strip().lower()[:24]
        if self.category not in {
            "transient",
            "episodic",
            "psychological",
            "concrete",
        }:
            self.category = self._legacy_category(self.tags, self.fact)
        self.valence = clamp(self.valence, -1.0, 1.0)
        self.intensity = clamp(self.intensity)
        self.confidence = clamp(self.confidence)
        self.occurrence_count = max(1, int(self.occurrence_count))

    @staticmethod
    def _legacy_category(tags: list[str], fact: str = "") -> EventCategory:
        tag_set = set(tags)
        if "transient" in tag_set:
            return "transient"
        if "psychological" in tag_set:
            return "psychological"
        if {"flirt", "positive"} & tag_set:
            concrete_cues = (
                "因为",
                "由于",
                "记得",
                "生日",
                "礼物",
                "答应",
                "约定",
                "帮我",
                "陪我",
                "会议",
                "工作",
                "考试",
                "项目",
                "见面",
                "聊天",
                "视频",
                "图片",
                "同事",
                "朋友",
                "家人",
                "回复",
                "消息",
            )
            return (
                "concrete" if any(cue in fact for cue in concrete_cues) else "transient"
            )
        return "concrete"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def review_view(self) -> dict[str, Any]:
        """Expose only bounded fields needed for private-summary review."""
        return {
            "event_id": self.id,
            "event_version": self.version,
            "category": self.category,
            "fact": bound_complete_text(self.fact, EVENT_FACT_REVIEW_CHARS),
            "emotional_meaning": self.emotional_meaning[:160],
            "target": self.target,
            "target_basis": self.target_basis,
            "evidence_quote": self.evidence_quote[:160],
            "evidence_speaker": self.evidence_speaker,
            "intensity": round(self.intensity, 3),
            "confidence": round(self.confidence, 3),
            "lifecycle": self.lifecycle,
            "last_stimulated_at": self.last_stimulated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InnerEvent:
        known = {
            key: data[key]
            for key in cls.__dataclass_fields__
            if key in data and key != "traces"
        }
        known["traces"] = [
            EventTrace.from_dict(item)
            for item in data.get("traces", [])
            if isinstance(item, dict)
        ]
        if "category" not in known:
            known["category"] = cls._legacy_category(
                list(data.get("tags") or []), str(data.get("fact") or "")
            )
        return cls(**known)


@dataclass(slots=True)
class MoodState:
    valence: float = 0.0
    energy: float = 0.45
    tension: float = 0.2
    label: str = "平静"
    updated_at: str = field(default_factory=iso_now)
    confidence: float = 0.5

    def __post_init__(self) -> None:
        self.valence = clamp(self.valence, -1.0, 1.0)
        self.energy = clamp(self.energy)
        self.tension = clamp(self.tension)
        self.confidence = clamp(self.confidence)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoodState:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class IntimacyState:
    body_sensitivity: float = 0.0
    sexual_arousal: float = 0.0
    intimacy_willingness: float = 0.15
    inhibition: float = 0.8
    stage: str = "not_noticeable"
    updated_at: str = field(default_factory=iso_now)
    last_relevant_at: str = ""

    def __post_init__(self) -> None:
        self.body_sensitivity = clamp(self.body_sensitivity)
        self.sexual_arousal = clamp(self.sexual_arousal)
        self.intimacy_willingness = clamp(self.intimacy_willingness)
        self.inhibition = clamp(self.inhibition)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntimacyState:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class JealousyState:
    intensity: float = 0.0
    confidence: float = 0.0
    last_evidence_at: str = ""
    sources: list[str] = field(default_factory=list)
    evidence_count: int = 0
    updated_at: str = field(default_factory=iso_now)

    def __post_init__(self) -> None:
        self.intensity = clamp(self.intensity, 0.0, 0.6)
        self.confidence = clamp(self.confidence)
        self.evidence_count = max(0, int(self.evidence_count))
        self.sources = list(dict.fromkeys(str(item) for item in self.sources))[-5:]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JealousyState:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class AttentionItem:
    content: str
    kind: AttentionKind = "follow_up"
    status: AttentionStatus = "open"
    actor: str = "both"
    time_hint: str = ""
    due_at: str = ""
    confidence: float = 0.7
    explicit: bool = True
    source: str = "local_rule"
    evidence_speaker: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    fingerprint: str = ""
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)
    last_evidence_at: str = field(default_factory=iso_now)
    completed_at: str = ""
    archived_at: str = ""
    version: int = 1
    evidence: list[EventTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.content = self.content.strip()[:240]
        if self.kind not in {"commitment", "plan", "remember", "follow_up"}:
            self.kind = "follow_up"
        if self.status not in {
            "proposed",
            "open",
            "completed",
            "cancelled",
            "superseded",
            "archived",
        }:
            self.status = "open"
        self.actor = str(self.actor or "both").strip()[:40]
        self.time_hint = str(self.time_hint or "").strip()[:80]
        self.due_at = str(self.due_at or "").strip()[:64]
        self.evidence_speaker = str(self.evidence_speaker or "").strip().lower()[:24]
        self.confidence = clamp(self.confidence)
        self.version = max(1, int(self.version))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def review_view(self) -> dict[str, Any]:
        return {
            "item_id": self.id,
            "item_version": self.version,
            "kind": self.kind,
            "status": self.status,
            "content": self.content,
            "actor": self.actor,
            "time_hint": self.time_hint,
            "due_at": self.due_at,
            "confidence": round(self.confidence, 3),
            "last_evidence_at": self.last_evidence_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttentionItem:
        known = {
            key: data[key]
            for key in cls.__dataclass_fields__
            if key in data and key != "evidence"
        }
        known["evidence"] = [
            EventTrace.from_dict(item)
            for item in data.get("evidence", [])
            if isinstance(item, dict)
        ]
        return cls(**known)


@dataclass(slots=True)
class DiaryEntry:
    cycle_date: str
    diary: str
    day_summary: str
    mood_proposal: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    provider_id: str = "local_fallback"
    created_at: str = field(default_factory=iso_now)
    input_watermark: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiaryEntry:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class ProactiveEvidenceProgress:
    evidence_id: str
    sent_at: float = 0.0
    applied_stage: int = 0
    replied: bool = False
    event_id: str = ""
    updated_at: str = field(default_factory=iso_now)

    def __post_init__(self) -> None:
        self.evidence_id = str(self.evidence_id or "").strip()[:96]
        self.sent_at = max(0.0, float(self.sent_at or 0.0))
        self.applied_stage = max(0, int(self.applied_stage or 0))
        self.event_id = str(self.event_id or "").strip()[:96]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProactiveEvidenceProgress:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class MoodOffsetState:
    """Fast-decaying short-term mood offset layered on top of the settled baseline.

    Transient reactions (being teased into a laugh, a brief annoyance) land here
    instead of polluting the slow baseline mood; it decays to zero within minutes.
    """

    valence: float = 0.0
    energy: float = 0.0
    tension: float = 0.0
    updated_at: str = field(default_factory=iso_now)

    def __post_init__(self) -> None:
        self.valence = clamp(self.valence, -0.35, 0.35)
        self.energy = clamp(self.energy, -0.35, 0.35)
        self.tension = clamp(self.tension, -0.35, 0.35)

    @property
    def significant(self) -> bool:
        return abs(self.valence) >= 0.08 or abs(self.tension) >= 0.08

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoodOffsetState:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class TemperamentState:
    """Deterministic per-day temperament draw that shifts the mood baseline."""

    word: str = ""
    date: str = ""
    valence_shift: float = 0.0
    energy_shift: float = 0.0
    tension_shift: float = 0.0
    drawn_at: str = field(default_factory=iso_now)

    def __post_init__(self) -> None:
        self.word = str(self.word or "").strip()[:24]
        self.date = str(self.date or "").strip()[:10]
        self.valence_shift = clamp(self.valence_shift, -0.3, 0.3)
        self.energy_shift = clamp(self.energy_shift, -0.3, 0.3)
        self.tension_shift = clamp(self.tension_shift, -0.3, 0.3)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemperamentState:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class ExpressionGuidance:
    """Cached expression guidance used to build the injected reply-suggestion block."""

    tone: str = ""
    can_say: str = ""
    avoid: str = ""
    generated_at: str = field(default_factory=iso_now)
    trigger: str = ""
    regime: str = ""
    provider_id: str = ""
    model_generated: bool = False

    def __post_init__(self) -> None:
        self.tone = str(self.tone or "").strip()[:200]
        self.can_say = str(self.can_say or "").strip()[:300]
        self.avoid = str(self.avoid or "").strip()[:200]
        self.trigger = str(self.trigger or "").strip()[:48]
        self.regime = str(self.regime or "").strip()[:120]
        self.provider_id = str(self.provider_id or "").strip()[:96]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpressionGuidance:
        return cls(
            **{key: data[key] for key in cls.__dataclass_fields__ if key in data}
        )


@dataclass(slots=True)
class StateLedger:
    user_key: str
    schema_version: int = 1
    state_version: int = 0
    message_watermark: int = 0
    memory_summary_watermark: int = 0
    processed_memory_summary_ids: list[str] = field(default_factory=list)
    attention_reconciliation_version: int = 0
    logical_day: str = ""
    mood: MoodState = field(default_factory=MoodState)
    mood_offset: MoodOffsetState = field(default_factory=MoodOffsetState)
    today_temperament: TemperamentState = field(default_factory=TemperamentState)
    last_user_message_ts: float = 0.0
    last_reviewed_watermark: int = 0
    last_batch_review_at: str = ""
    expression_guidance: ExpressionGuidance | None = None
    life_event_slots: list[str] = field(default_factory=list)
    life_event_slots_date: str = ""
    life_events_today: int = 0
    events: list[InnerEvent] = field(default_factory=list)
    attention_items: list[AttentionItem] = field(default_factory=list)
    intimacy: IntimacyState = field(default_factory=IntimacyState)
    jealousy: JealousyState = field(default_factory=JealousyState)
    diaries: list[DiaryEntry] = field(default_factory=list)
    proactive_message_ts: float = 0.0
    proactive_applied_stage: int = 0
    proactive_replied: bool = False
    proactive_evidence_progress: list[ProactiveEvidenceProgress] = field(
        default_factory=list
    )
    last_settled_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], user_key: str = "") -> StateLedger:
        raw_summary_ids = data.get("processed_memory_summary_ids", [])
        summary_ids = raw_summary_ids if isinstance(raw_summary_ids, list) else []
        return cls(
            user_key=str(data.get("user_key") or user_key),
            schema_version=int(data.get("schema_version", 1)),
            state_version=int(data.get("state_version", 0)),
            message_watermark=int(data.get("message_watermark", 0)),
            memory_summary_watermark=int(data.get("memory_summary_watermark", 0)),
            processed_memory_summary_ids=[
                str(item)[:160] for item in summary_ids if str(item).strip()
            ][-32:],
            attention_reconciliation_version=max(
                0, int(data.get("attention_reconciliation_version", 0) or 0)
            ),
            logical_day=str(data.get("logical_day") or ""),
            mood=MoodState.from_dict(data.get("mood") or {}),
            mood_offset=MoodOffsetState.from_dict(data.get("mood_offset") or {}),
            today_temperament=TemperamentState.from_dict(
                data.get("today_temperament") or {}
            ),
            last_user_message_ts=float(data.get("last_user_message_ts", 0.0) or 0.0),
            last_reviewed_watermark=max(
                0, int(data.get("last_reviewed_watermark", 0) or 0)
            ),
            last_batch_review_at=str(data.get("last_batch_review_at") or ""),
            expression_guidance=(
                ExpressionGuidance.from_dict(data["expression_guidance"])
                if isinstance(data.get("expression_guidance"), dict)
                else None
            ),
            life_event_slots=[
                str(item)[:32]
                for item in data.get("life_event_slots", [])
                if isinstance(item, (str, int, float)) and str(item).strip()
            ][-8:],
            life_event_slots_date=str(data.get("life_event_slots_date") or "")[:10],
            life_events_today=max(0, int(data.get("life_events_today", 0) or 0)),
            events=[
                InnerEvent.from_dict(item)
                for item in data.get("events", [])
                if isinstance(item, dict)
            ],
            attention_items=[
                AttentionItem.from_dict(item)
                for item in data.get("attention_items", [])
                if isinstance(item, dict) and str(item.get("content") or "").strip()
            ],
            intimacy=IntimacyState.from_dict(data.get("intimacy") or {}),
            jealousy=JealousyState.from_dict(data.get("jealousy") or {}),
            diaries=[
                DiaryEntry.from_dict(item)
                for item in data.get("diaries", [])
                if isinstance(item, dict)
            ],
            proactive_message_ts=float(data.get("proactive_message_ts", 0.0) or 0.0),
            proactive_applied_stage=int(data.get("proactive_applied_stage", 0) or 0),
            proactive_replied=bool(data.get("proactive_replied", False)),
            proactive_evidence_progress=[
                ProactiveEvidenceProgress.from_dict(item)
                for item in data.get("proactive_evidence_progress", [])
                if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
            ][-32:],
            last_settled_at=str(data.get("last_settled_at") or iso_now()),
            updated_at=str(data.get("updated_at") or iso_now()),
        )


@dataclass(slots=True)
class AttentionObservation:
    action: AttentionAction
    content: str = ""
    kind: AttentionKind = "follow_up"
    status: AttentionStatus = "open"
    actor: str = "both"
    time_hint: str = ""
    due_at: str = ""
    confidence: float = 0.7
    explicit: bool = True
    source: str = "local_rule"
    item_id: str = ""
    item_version: int | None = None
    fingerprint: str = ""
    expected_state_version: int | None = None
    evidence_quote: str = ""
    evidence_speaker: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        self.action = str(self.action or "").strip().lower()
        if self.kind not in {"commitment", "plan", "remember", "follow_up"}:
            self.kind = "follow_up"
        if self.status not in {
            "proposed",
            "open",
            "completed",
            "cancelled",
            "superseded",
            "archived",
        }:
            self.status = "open"
        self.content = str(self.content or "").strip()[:240]
        self.actor = str(self.actor or "both").strip()[:40]
        self.time_hint = str(self.time_hint or "").strip()[:80]
        self.due_at = str(self.due_at or "").strip()[:64]
        self.confidence = clamp(self.confidence)
        self.item_id = str(self.item_id or "").strip()[:80]
        self.evidence_quote = str(self.evidence_quote or "").strip()[:240]
        self.evidence_speaker = str(self.evidence_speaker or "").strip().lower()[:24]
        self.note = str(self.note or "").strip()[:240]


@dataclass(slots=True)
class EventObservation:
    action: ObservationAction
    fact: str
    emotional_meaning: str
    target: str = "unknown"
    target_basis: str = ""
    evidence_quote: str = ""
    evidence_speaker: str = ""
    category: EventCategory = "concrete"
    valence: float = 0.0
    intensity: float = 0.35
    confidence: float = 0.5
    source: str = "local_rule"
    event_id: str = ""
    event_version: int | None = None
    fingerprint: str = ""
    message_watermark: int = 0
    expected_state_version: int | None = None
    uncertain: bool = False
    note: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.action = str(self.action or "").strip().lower()
        self.fact = bound_complete_text(self.fact, EVENT_FACT_STORAGE_CHARS)
        self.emotional_meaning = str(self.emotional_meaning or "").strip()[:240]
        self.target = str(self.target or "unknown").strip()[:80] or "unknown"
        self.target_basis = str(self.target_basis or "").strip()[:120]
        self.evidence_quote = str(self.evidence_quote or "").strip()[:240]
        self.evidence_speaker = str(self.evidence_speaker or "").strip().lower()[:24]
        if self.category not in {
            "transient",
            "episodic",
            "psychological",
            "concrete",
        }:
            self.category = "concrete"
        self.valence = clamp(self.valence, -1.0, 1.0)
        self.intensity = clamp(self.intensity)
        self.confidence = clamp(self.confidence)
        self.tags = list(
            dict.fromkeys(
                str(tag).strip()[:40] for tag in self.tags if str(tag).strip()
            )
        )


@dataclass(slots=True)
class SensitivityParams:
    """User-tunable multipliers controlling how strongly emotions move and show."""

    overall: float = 1.0
    negative: float = 1.0
    positive: float = 1.0
    recovery: float = 1.0
    attachment: float = 1.0

    def __post_init__(self) -> None:
        self.overall = clamp(self.overall, 0.0, 2.0) or 0.0001
        self.negative = clamp(self.negative, 0.0, 2.0)
        self.positive = clamp(self.positive, 0.0, 2.0)
        self.recovery = clamp(self.recovery, 0.1, 2.0) or 0.1
        self.attachment = clamp(self.attachment, 0.0, 2.0)

    def direction_factor(self, valence: float) -> float:
        return self.negative if valence < 0 else self.positive


@dataclass(slots=True)
class IntrinsicParams:
    """Configuration snapshot for endogenous mood dynamics applied during decay."""

    night_hours: tuple[int, ...] = ()
    night_strength: float = 0.0
    night_missing_after_hours: float = 0.0
    temperament_enabled: bool = False
    drift_amplitude: float = 0.0
    temperament_words: tuple[str, ...] = ()
    sensitivity: SensitivityParams = field(default_factory=SensitivityParams)
