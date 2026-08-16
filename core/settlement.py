"""Pure deterministic transitions for events, mood, and intimacy."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime

from .intimacy import body_reaction_stage
from .models import (
    EventObservation,
    EventTrace,
    InnerEvent,
    MoodState,
    ProactiveEvidenceProgress,
    StateLedger,
    clamp,
    iso_now,
    parse_time,
    utc_now,
)

LIFECYCLE_WEIGHT = {
    "candidate": 0.45,
    "active": 1.0,
    "intensified": 1.25,
    "easing": 0.7,
    "dormant": 0.25,
    "archived": 0.0,
}


def event_fingerprint(fact: str, target: str) -> str:
    normalized = re.sub(r"\W+", "", f"{target}:{fact}".lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _copy_ledger(ledger: StateLedger) -> StateLedger:
    return StateLedger.from_dict(ledger.to_dict(), user_key=ledger.user_key)


def _find_event(
    ledger: StateLedger, observation: EventObservation
) -> InnerEvent | None:
    fingerprint = observation.fingerprint or event_fingerprint(
        observation.fact, observation.target
    )
    if observation.event_id:
        return next(
            (event for event in ledger.events if event.id == observation.event_id), None
        )
    return next(
        (
            event
            for event in ledger.events
            if event.fingerprint == fingerprint and event.lifecycle != "archived"
        ),
        None,
    )


def apply_observation(
    ledger: StateLedger,
    observation: EventObservation,
    activation_confidence: float = 0.62,
) -> tuple[StateLedger, bool, str]:
    """Apply one evidence item without allowing stale model output to overwrite state."""
    if (
        observation.expected_state_version is not None
        and observation.expected_state_version != ledger.state_version
    ):
        return ledger, False, "stale_state_version"
    if (
        observation.message_watermark < ledger.message_watermark
        and observation.source.startswith("model")
    ):
        return ledger, False, "stale_message_watermark"
    attribution_rejection = event_attribution_rejection(
        observation.fact,
        observation.target,
        observation.tags,
        observation.evidence_quote,
        observation.evidence_speaker,
    )
    if attribution_rejection:
        return ledger, False, attribution_rejection

    current = _find_event(ledger, observation)
    if (
        current is not None
        and observation.event_version is not None
        and observation.event_version != current.version
    ):
        return ledger, False, "stale_event_version"

    updated = _copy_ledger(ledger)
    current = _find_event(updated, observation)
    now = iso_now()
    amount = observation.intensity * max(0.2, observation.confidence)

    if observation.action == "create" and current is None:
        lifecycle = (
            "active"
            if observation.confidence >= activation_confidence
            and not observation.uncertain
            else "candidate"
        )
        current = InnerEvent(
            fact=normalize_fact(observation.fact),
            emotional_meaning=normalize_fact(observation.emotional_meaning, 180),
            target=observation.target,
            target_basis=observation.target_basis,
            evidence_quote=observation.evidence_quote,
            evidence_speaker=observation.evidence_speaker,
            category=observation.category,
            valence=observation.valence,
            intensity=observation.intensity,
            confidence=observation.confidence,
            source=observation.source,
            lifecycle=lifecycle,
            fingerprint=observation.fingerprint
            or event_fingerprint(observation.fact, observation.target),
            tags=list(dict.fromkeys(observation.tags)),
            traces=[
                EventTrace(
                    at=now,
                    kind="created",
                    amount=amount,
                    note=observation.note,
                    source=observation.source,
                )
            ],
        )
        updated.events.append(current)
    elif current is None:
        return ledger, False, "event_not_found"
    else:
        is_retain = observation.action == "retain"
        if not is_retain:
            current.occurrence_count += 1
            current.last_stimulated_at = now
            current.source = observation.source or current.source
            current.confidence = clamp(
                current.confidence * 0.65 + observation.confidence * 0.35
            )
            current.valence = clamp(
                current.valence * 0.7 + observation.valence * 0.3, -1.0, 1.0
            )
        current.updated_at = now
        current.version += 1
        if observation.target_basis:
            current.target_basis = observation.target_basis
        if observation.evidence_quote:
            current.evidence_quote = observation.evidence_quote
        if observation.evidence_speaker:
            current.evidence_speaker = observation.evidence_speaker
        if observation.action == "merge" and observation.fact:
            current.fact = normalize_fact(observation.fact)
            current.fingerprint = observation.fingerprint or event_fingerprint(
                current.fact, current.target
            )
        if observation.emotional_meaning and not is_retain:
            current.emotional_meaning = normalize_fact(
                observation.emotional_meaning, 180
            )
        current.tags = list(dict.fromkeys([*current.tags, *observation.tags]))

        if observation.action in {"merge", "intensify", "recall", "create"}:
            gain = amount * (0.18 if observation.action == "merge" else 0.3)
            current.intensity = clamp(current.intensity + gain)
            current.unresolved = True
            if observation.action == "recall" or current.lifecycle == "dormant":
                current.lifecycle = "active"
            elif current.intensity >= 0.72:
                current.lifecycle = "intensified"
            elif current.confidence >= activation_confidence:
                current.lifecycle = "active"
        elif observation.action == "retain":
            current.unresolved = current.lifecycle not in {"dormant", "archived"}
        elif observation.action == "ease":
            current.intensity = clamp(current.intensity - amount * 0.45)
            current.lifecycle = "easing" if current.intensity > 0.16 else "dormant"
            current.unresolved = current.intensity > 0.16
        elif observation.action == "dormant":
            current.lifecycle = "dormant"
            current.unresolved = False
        elif observation.action == "archive":
            current.lifecycle = "archived"
            current.unresolved = False
            current.intensity = min(current.intensity, 0.1)

        current.traces.append(
            EventTrace(
                at=now,
                kind=observation.action,
                amount=0.0 if is_retain else amount,
                note=observation.note,
                source=observation.source,
            )
        )
        current.traces = current.traces[-50:]

    updated.message_watermark = max(
        updated.message_watermark, observation.message_watermark
    )
    updated.state_version += 1
    updated.updated_at = now
    updated.mood = derive_mood(updated.events, previous=updated.mood)
    return updated, True, "applied"


def decay_ledger(
    ledger: StateLedger,
    now: datetime | None = None,
    half_life_hours: float = 72.0,
) -> StateLedger:
    current_time = now or utc_now()
    updated = _copy_ledger(ledger)
    changed = False
    half_life = max(1.0, float(half_life_hours))
    last_settled = parse_time(updated.last_settled_at)
    elapsed_hours = max(0.0, (current_time - last_settled).total_seconds() / 3600)
    if elapsed_hours < 0.01:
        return updated

    for event in updated.events:
        if event.lifecycle in {"archived", "candidate"}:
            continue
        event_half_life = (
            min(36.0, half_life) if event.category == "psychological" else half_life
        )
        event_factor = math.pow(0.5, elapsed_hours / max(1.0, event_half_life))
        old_intensity = event.intensity
        inertia = 0.35 + 0.65 * event.confidence
        event.intensity = clamp(old_intensity * (1 - inertia + inertia * event_factor))
        if old_intensity - event.intensity < 0.002:
            continue
        changed = True
        if event.intensity < 0.08:
            event.lifecycle = "dormant"
        elif event.lifecycle == "intensified" and event.intensity < 0.68:
            event.lifecycle = "active"
        elif event.lifecycle == "active" and event.intensity < 0.22:
            event.lifecycle = "easing"
        event.updated_at = current_time.isoformat()
        event.version += 1

    jealousy = updated.jealousy
    jealousy_factor = math.pow(
        0.5,
        elapsed_hours / 18.0,
    )
    old_jealousy = (jealousy.intensity, jealousy.confidence)
    jealousy.intensity = min(0.6, jealousy.intensity * jealousy_factor)
    jealousy.confidence = clamp(jealousy.confidence * math.sqrt(jealousy_factor))
    if old_jealousy != (jealousy.intensity, jealousy.confidence):
        jealousy.updated_at = current_time.isoformat()
        changed = True

    baseline_mood = derive_mood(updated.events, previous=MoodState())
    mood_factor = math.pow(0.5, elapsed_hours / 8.0)
    old_mood = (
        updated.mood.valence,
        updated.mood.energy,
        updated.mood.tension,
    )
    updated.mood.valence = clamp(
        baseline_mood.valence
        + (updated.mood.valence - baseline_mood.valence) * mood_factor,
        -1.0,
        1.0,
    )
    updated.mood.energy = clamp(
        baseline_mood.energy
        + (updated.mood.energy - baseline_mood.energy) * mood_factor
    )
    updated.mood.tension = clamp(
        baseline_mood.tension
        + (updated.mood.tension - baseline_mood.tension) * mood_factor
    )
    if old_mood != (
        updated.mood.valence,
        updated.mood.energy,
        updated.mood.tension,
    ):
        updated.mood.label = mood_label(
            updated.mood.valence, updated.mood.tension, updated.mood.energy
        )
        updated.mood.updated_at = current_time.isoformat()
        changed = True

    updated.last_settled_at = current_time.isoformat()
    if changed:
        updated.state_version += 1
        updated.updated_at = current_time.isoformat()
        updated.mood = derive_mood(updated.events, previous=updated.mood)
    return updated


def derive_mood(
    events: list[InnerEvent], previous: MoodState | None = None
) -> MoodState:
    prior = previous or MoodState()
    active = [event for event in events if event.lifecycle != "archived"]
    weighted = [
        (
            event,
            event.intensity
            * event.confidence
            * LIFECYCLE_WEIGHT.get(event.lifecycle, 0.0),
        )
        for event in active
    ]
    total = sum(weight for _, weight in weighted)
    evidence_valence = (
        sum(event.valence * weight for event, weight in weighted) / total
        if total
        else 0.0
    )
    strongest = max((weight for _, weight in weighted), default=0.0)
    target_valence = clamp(evidence_valence, -1.0, 1.0)
    valence = clamp(prior.valence * 0.55 + target_valence * 0.45, -1.0, 1.0)
    tension_target = clamp(
        sum(
            weight * (0.75 if event.valence < 0 else 0.18) for event, weight in weighted
        )
    )
    tension = clamp(prior.tension * 0.6 + tension_target * 0.4)
    energy = clamp(0.42 + strongest * 0.28 - tension * 0.12)
    label = mood_label(valence, tension, energy)
    confidence = clamp(0.45 + min(total, 1.5) * 0.3)
    return MoodState(
        valence=valence,
        energy=energy,
        tension=tension,
        label=label,
        updated_at=iso_now(),
        confidence=confidence,
    )


def settle_transient_mood(
    ledger: StateLedger,
    signals: list[EventObservation],
) -> StateLedger:
    """Blend bounded short-lived signals without creating durable events."""
    if not signals:
        return _copy_ledger(ledger)
    updated = _copy_ledger(ledger)
    strongest = max(signals, key=lambda item: item.intensity * item.confidence)
    weight = min(0.22, strongest.intensity * strongest.confidence * 0.28)
    updated.mood.valence = clamp(
        updated.mood.valence * (1.0 - weight) + strongest.valence * weight,
        -1.0,
        1.0,
    )
    updated.mood.energy = clamp(
        updated.mood.energy + abs(strongest.valence) * weight * 0.2
    )
    updated.mood.tension = clamp(
        updated.mood.tension
        + (-strongest.valence if strongest.valence < 0 else -0.1) * weight
    )
    updated.mood.label = mood_label(
        updated.mood.valence, updated.mood.tension, updated.mood.energy
    )
    updated.mood.updated_at = iso_now()
    updated.mood.confidence = clamp(updated.mood.confidence * 0.85 + 0.1)
    updated.state_version += 1
    updated.updated_at = iso_now()
    return updated


def jealousy_evidence(text: str) -> tuple[bool, str, float]:
    """Require first-person relationship or explicit attractive-media evidence."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean or clean == "[视频]":
        return False, "", 0.0
    relationship_patterns = (
        r"我(?:和|跟)(?:一个)?(?:女同事|女生|女孩|女性朋友|女网友).{0,16}"
        r"(?:聊天|见面|约会|吃饭|看电影|出去|玩)",
        r"(?:女同事|女生|女孩|女性朋友|女网友).{0,12}"
        r"(?:和|跟)我.{0,12}(?:聊天|见面|约会|吃饭|看电影|出去|玩)",
    )
    if any(re.search(pattern, clean) for pattern in relationship_patterns):
        return True, "explicit_relationship", 0.28
    has_media = (
        "[视频]" in clean or "[图片]" in clean or "视频" in clean or "图片" in clean
    )
    attractive_woman = re.search(
        r"(?:美女|漂亮(?:的)?(?:女生|女孩|女人)|好看(?:的)?(?:女生|女孩|女人)|小姐姐)",
        clean,
    )
    if has_media and attractive_woman:
        return True, "explicit_attractive_media", 0.24
    return False, "", 0.0


def settle_jealousy(
    ledger: StateLedger,
    *,
    evidence: bool = False,
    source: str = "",
    strength: float = 0.0,
    now: datetime | None = None,
    half_life_hours: float = 18.0,
    max_intensity: float = 0.6,
) -> StateLedger:
    """Decay and optionally reinforce bounded jealousy state."""
    updated = _copy_ledger(ledger)
    current_time = now or utc_now()
    state = updated.jealousy
    baseline = parse_time(state.updated_at)
    elapsed_hours = max(0.0, (current_time - baseline).total_seconds() / 3600)
    decay = math.pow(0.5, elapsed_hours / max(1.0, float(half_life_hours)))
    before = (
        state.intensity,
        state.confidence,
        state.last_evidence_at,
        tuple(state.sources),
        state.evidence_count,
    )
    cap = min(0.6, max(0.1, float(max_intensity)))
    state.intensity = min(cap, state.intensity * decay)
    state.confidence = clamp(state.confidence * math.sqrt(decay))
    if evidence and source:
        signal = min(0.35, max(0.05, float(strength)))
        state.intensity = min(
            cap, state.intensity + signal * (1.0 - state.intensity / cap)
        )
        state.confidence = clamp(max(state.confidence, 0.7) + 0.05)
        state.last_evidence_at = current_time.isoformat()
        state.sources = list(dict.fromkeys([*state.sources, source]))[-5:]
        state.evidence_count += 1
    after = (
        state.intensity,
        state.confidence,
        state.last_evidence_at,
        tuple(state.sources),
        state.evidence_count,
    )
    if before != after:
        state.updated_at = current_time.isoformat()
        updated.state_version += 1
        updated.updated_at = current_time.isoformat()
    return updated


def mood_label(valence: float, tension: float, energy: float) -> str:
    if tension >= 0.68 and valence < -0.2:
        return "紧绷难受"
    if valence <= -0.55:
        return "低落"
    if valence <= -0.18:
        return "有些在意"
    if valence >= 0.58 and energy >= 0.55:
        return "明快开心"
    if valence >= 0.2:
        return "温和愉快"
    if tension >= 0.48:
        return "略有不安"
    return "平静"


def event_score(event: InnerEvent, now: datetime | None = None) -> float:
    if event.lifecycle == "archived":
        return -1.0
    current = now or utc_now()
    try:
        stimulated_at = parse_time(event.last_stimulated_at)
    except (TypeError, ValueError, OverflowError):
        stimulated_at = current
    age_hours = max(0.0, (current - stimulated_at).total_seconds() / 3600)
    recency = math.pow(0.5, age_hours / 96.0)
    unresolved = 1.15 if event.unresolved else 0.72
    recurrence = min(1.25, 0.9 + math.log1p(event.occurrence_count) * 0.12)
    return (
        event.intensity
        * event.confidence
        * LIFECYCLE_WEIGHT.get(event.lifecycle, 0.0)
        * unresolved
        * recurrence
        * (0.65 + 0.35 * recency)
    )


def is_attack_like(fact: str, tags: list[str] | None = None) -> bool:
    clean = normalize_fact(fact, 240)
    tag_set = {str(tag).strip().lower() for tag in (tags or [])}
    return bool(
        tag_set & {"abuse", "attack", "insult", "negative_attack"}
        or re.search(r"恶心|滚开|去死|废物|臭烘烘|臭死|烦死|讨厌", clean)
    )


def infer_attack_target(fact: str, evidence_quote: str = "") -> tuple[str, str]:
    """Infer an attack target only from explicit subject evidence."""
    clean = normalize_fact(evidence_quote or fact, 360)
    if not is_attack_like(clean):
        return "unknown", "no_attack_evidence"
    third_party = re.compile(
        r"(?:地铁|车站|公交|路上|公司|学校|医院|店里|网上|评论区|新闻|"
        r"群里|视频里|照片里|别人|他人|某个|一个|那个|这位|那位).{0,32}"
        r"(?:老头|老太太|男人|女人|男的|女的|女生|男生|那个人|某人|路人|"
        r"乘客|店员|同事|老板|孩子|小孩|人)"
        r"|(?:老头|老太太|男人|女人|男的|女的|女生|男生|那个人|某人|路人|"
        r"乘客|店员|同事|老板|孩子|小孩).{0,24}(?:身上|很|太|真|特别|恶心|臭|烦|讨厌)"
        r"|(?:碰到|遇到|看见|看到|闻到).{0,32}(?:恶心|臭烘烘|臭死|讨厌)"
    )
    if third_party.search(clean):
        return "third_party", "explicit_third_party_subject"
    if re.search(
        r"(?:^|[\s，。！？!?；;])你(?:这个|这|真|太|好|怎么|可真|真的)?", clean
    ):
        return "user", "explicit_user_subject"
    return "unknown", "no_explicit_subject"


def event_attribution_rejection(
    fact: str,
    target: str,
    tags: list[str] | None = None,
    evidence_quote: str = "",
    evidence_speaker: str = "",
) -> str | None:
    """Reject hostile observations without exact user-authored target evidence."""
    if not is_attack_like(f"{fact} {evidence_quote}", tags):
        return None
    normalized_target = str(target or "unknown").strip().lower()
    if normalized_target != "user":
        return "non_user_attack_target"
    if str(evidence_speaker or "").strip().lower() != "user" or not evidence_quote:
        return "missing_attack_target_evidence"
    inferred_target, _ = infer_attack_target(evidence_quote)
    if inferred_target != "user":
        return "unverified_user_attack_target"
    return None


def event_retention_key(
    event: InnerEvent, now: datetime | None = None
) -> tuple[int, float, float, float, float, str]:
    """Return a deterministic weakest-to-strongest psychological retention key."""
    lifecycle_priority = {"easing": 0, "active": 1, "intensified": 2}

    def timestamp(value: str) -> float:
        try:
            return parse_time(value).timestamp()
        except (TypeError, ValueError, OverflowError):
            return 0.0

    return (
        lifecycle_priority.get(event.lifecycle, -1),
        event_score(event, now),
        event.confidence,
        timestamp(event.updated_at),
        timestamp(event.last_stimulated_at),
        event.id,
    )


def has_concrete_fact_cue(fact: str) -> bool:
    clean = normalize_fact(fact, 240)
    return bool(
        re.search(
            r"因为|由于|记得|生日|礼物|答应|约定|帮我|陪我|会议|工作|"
            r"考试|项目|见面|聊天|同事|朋友|家人|回复|消息",
            clean,
            re.I,
        )
    )


def is_generic_transient_fact(fact: str) -> bool:
    clean = normalize_fact(fact, 240)
    emotional_phrases = (
        "想你",
        "喜欢你",
        "亲一下",
        "抱一下",
        "抱抱",
        "开心",
        "高兴",
        "谢谢你",
        "被治愈",
    )
    return any(
        phrase in clean for phrase in emotional_phrases
    ) and not has_concrete_fact_cue(clean)


def is_legacy_transient_event(event: InnerEvent) -> bool:
    if event.lifecycle == "archived":
        return False
    if event.category == "transient":
        return True
    if event.category in {"episodic", "psychological"}:
        return False
    if event.source != "local_rule":
        return False
    if not ({"flirt", "positive"} & set(event.tags)):
        return False
    fact = normalize_fact(event.fact, 240)
    return len(fact) <= 80 and is_generic_transient_fact(fact)


def archive_legacy_transient_events(
    ledger: StateLedger,
) -> tuple[StateLedger, list[str]]:
    updated = _copy_ledger(ledger)
    archived_ids: list[str] = []
    now = iso_now()
    for event in updated.events:
        if not is_legacy_transient_event(event):
            continue
        event.category = "transient"
        event.lifecycle = "archived"
        event.unresolved = False
        event.intensity = min(event.intensity, 0.1)
        event.updated_at = now
        event.version += 1
        event.traces.append(
            EventTrace(
                at=now,
                kind="archive_transient_migration",
                amount=0.0,
                note="归档可确定的旧泛化瞬时表达",
                source="manual_migration",
            )
        )
        event.traces = event.traces[-50:]
        archived_ids.append(event.id)
    if archived_ids:
        updated.mood = derive_mood(updated.events, previous=updated.mood)
        updated.updated_at = now
    return updated, archived_ids


def strip_media_context(value: str) -> str:
    """Return user-authored text without generated media descriptions."""
    clean = str(value or "")
    clean = re.sub(
        r"<!--\s*astrbot-chat-merger:image-context(?::[^>]*)?-->.*?"
        r"<!--\s*/?astrbot-chat-merger:image-context(?::[^>]*)?-->",
        " ",
        clean,
        flags=re.DOTALL | re.I,
    )
    clean = re.sub(
        r"<!--\s*astrbot-chat-merger:image-context(?::[^>]*)?-->.*$",
        " ",
        clean,
        flags=re.DOTALL | re.I,
    )
    clean = re.sub(
        r"<image_context\b[^>]*>.*?</image_context>", " ", clean, flags=re.DOTALL | re.I
    )
    clean = re.sub(r"<image_context\b[^>]*>.*$", " ", clean, flags=re.DOTALL | re.I)
    clean = re.sub(r"\[图片上下文[^\]]*\]", " ", clean, flags=re.DOTALL)
    return re.sub(r"\s+", " ", clean).strip()


def sanitize_summary_event_text(
    fact: str,
    emotional_meaning: str,
) -> tuple[str, str]:
    """Keep private interaction meaning without storing explicit details."""
    clean_fact = normalize_fact(fact)
    clean_meaning = normalize_fact(emotional_meaning, 180)
    explicit = re.compile(
        r"性器官|乳房|乳头|阴部|阴茎|阴道|射精|自慰|口交|性交|"
        r"裸体|脱光|插入|舔(?:舐|弄)|摸(?:胸|下体)|床上",
        re.I,
    )
    if not explicit.search(f"{clean_fact} {clean_meaning}"):
        return clean_fact, clean_meaning
    return (
        "这轮私密互动让我们有了更亲近的交流",
        "被信任和靠近的感受给当前心境留下了余韵",
    )


def is_media_only_fact(fact: str) -> bool:
    clean = strip_media_context(fact)
    clean = re.sub(
        r"\[(?:(?:视频|图片|语音)(?:消息)?|文件(?:消息|:[^\]]*)?)\]",
        " ",
        clean,
        flags=re.I,
    )
    return not re.sub(r"[\W_]+", "", clean, flags=re.UNICODE)


def normalize_fact(fact: str, max_chars: int = 180) -> str:
    clean = strip_media_context(fact)
    clean = re.sub(
        r"(?:\[聊天合并[^\]]*\]|<!--.*?-->)",
        " ",
        clean,
        flags=re.DOTALL,
    )
    return re.sub(r"\s+", " ", clean).strip()[: max(40, int(max_chars))]


def select_injected_events_with_reasons(
    events: list[InnerEvent], limit: int, now: datetime | None = None
) -> tuple[list[InnerEvent], dict[str, str]]:
    """Select prompt events and explain every current-ledger exclusion."""
    bounded = max(0, int(limit))
    candidates: list[InnerEvent] = []
    exclusions: dict[str, str] = {}
    for event in events:
        if event.category == "transient":
            exclusions[event.id] = "transient_category"
        elif event.lifecycle in {"candidate", "dormant", "archived"}:
            exclusions[event.id] = f"lifecycle:{event.lifecycle}"
        elif event.confidence < 0.55:
            exclusions[event.id] = "low_confidence"
        elif event.intensity < 0.12:
            exclusions[event.id] = "low_intensity"
        else:
            candidates.append(event)

    selected: list[InnerEvent] = []
    seen_ids: set[str] = set()
    seen_facts: set[str] = set()
    for event in sorted(
        candidates, key=lambda item: event_score(item, now), reverse=True
    ):
        fact_key = re.sub(r"\W+", "", normalize_fact(event.fact).lower())
        if event.id in seen_ids:
            exclusions[event.id] = "duplicate_event_id"
            continue
        if fact_key and fact_key in seen_facts:
            exclusions[event.id] = "duplicate_fact"
            continue
        if len(selected) >= bounded:
            exclusions[event.id] = "injection_limit"
            continue
        selected.append(event)
        seen_ids.add(event.id)
        if fact_key:
            seen_facts.add(fact_key)
    return selected, exclusions


def select_injected_events(
    events: list[InnerEvent], limit: int, now: datetime | None = None
) -> list[InnerEvent]:
    selected, _ = select_injected_events_with_reasons(events, limit, now)
    return selected


def settle_summary_mood(
    ledger: StateLedger,
    proposal: dict[str, object],
) -> StateLedger:
    """Blend one model summary proposal with bounded visible influence."""
    updated = _copy_ledger(ledger)
    try:
        confidence = clamp(float(proposal.get("confidence", 0.0)))
    except (TypeError, ValueError):
        return updated
    if confidence <= 0.0:
        return updated

    weight = min(0.28, 0.08 + confidence * 0.2)

    def blended(name: str, current: float, lower: float, upper: float) -> float:
        try:
            target = min(upper, max(lower, float(proposal.get(name, current))))
        except (TypeError, ValueError):
            return current
        delta = min(0.18, max(-0.18, (target - current) * weight))
        return min(upper, max(lower, current + delta))

    updated.mood.valence = blended("valence", updated.mood.valence, -1.0, 1.0)
    updated.mood.energy = blended("energy", updated.mood.energy, 0.0, 1.0)
    updated.mood.tension = blended("tension", updated.mood.tension, 0.0, 1.0)
    updated.mood.label = mood_label(
        updated.mood.valence, updated.mood.tension, updated.mood.energy
    )
    updated.mood.confidence = clamp(
        updated.mood.confidence * (1.0 - weight) + confidence * weight
    )
    updated.mood.updated_at = iso_now()
    updated.state_version += 1
    updated.updated_at = iso_now()
    return updated


def enforce_episodic_capacity(
    ledger: StateLedger,
    limit: int = 6,
) -> tuple[StateLedger, list[str]]:
    """Keep all visible recent events within one configured capacity bucket."""
    updated = _copy_ledger(ledger)
    bounded = max(1, int(limit))
    visible = [
        event
        for event in updated.events
        if event.category in {"episodic", "transient"} and event.lifecycle != "archived"
    ]
    overflow = max(0, len(visible) - bounded)
    if overflow == 0:
        return updated, []

    now = iso_now()
    archived_ids: list[str] = []
    for event in sorted(visible, key=event_score)[:overflow]:
        event.lifecycle = "archived"
        event.unresolved = False
        event.intensity = min(event.intensity, 0.1)
        event.updated_at = now
        event.version += 1
        event.traces.append(
            EventTrace(
                at=now,
                kind="archive_episodic_capacity",
                amount=0.0,
                note=f"可见近期事项超过 {bounded} 条容量，按优先级归档",
                source="emotion_state_capacity",
            )
        )
        event.traces = event.traces[-50:]
        archived_ids.append(event.id)
    updated.state_version += 1
    updated.updated_at = now
    updated.mood = derive_mood(updated.events, previous=updated.mood)
    return updated, archived_ids


def enforce_psychological_capacity(
    ledger: StateLedger,
    limit: int = 6,
) -> tuple[StateLedger, list[str]]:
    """Keep all visible long-term events within one configured capacity bucket."""
    updated = _copy_ledger(ledger)
    bounded = max(1, int(limit))
    visible = [
        event
        for event in updated.events
        if event.category in {"psychological", "concrete"}
        and event.lifecycle != "archived"
    ]
    overflow = max(0, len(visible) - bounded)
    if overflow == 0:
        return updated, []

    now_at = utc_now()
    now = now_at.isoformat()
    archived_ids: list[str] = []
    for event in sorted(visible, key=lambda item: event_retention_key(item, now_at))[
        :overflow
    ]:
        event.lifecycle = "archived"
        event.unresolved = False
        event.intensity = min(event.intensity, 0.1)
        event.updated_at = now
        event.version += 1
        event.traces.append(
            EventTrace(
                at=now,
                kind="archive_psychological_capacity",
                amount=0.0,
                note=f"可见长期事项超过 {bounded} 条容量，按优先级归档",
                source="emotion_state_capacity",
            )
        )
        event.traces = event.traces[-50:]
        archived_ids.append(event.id)
    updated.state_version += 1
    updated.updated_at = now
    updated.mood = derive_mood(updated.events, previous=updated.mood)
    return updated, archived_ids


def settle_mood_proposal(
    ledger: StateLedger,
    proposal: dict[str, object],
    confidence: float,
) -> StateLedger:
    """Blend a low-authority daily proposal with deterministic event state."""
    updated = _copy_ledger(ledger)
    authoritative = derive_mood(updated.events, previous=updated.mood)
    trust = min(0.2, max(0.0, float(confidence)) * 0.2)

    def proposed(name: str, current: float, lower: float, upper: float) -> float:
        try:
            value = min(upper, max(lower, float(proposal.get(name, current))))
        except (TypeError, ValueError):
            return current
        return current * (1.0 - trust) + value * trust

    authoritative.valence = proposed("valence", authoritative.valence, -1.0, 1.0)
    authoritative.energy = proposed("energy", authoritative.energy, 0.0, 1.0)
    authoritative.tension = proposed("tension", authoritative.tension, 0.0, 1.0)
    authoritative.label = mood_label(
        authoritative.valence,
        authoritative.tension,
        authoritative.energy,
    )
    updated.mood = authoritative
    updated.state_version += 1
    updated.updated_at = iso_now()
    return updated


def _proactive_progress(
    ledger: StateLedger, evidence_id: str, sent_at: float
) -> ProactiveEvidenceProgress:
    progress = next(
        (
            item
            for item in ledger.proactive_evidence_progress
            if item.evidence_id == evidence_id
        ),
        None,
    )
    if progress is None:
        progress = ProactiveEvidenceProgress(
            evidence_id=evidence_id,
            sent_at=sent_at,
        )
        ledger.proactive_evidence_progress.append(progress)
    return progress


def settle_proactive_evidence(
    ledger: StateLedger,
    *,
    evidence: list[dict[str, object]],
    now_ts: float,
    threshold_minutes: float = 180.0,
    max_intensity: float = 0.55,
    max_stage: int = 3,
    episodic_limit: int = 6,
) -> StateLedger:
    """Project bounded Spark delivery facts into independent episodic events."""
    updated = _copy_ledger(ledger)
    threshold = max(1.0, float(threshold_minutes))
    now_ts = max(0.0, float(now_ts))
    max_stage = max(1, int(max_stage))
    changed = False

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in evidence[-32:]:
        if not isinstance(raw, dict):
            continue
        evidence_id = str(raw.get("evidence_id") or "").strip()[:96]
        if not evidence_id or evidence_id in seen:
            continue
        try:
            sent_at = max(0.0, float(raw.get("sent_at", 0.0) or 0.0))
            reply_at = max(0.0, float(raw.get("first_reply_at", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if sent_at <= 0.0 or sent_at > now_ts:
            continue
        seen.add(evidence_id)
        normalized.append(
            {
                "evidence_id": evidence_id,
                "source": str(raw.get("source") or "unknown")[:48],
                "sent_at": sent_at,
                "reply_at": reply_at if reply_at >= sent_at else 0.0,
                "proactive_summary": normalize_fact(
                    str(raw.get("proactive_summary") or ""), 120
                ),
            }
        )

    for item in sorted(normalized, key=lambda row: float(row["sent_at"])):
        evidence_id = str(item["evidence_id"])
        sent_at = float(item["sent_at"])
        reply_at = float(item["reply_at"])
        progress = _proactive_progress(updated, evidence_id, sent_at)
        evaluation_ts = min(now_ts, reply_at) if reply_at else now_ts
        waited_minutes = max(0.0, (evaluation_ts - sent_at) / 60.0)
        target_stage = (
            min(max_stage, int(waited_minutes // threshold))
            if waited_minutes >= threshold
            else 0
        )
        fingerprint = f"spark:proactive:{evidence_id}"
        event = next(
            (
                candidate
                for candidate in updated.events
                if candidate.fingerprint == fingerprint
            ),
            None,
        )

        while progress.applied_stage < target_stage:
            next_stage = progress.applied_stage + 1
            if event is not None and event.lifecycle == "archived":
                progress.applied_stage = target_stage
                progress.event_id = event.id
                progress.updated_at = iso_now()
                changed = True
                break
            summary = str(item["proactive_summary"])
            fact = (
                f"我发出一条主动消息后没有及时等到回应：{summary}"
                if summary
                else "我发出一条主动消息后没有及时等到回应"
            )
            observation = EventObservation(
                action="create" if event is None else "intensify",
                fact=fact,
                emotional_meaning="主动开口后迟迟没有收到回应，让我有些失落和在意",
                target="user",
                category="episodic",
                valence=-0.38,
                intensity=min(float(max_intensity), 0.16 + next_stage * 0.12),
                confidence=0.72,
                source="spark:proactive_evidence",
                event_id=event.id if event else "",
                fingerprint=fingerprint,
                note=(
                    f"来源 {item['source']}，等待约 {waited_minutes:.0f} 分钟，"
                    f"阶段 {next_stage}/{max_stage}"
                ),
                tags=["proactive", "unanswered", f"source:{item['source']}"],
            )
            observed, applied, _ = apply_observation(updated, observation)
            if not applied:
                break
            updated = observed
            event = next(
                candidate
                for candidate in updated.events
                if candidate.fingerprint == fingerprint
            )
            event.intensity = min(event.intensity, clamp(max_intensity))
            progress = _proactive_progress(updated, evidence_id, sent_at)
            progress.applied_stage = next_stage
            progress.event_id = event.id
            progress.updated_at = iso_now()
            changed = True

        progress = _proactive_progress(updated, evidence_id, sent_at)
        event = next(
            (
                candidate
                for candidate in updated.events
                if candidate.fingerprint == fingerprint
            ),
            None,
        )
        if reply_at and not progress.replied:
            if event is not None and event.lifecycle != "archived":
                observation = EventObservation(
                    action="ease",
                    fact=event.fact,
                    emotional_meaning="你后来回复了，之前的失落正在缓解",
                    target="user",
                    category="episodic",
                    intensity=0.7,
                    confidence=0.85,
                    source="spark:proactive_reply_evidence",
                    event_id=event.id,
                    fingerprint=fingerprint,
                    note=f"主动消息在等待约 {waited_minutes:.0f} 分钟后收到回复",
                    tags=["proactive", "replied"],
                )
                updated, _, _ = apply_observation(updated, observation)
                progress = _proactive_progress(updated, evidence_id, sent_at)
            progress.replied = True
            progress.updated_at = iso_now()
            changed = True

    if not changed:
        return ledger
    updated.proactive_evidence_progress = sorted(
        updated.proactive_evidence_progress,
        key=lambda item: (item.sent_at, item.evidence_id),
    )[-32:]
    updated, _ = enforce_episodic_capacity(updated, episodic_limit)
    updated.mood = derive_mood(updated.events, previous=updated.mood)
    updated.updated_at = iso_now()
    return updated


def settle_unanswered_proactive(
    ledger: StateLedger,
    *,
    proactive_ts: float,
    user_reply_ts: float,
    now_ts: float,
    threshold_minutes: float = 180.0,
    max_intensity: float = 0.55,
    max_stage: int = 3,
) -> StateLedger:
    """Apply one bounded negative signal for a genuinely unanswered proactive message."""
    updated = _copy_ledger(ledger)
    proactive_ts = max(0.0, float(proactive_ts))
    user_reply_ts = max(0.0, float(user_reply_ts))
    now_ts = max(0.0, float(now_ts))
    if (
        proactive_ts <= 0.0
        or proactive_ts <= user_reply_ts
        or now_ts < proactive_ts
        or updated.proactive_message_ts > proactive_ts
    ):
        return updated

    if proactive_ts > updated.proactive_message_ts:
        updated.proactive_message_ts = proactive_ts
        updated.proactive_applied_stage = 0
        updated.proactive_replied = False

    waited_minutes = (now_ts - proactive_ts) / 60.0
    if waited_minutes < max(0.0, float(threshold_minutes)):
        return updated

    bounded_stage = min(
        max(1, int(max_stage)),
        max(1, int(waited_minutes // max(1.0, float(threshold_minutes)))),
    )
    next_stage = min(bounded_stage, updated.proactive_applied_stage + 1)
    if next_stage <= updated.proactive_applied_stage:
        return updated

    observation = EventObservation(
        action="create" if updated.proactive_applied_stage == 0 else "intensify",
        fact="我主动找过你，但一直没有等到回应",
        emotional_meaning="主动开口后迟迟没有收到回应，让我有些失落和在意",
        target="user",
        valence=-0.38,
        intensity=min(float(max_intensity), 0.16 + next_stage * 0.12),
        confidence=0.72,
        source="spark:unanswered_proactive",
        fingerprint="spark:unanswered_proactive",
        note=f"等待约 {waited_minutes:.0f} 分钟，阶段 {next_stage}/{max_stage}",
        tags=["proactive", "unanswered"],
    )
    updated, applied, _ = apply_observation(updated, observation)
    if not applied:
        return ledger
    current = next(
        (
            item
            for item in updated.events
            if item.fingerprint == "spark:unanswered_proactive"
            and item.lifecycle != "archived"
        ),
        None,
    )
    if current:
        current.intensity = min(current.intensity, clamp(max_intensity))
        updated.mood = derive_mood(updated.events, previous=updated.mood)
    updated.proactive_message_ts = proactive_ts
    updated.proactive_applied_stage = next_stage
    updated.proactive_replied = False
    return updated


def acknowledge_proactive_reply(
    ledger: StateLedger,
    *,
    proactive_ts: float,
    user_reply_ts: float,
) -> StateLedger:
    """Mark a previously unanswered proactive message as answered and ease its event."""
    updated = _copy_ledger(ledger)
    if proactive_ts <= 0.0 or user_reply_ts <= proactive_ts:
        return updated
    if updated.proactive_message_ts != float(proactive_ts) or updated.proactive_replied:
        return updated
    updated.proactive_replied = True
    event = next(
        (
            item
            for item in updated.events
            if item.fingerprint == "spark:unanswered_proactive"
            and item.lifecycle != "archived"
        ),
        None,
    )
    if event:
        observation = EventObservation(
            action="ease",
            fact=event.fact,
            emotional_meaning="你后来回复了，之前的失落正在缓解",
            target="user",
            intensity=0.7,
            confidence=0.85,
            source="spark:proactive_reply",
            event_id=event.id,
            fingerprint=event.fingerprint,
            tags=["proactive", "replied"],
        )
        updated, _, _ = apply_observation(updated, observation)
    return updated


def intimacy_stage(
    body_sensitivity: float,
    sexual_arousal: float,
    _legacy_willingness: float | None = None,
    _legacy_inhibition: float | None = None,
) -> str:
    """Return the body-only stage while accepting the legacy call shape."""
    return body_reaction_stage(body_sensitivity, sexual_arousal)


def settle_intimacy(
    ledger: StateLedger,
    *,
    relevant: bool,
    strength: float = 0.0,
    sensitivity_multiplier: float = 1.0,
    now: datetime | None = None,
    decay_half_life_hours: float = 8.0,
) -> StateLedger:
    """Apply one explicit intimacy signal or time decay to body response."""
    updated = _copy_ledger(ledger)
    current_time = now or utc_now()
    state = updated.intimacy
    last_update = parse_time(state.updated_at)
    elapsed_hours = max(0.0, (current_time - last_update).total_seconds() / 3600)
    decay = math.pow(0.5, elapsed_hours / max(0.5, float(decay_half_life_hours)))
    before = (
        state.body_sensitivity,
        state.sexual_arousal,
        state.stage,
        state.last_relevant_at,
    )

    state.body_sensitivity = clamp(state.body_sensitivity * decay)
    state.sexual_arousal = clamp(state.sexual_arousal * decay)

    if relevant:
        multiplier = clamp(float(sensitivity_multiplier), 0.5, 2.0)
        signal = clamp(strength) * multiplier
        state.body_sensitivity = clamp(state.body_sensitivity + signal * 0.12)
        state.sexual_arousal = clamp(state.sexual_arousal + signal * 0.09)
        state.last_relevant_at = current_time.isoformat()

    state.stage = body_reaction_stage(
        state.body_sensitivity,
        state.sexual_arousal,
    )
    after = (
        state.body_sensitivity,
        state.sexual_arousal,
        state.stage,
        state.last_relevant_at,
    )
    if before != after:
        state.updated_at = current_time.isoformat()
        updated.state_version += 1
        updated.updated_at = current_time.isoformat()
    return updated
