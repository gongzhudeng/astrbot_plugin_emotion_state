"""Deterministic lifecycle transitions for non-emotional attention items."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import datetime

from .models import (
    AttentionItem,
    AttentionObservation,
    EventTrace,
    StateLedger,
    iso_now,
    parse_time,
    utc_now,
)

_TERMINAL_STATUSES = {"completed", "cancelled", "superseded", "archived"}
_ATTENTION_ACTIONS = {
    "create",
    "confirm",
    "update",
    "complete",
    "cancel",
    "supersede",
}
_KIND_LABELS = {
    "commitment": "约定",
    "plan": "计划",
    "remember": "需要记住",
    "follow_up": "待接续",
}
_STATUS_LABELS = {
    "proposed": "待确认",
    "open": "仍待关注",
    "completed": "已完成",
    "cancelled": "已取消",
    "superseded": "已被替代",
    "archived": "已归档",
}
_COMPLETED_OR_PAST_CUES = re.compile(
    r"(?:已经|已|刚|早就).{0,20}(?:完成|做完|搞定|弄好|做好|发出|发送|发了|拍了|给了|到达|结束)"
    r"|(?:上午|今天|刚才|早些时候).{0,18}(?:完成|做完|搞定|弄好|发出|发送|发了|拍了|给了|到达|结束)"
)
_GENERIC_COMPLETION_EVIDENCE = re.compile(
    r"^(?:好了|行了|完成了|搞定了|弄好了|做好了|发了|拍了)[。！! ]*$"
)
_MEDIA_COMPLETION_EVIDENCE = re.compile(r"^\[(图片|语音|视频|文件)消息\]$")
_PARTIAL_COMPLETION_EVIDENCE = re.compile(
    r"(?:正在|刚开始|才开始|还在|进行中|只完成|完成了?一部分|先做了?一部分|还没(?:有)?完成)"
)
_MEDIA_ITEM_CUES = {
    "图片": re.compile(r"(?:图片|照片|相片|拍照|图像|一张图|张图)"),
    "语音": re.compile(r"(?:语音|声音|录音|音频)"),
    "视频": re.compile(r"(?:视频|录像|录屏)"),
    "文件": re.compile(r"(?:文件|文档|附件|资料)"),
}
_ONGOING_ATTENTION_CUES = re.compile(r"(?:以后|每次|每天|每晚|长期|一直|持续|不能忘)")


def normalize_attention_content(value: str, limit: int = 240) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    clean = re.sub(r"^[，。！？!?；;：:\s]+|[，。！？!?；;：:\s]+$", "", clean)
    return clean[: max(1, int(limit))]


def attention_fingerprint(content: str) -> str:
    normalized = re.sub(r"[\W_]+", "", normalize_attention_content(content).lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def attention_kind_label(kind: str) -> str:
    return _KIND_LABELS.get(str(kind), "待关注")


def attention_status_label(status: str) -> str:
    return _STATUS_LABELS.get(str(status), str(status))


def is_open_attention(item: AttentionItem) -> bool:
    return item.status not in _TERMINAL_STATUSES


def is_attention_overdue(item: AttentionItem, now: datetime | None = None) -> bool:
    if not is_open_attention(item) or not item.due_at:
        return False
    if _ONGOING_ATTENTION_CUES.search(f"{item.time_hint} {item.content}"):
        return False
    try:
        return parse_time(item.due_at) < (now or utc_now())
    except (TypeError, ValueError):
        return False


def archive_expired_attention_items(
    ledger: StateLedger,
    now: datetime | None = None,
) -> tuple[StateLedger, list[str]]:
    """Archive one-off items after an explicit deadline without claiming success."""
    current_time = now or utc_now()
    updated = _copy_ledger(ledger)
    timestamp = current_time.isoformat()
    archived_ids: list[str] = []
    for item in updated.attention_items:
        if not is_attention_overdue(item, current_time):
            continue
        item.status = "archived"
        item.archived_at = timestamp
        item.updated_at = timestamp
        item.version += 1
        item.evidence.append(
            EventTrace(
                at=timestamp,
                kind="archive_attention_expired",
                amount=0.0,
                note="已超过明确截止时间，自动归档（不视为完成）",
                source="emotion_state_expiry",
            )
        )
        item.evidence = item.evidence[-30:]
        archived_ids.append(item.id)
    if archived_ids:
        updated.state_version += 1
        updated.updated_at = timestamp
    return updated, archived_ids


def is_valid_completion_evidence(value: str) -> bool:
    """Reject bare success claims while accepting a concrete text or media record."""
    clean = normalize_attention_content(value)
    if not clean:
        return False
    if _PARTIAL_COMPLETION_EVIDENCE.search(clean):
        return False
    if _MEDIA_COMPLETION_EVIDENCE.fullmatch(clean):
        return True
    return not _GENERIC_COMPLETION_EVIDENCE.fullmatch(clean)


def completion_evidence_matches_item(item: AttentionItem, value: str) -> bool:
    """Require a media-only completion record to name the same medium as the item."""
    clean = normalize_attention_content(value)
    match = _MEDIA_COMPLETION_EVIDENCE.fullmatch(clean)
    if not match:
        return True
    cue = _MEDIA_ITEM_CUES[match.group(1)]
    return bool(cue.search(f"{item.content} {item.time_hint}"))


def is_short_lived_attention(item: AttentionItem) -> bool:
    evidence = f"{item.time_hint} {item.content}"
    return bool(
        re.search(
            r"(?:等一下|一会儿|待会儿|马上|现在|正在).{0,28}"
            r"(?:发|拍|聊|说|做|去|给|看|回复|玩)",
            evidence,
        )
    )


def _copy_ledger(ledger: StateLedger) -> StateLedger:
    return StateLedger.from_dict(ledger.to_dict(), user_key=ledger.user_key)


def _similarity(left: str, right: str) -> float:
    left_tokens = set(
        re.findall(r"[\w\u4e00-\u9fff]", normalize_attention_content(left))
    )
    right_tokens = set(
        re.findall(r"[\w\u4e00-\u9fff]", normalize_attention_content(right))
    )
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _find_item(
    ledger: StateLedger, observation: AttentionObservation
) -> AttentionItem | None:
    if observation.item_id:
        return next(
            (item for item in ledger.attention_items if item.id == observation.item_id),
            None,
        )
    fingerprint = observation.fingerprint or (
        attention_fingerprint(observation.content) if observation.content else ""
    )
    if fingerprint:
        exact = next(
            (
                item
                for item in reversed(ledger.attention_items)
                if item.fingerprint == fingerprint and is_open_attention(item)
            ),
            None,
        )
        if exact:
            return exact
    if not observation.content:
        return next(
            (
                item
                for item in reversed(ledger.attention_items)
                if is_open_attention(item)
            ),
            None,
        )
    candidates = [item for item in ledger.attention_items if is_open_attention(item)]
    if not candidates:
        return None
    matched = max(
        candidates,
        key=lambda item: _similarity(item.content, observation.content),
    )
    return (
        matched if _similarity(matched.content, observation.content) >= 0.42 else None
    )


def _trace(observation: AttentionObservation, kind: str, at: str) -> EventTrace:
    return EventTrace(
        at=at,
        kind=kind,
        amount=observation.confidence,
        note=observation.evidence_quote or observation.note,
        source=observation.source,
    )


def attention_observation_rejection(
    observation: AttentionObservation,
) -> str | None:
    """Reject weak model suggestions before they can change the durable ledger."""
    if observation.source != "livingmemory_summary":
        return "unsupported_attention_source"
    if observation.action not in _ATTENTION_ACTIONS:
        return "invalid_attention_action"
    if not observation.evidence_quote or not observation.evidence_speaker:
        return "missing_attention_evidence"

    evidence = f"{observation.evidence_quote} {observation.content}"
    if observation.action == "create":
        is_proposed = observation.status == "proposed"
        if observation.status not in {"proposed", "open"}:
            return "invalid_attention_create_status"
        if not observation.content:
            return "empty_attention_content"
        if is_proposed:
            if observation.evidence_speaker not in {"user", "assistant", "both"}:
                return "missing_proposal_evidence"
            if observation.confidence < 0.62:
                return "insufficient_proposed_attention_confidence"
        else:
            if observation.evidence_speaker != "user":
                return "missing_user_evidence"
            if not observation.explicit:
                return "non_explicit_attention"
            if observation.confidence < 0.82:
                return "insufficient_attention_confidence"
        if re.search(
            r"(?:等一下|一会儿|待会儿|马上|现在|正在).{0,28}"
            r"(?:发|拍|聊|说|做|去|给|看|回复)",
            evidence,
        ):
            return "short_lived_attention"
        if re.search(
            r"(?:有个|这个|那个).{0,12}(?:计划|约定|事项).{0,20}"
            r"(?:能看见|看得到|还在吗)|"
            r"(?:插件|模型|你).{0,16}(?:记错|说错|弄错|发癫)|"
            r"(?:不是|并非).{0,8}(?:计划|约定)|"
            r"(?:你|你说的|这个).{0,12}周[一二三四五六日天].{0,12}"
            r"(?:不对|不准|错了|不是)",
            evidence,
        ):
            return "quoted_or_corrected_attention"
        if _COMPLETED_OR_PAST_CUES.search(evidence) or re.search(
            r"(?:昨天|前天|上周|上个月|去年|已经过去|已经过期|过期的)",
            evidence,
        ):
            return "past_attention"
        if observation.due_at:
            try:
                if parse_time(observation.due_at) < utc_now():
                    return "past_attention"
            except (TypeError, ValueError):
                return "invalid_attention_due_at"
    else:
        allowed_speakers = (
            {"user", "assistant", "both"}
            if observation.action == "complete"
            else {"user"}
        )
        if observation.evidence_speaker not in allowed_speakers:
            return "missing_user_evidence"
        if observation.confidence < 0.78:
            return "insufficient_attention_review_confidence"
        if observation.action == "complete":
            if _PARTIAL_COMPLETION_EVIDENCE.search(
                normalize_attention_content(observation.evidence_quote)
            ):
                return "partial_completion_evidence"
            if not is_valid_completion_evidence(observation.evidence_quote):
                return "generic_completion_evidence"
    return None


def apply_attention_observation(
    ledger: StateLedger,
    observation: AttentionObservation,
) -> tuple[StateLedger, bool, str]:
    """Apply one attention observation without coupling it to emotional decay."""
    rejection = attention_observation_rejection(observation)
    if rejection:
        return ledger, False, rejection
    if (
        observation.expected_state_version is not None
        and observation.expected_state_version != ledger.state_version
    ):
        return ledger, False, "stale_state_version"

    current = _find_item(ledger, observation)
    if observation.action != "create" and observation.item_version is None:
        return ledger, False, "missing_attention_item_version"
    if (
        current is not None
        and observation.item_version is not None
        and observation.item_version != current.version
    ):
        return ledger, False, "stale_item_version"

    if observation.action != "create" and not observation.item_id:
        return ledger, False, "missing_attention_item_id"
    if observation.action == "create" and observation.item_id and current is None:
        return ledger, False, "attention_item_not_found"
    if observation.action != "create" and current is None:
        return ledger, False, "attention_item_not_found"
    if (
        observation.action == "complete"
        and current is not None
        and not completion_evidence_matches_item(current, observation.evidence_quote)
    ):
        return ledger, False, "unrelated_completion_evidence"
    if observation.action == "create" and current is None and not observation.content:
        return ledger, False, "empty_attention_content"

    updated = _copy_ledger(ledger)
    current = _find_item(updated, observation)
    now = iso_now()

    if observation.action == "create" and current is None:
        item = AttentionItem(
            content=normalize_attention_content(observation.content),
            kind=observation.kind,
            status=observation.status,
            actor=observation.actor,
            time_hint=observation.time_hint,
            due_at=observation.due_at,
            confidence=observation.confidence,
            explicit=observation.explicit,
            source=observation.source,
            evidence_speaker=observation.evidence_speaker,
            fingerprint=observation.fingerprint
            or attention_fingerprint(observation.content),
            evidence=[_trace(observation, "created", now)],
        )
        updated.attention_items.append(item)
    else:
        if current is None:
            return ledger, False, "attention_item_not_found"
        before = current.to_dict()
        if observation.action == "create":
            if current.status == "proposed" and observation.status == "open":
                current.status = "open"
            current.confidence = max(current.confidence, observation.confidence)
            current.explicit = current.explicit or observation.explicit
        elif observation.action == "confirm":
            if current.status == "proposed":
                current.status = "open"
            current.explicit = True
            current.confidence = max(current.confidence, observation.confidence)
        elif observation.action == "update":
            if current.status in _TERMINAL_STATUSES:
                return ledger, False, "attention_item_closed"
            if observation.content:
                current.content = normalize_attention_content(observation.content)
                current.fingerprint = observation.fingerprint or attention_fingerprint(
                    current.content
                )
            current.time_hint = observation.time_hint or current.time_hint
            current.due_at = observation.due_at or current.due_at
            current.confidence = max(current.confidence, observation.confidence)
        elif observation.action in {"complete", "cancel", "supersede"}:
            if current.status in _TERMINAL_STATUSES:
                return ledger, False, "attention_item_closed"
            current.status = {
                "complete": "completed",
                "cancel": "cancelled",
                "supersede": "superseded",
            }[observation.action]
            current.completed_at = now

        after = current.to_dict()
        if before == after and (
            observation.action in {"create", "confirm"}
            or any(
                trace.kind == observation.action
                and trace.note == (observation.evidence_quote or observation.note)
                and trace.source == observation.source
                for trace in current.evidence[-8:]
            )
        ):
            return ledger, False, "duplicate_attention_observation"
        current.source = observation.source or current.source
        current.last_evidence_at = now
        current.updated_at = now
        current.version += 1
        current.evidence.append(_trace(observation, observation.action, now))
        current.evidence = current.evidence[-30:]

    updated.state_version += 1
    updated.updated_at = now
    return updated, True, "applied"


def enforce_attention_capacity(
    ledger: StateLedger,
    limit: int = 8,
) -> tuple[StateLedger, list[str]]:
    """Archive the least-supported unfinished attention overflow."""
    updated = _copy_ledger(ledger)
    bounded = max(1, int(limit))
    active = [item for item in updated.attention_items if is_open_attention(item)]
    overflow = max(0, len(active) - bounded)
    if overflow == 0:
        return updated, []

    def retention_key(item: AttentionItem) -> tuple[int, int, float, float, str]:
        try:
            evidence_at = parse_time(item.last_evidence_at).timestamp()
        except (TypeError, ValueError, OverflowError):
            evidence_at = 0.0
        return (
            1 if item.status == "open" else 0,
            1 if item.explicit else 0,
            item.confidence,
            evidence_at,
            item.id,
        )

    now = iso_now()
    archived_ids: list[str] = []
    for item in sorted(active, key=retention_key)[:overflow]:
        item.status = "archived"
        item.archived_at = now
        item.updated_at = now
        item.version += 1
        item.evidence.append(
            EventTrace(
                at=now,
                kind="archive_attention_capacity",
                amount=0.0,
                note=f"未终结待关注事项超过 {bounded} 条容量，按证据强度归档",
                source="emotion_state_capacity",
            )
        )
        item.evidence = item.evidence[-30:]
        archived_ids.append(item.id)
    updated.state_version += 1
    updated.updated_at = now
    return updated, archived_ids


def apply_attention_observations(
    ledger: StateLedger,
    observations: list[AttentionObservation],
    *,
    limit: int = 6,
) -> tuple[StateLedger, list[str]]:
    updated = ledger
    reasons: list[str] = []
    for original in observations[: max(0, int(limit))]:
        observation = replace(original, expected_state_version=None)
        updated, applied, reason = apply_attention_observation(updated, observation)
        reasons.append(reason)
        if not applied and reason == "stale_item_version":
            continue
    return updated, reasons


def select_attention_items(
    items: list[AttentionItem],
    limit: int = 4,
    *,
    include_proposed: bool = False,
    now: datetime | None = None,
) -> list[AttentionItem]:
    current_time = now or utc_now()
    candidates = [
        item
        for item in items
        if is_open_attention(item)
        and (include_proposed or item.status == "open")
        and item.explicit
        and not is_short_lived_attention(item)
        and item.confidence >= (0.62 if item.status == "open" else 0.72)
    ]

    def evidence_timestamp(item: AttentionItem) -> float:
        try:
            return parse_time(item.last_evidence_at).timestamp()
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def score(item: AttentionItem) -> tuple[float, float, float, float, float, str]:
        overdue = 1.0 if is_attention_overdue(item, current_time) else 0.0
        open_status = 1.0 if item.status == "open" else 0.0
        explicit = 1.0 if item.explicit else 0.0
        return (
            -overdue,
            -open_status,
            -explicit,
            -item.confidence,
            -evidence_timestamp(item),
            item.id,
        )

    candidates.sort(key=score)
    return candidates[: max(0, int(limit))]
