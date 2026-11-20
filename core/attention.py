"""Deterministic lifecycle transitions for non-emotional attention items."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import datetime, timedelta

from .models import (
    AttentionItem,
    AttentionObservation,
    EventTrace,
    StateLedger,
    iso_now,
    normalize_due_at,
    parse_time,
    utc_now,
)

_TERMINAL_STATUSES = {"completed", "cancelled", "superseded", "archived"}
# v0.3.25 premature-completion guard: if an item's due_at is still more than
# this many hours away, a model-suggested ``complete`` cannot be trusted
# ("晚上拍视频" closed at 13:21, "晚饭" closed at 15:04). Code-level gate,
# independent of prompt discipline; 0 disables the guard entirely.
#
# v0.3.30: the gate now also requires the item to name a *period* that has not
# arrived yet. The old check looked only at the remaining time, so anything
# whose due_at had been defaulted to "today 23:59" (takeout, a meal, a quick
# clip) was refused all afternoon and could only be closed after 21:00 —
# a hotpot that was eaten at 12:30 stayed open until 22:02 and the character
# kept asking why the takeout had not been ordered. Periodless items are now
# released; genuine "tonight/tomorrow" promises are still blocked.
_PREMATURE_COMPLETE_GUARD_HOURS = 3.0
# 只认"用户点明了时点"的词。故意不含「中午」：外卖类事项常写「今天中午点」，
# 那是当天的事，不该拦到晚上。
_EXPLICIT_PERIOD_CUES = re.compile(
    r"下午|傍晚|晚上|今晚|今夜|夜里|睡前|凌晨|"
    r"明晚|明早|明天|后天|大后天|"
    # 「周X」和「星期X」都要认：2026-11-06 的「星期天提醒充电」因为只写了周X，
    # 守门认不出这是将来时段，建项 21 分钟就被「我记着呢」关掉了。
    r"周[一二三四五六日天]|星期[一二三四五六日天]|下周|下个月"
)
# Sources trusted to mutate attention items. ``local_catch`` is the zero-cost
# catch for explicit reminder wording and never depends on a model call.
# ``webui`` is a human editing the dashboard by hand: highest trust.
# (``local_rule`` stays untrusted: it is the legacy keyword engine.)
_ATTENTION_REVIEW_SOURCES = {
    "livingmemory_summary",
    "attention_review",
    "local_catch",
    "webui",
}
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
    "care": "状态",
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


# care（用户当下的身体/情绪状态）是短时事项：默认 24 小时后自然过期，
# 免得"头晕"这种一次性的状态变成永久挂起。用户说好了会提前 complete。
_CARE_DEFAULT_HOURS = 24.0


def default_care_due(now: datetime | None = None) -> str:
    moment = (now or utc_now()) + timedelta(hours=_CARE_DEFAULT_HOURS)
    return moment.isoformat()


def is_open_attention(item: AttentionItem) -> bool:
    return item.status not in _TERMINAL_STATUSES


def is_attention_overdue(item: AttentionItem, now: datetime | None = None) -> bool:
    if not is_open_attention(item):
        return False
    if _ONGOING_ATTENTION_CUES.search(f"{item.time_hint} {item.content}"):
        return False
    # Textual hints (明天/今晚/周X…) resolve to a concrete due time so one-off
    # items can actually expire instead of lingering forever.
    due_text = item.due_at or resolve_due_at(item.time_hint, now)
    if not due_text:
        return False
    try:
        return parse_time(due_text) < (now or utc_now())
    except (TypeError, ValueError):
        return False


_CN_DIGITS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_WEEKDAY_NAMES = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}


def resolve_due_at(time_hint: str, now: datetime | None = None) -> str:
    """Translate a textual time hint into a concrete due timestamp.

    Supports 今晚/今天/晚上 (today), 明天/明晚 (+1 day), 后天/大后天, 周X/星期X,
    and N 天/小时 (后|之内|内). Returns "" when nothing conclusive can be parsed
    (e.g. 以后/下次/有空), leaving the item to evidence-based cleanup instead.
    """
    anchor = (now or utc_now()).astimezone()
    text = str(time_hint or "").strip()

    # 凌晨语感：0:00-4:59 还没睡的人说"明天"，通常指当天白天（心理上的
    # "今天"仍是日历上的今天，而不是已经翻页的日期）。
    dawn_shift = 1 if 0 <= anchor.hour < 5 else 0

    def due(end_of_day: bool, days: int = 0, hours: float = 0.0) -> str:
        moment = anchor + timedelta(days=days, hours=hours)
        if end_of_day:
            moment = moment.replace(hour=23, minute=59, second=0, microsecond=0)
        return moment.isoformat()

    day_offsets = (("大后天", 3), ("后天", 2), ("明天", 1), ("明晚", 1), ("明早", 1))
    for cue, offset in day_offsets:
        if cue in text:
            return due(True, days=max(offset - dawn_shift, 0))
    if re.search(r"今晚|今夜|今天|当晚", text):
        return due(True)
    if (
        re.search(r"(?<!明)(?<!大)晚上|夜里|睡前", text)
        and "明天" not in text
        and "后天" not in text
    ):
        return due(True)
    weekday = re.search(r"(?:周|星期|礼拜)\s*([一二三四五六日天])", text)
    if weekday:
        target = _WEEKDAY_NAMES[weekday.group(1)]
        offset = (target - anchor.weekday()) % 7
        if offset == 0:
            offset = 7
        return due(True, days=offset)
    days_match = re.search(
        r"([一两二三四五六七八九十\d]+)\s*天(?:之后|以后|后|之内|内)?", text
    )
    if days_match:
        raw = days_match.group(1)
        count = int(raw) if raw.isdigit() else _CN_DIGITS.get(raw, 0)
        if count > 0:
            return due(True, days=count)
    hours_match = re.search(
        r"([一两二三四五六七八九十\d]+|半)\s*(?:个)?小时(?:之后|以后|后|之内|内)?", text
    )
    if hours_match:
        raw = hours_match.group(1)
        hours = (
            0.5
            if raw == "半"
            else (int(raw) if raw.isdigit() else _CN_DIGITS.get(raw, 0))
        )
        if hours > 0:
            return due(False, hours=hours)
    return ""


def format_attention_timing(item: AttentionItem, now: datetime | None = None) -> str:
    """Display word for injection/rendering, derived from the absolute due time.

    Stored time_hint words ("明天") are frozen at creation time; deriving the
    word from due_at at display time keeps "明天" turning into "今天" once the
    day actually arrives. Falls back to time_hint when no due time exists.

    v0.3.27: relative words travel together with the absolute date
    ("今天（10月12日）") — chat transcripts mix several "明天"s that point at
    different days, so the date pins the meaning down even when surrounding
    text is ambiguous.
    """
    if _ONGOING_ATTENTION_CUES.search(f"{item.time_hint} {item.content}"):
        return item.time_hint  # 进行性提示（每晚/每天…）原样保留
    # v0.3.33: care 的 due_at 是"什么时候过期"，不是"约定什么时候做"。
    # 按剩余天数格式化成"明天"会让模型以为明天才需要关心，这里直接说进行中；
    # 真过期了仍走下面的原逻辑（"已过期（原定X月X日）"）。
    if item.kind == "care" and not is_attention_overdue(item, now):
        return "进行中"
    due_text = item.due_at or resolve_due_at(item.time_hint, now)
    if not due_text:
        return item.time_hint
    try:
        due_dt = parse_time(due_text).astimezone()
    except (TypeError, ValueError):
        return item.time_hint
    anchor = (now or utc_now()).astimezone()
    days = (due_dt.date() - anchor.date()).days
    if days < 0:
        return f"已过期（原定{due_dt.month}月{due_dt.day}日）"
    date_suffix = f"（{due_dt.month}月{due_dt.day}日）"
    if days == 0:
        return f"今天{date_suffix}"
    if days == 1:
        return f"明天{date_suffix}"
    if days == 2:
        return f"后天{date_suffix}"
    return f"{due_dt.month}月{due_dt.day}日"


def archive_expired_attention_items(
    ledger: StateLedger,
    now: datetime | None = None,
    *,
    grace_days: float = 1.0,
) -> tuple[StateLedger, list[str]]:
    """Archive one-off items after an explicit deadline without claiming success.

    ``grace_days`` keeps an overdue item alive for a short window so the model
    reviewer can still read the chat and settle it as completed instead of the
    janitor silently archiving it.
    """
    current_time = now or utc_now()
    grace = max(0.0, float(grace_days))
    cutoff = current_time - timedelta(days=grace)
    updated = _copy_ledger(ledger)
    timestamp = current_time.isoformat()
    archived_ids: list[str] = []
    for item in updated.attention_items:
        if not is_attention_overdue(item, cutoff):
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


def archive_stale_attention_items(
    ledger: StateLedger,
    *,
    max_days: float = 3.0,
    now: datetime | None = None,
) -> tuple[StateLedger, list[str]]:
    """Archive one-off open items with no new evidence for ``max_days``.

    Long-term items are exempt: anything without an explicit ``due_at`` (an open
    project, "我在弄一个插件", "以后每次…") stays until a reviewer settles it.
    Cues like 以后/每次/每天/长期 also exempt. This is a janitor, not a
    completion claim: the archive note explicitly says it is not success.
    """
    bounded = float(max_days)
    updated = _copy_ledger(ledger)
    if bounded <= 0:
        return updated, []
    current_time = now or utc_now()
    timestamp = current_time.isoformat()
    archived_ids: list[str] = []
    for item in updated.attention_items:
        if not is_open_attention(item) or is_attention_overdue(item, current_time):
            continue
        # No explicit deadline means the user never said when it ends: keep it
        # until the model sees evidence that it is done or cancelled.
        if not item.due_at:
            continue
        if _ONGOING_ATTENTION_CUES.search(f"{item.time_hint} {item.content}"):
            continue
        anchor_text = item.last_evidence_at or item.created_at
        try:
            anchor = parse_time(anchor_text)
        except (TypeError, ValueError, OverflowError):
            continue
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=utc_now().tzinfo)
        quiet_days = (current_time - anchor).total_seconds() / 86400.0
        if quiet_days < bounded:
            continue
        item.status = "archived"
        item.archived_at = timestamp
        item.updated_at = timestamp
        item.version += 1
        item.evidence.append(
            EventTrace(
                at=timestamp,
                kind="archive_attention_stale",
                amount=0.0,
                note=f"超过 {bounded:.0f} 天没有任何新证据，自动归档（不视为完成）",
                source="emotion_state_stale",
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
    if observation.source not in _ATTENTION_REVIEW_SOURCES:
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
            if observation.confidence < 0.70:
                return "insufficient_attention_confidence"
        if observation.source != "webui":
            # The heuristics below guard against model hallucinations. A human
            # typing in the dashboard is the authority, so they are skipped:
            # back-filling an item whose deadline already passed is legitimate.
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


def _names_pending_period(item: AttentionItem) -> bool:
    """Whether the item itself points at a period that has not arrived yet.

    Only these items need the premature-completion gate: their deadline exists
    because someone said "晚上" / "明天" / "周X", not because ``due_at`` was
    filled with an end-of-day default. Items phrased "今天/待会/现在" are
    ordinary same-day chores — once the chat shows them done, they are done.
    """
    text = f"{getattr(item, 'time_hint', '')} {getattr(item, 'content', '')}"
    return bool(_EXPLICIT_PERIOD_CUES.search(text))


def _premature_complete_rejection(
    item: AttentionItem,
    guard_hours: float,
    *,
    evidence_speaker: str = "",
) -> str | None:
    """Reject completes whose promised window has not even started.

    An item due tonight has no trustworthy completion evidence in the
    afternoon: the reviewer can only see role-play phrasing, and every
    mis-close so far happened hours before due_at. Items without a
    parsable due_at, and due dates already in the past, are let through.

    v0.3.30: an item whose wording names no period ("今天点外卖", "待会拍一段")
    is released even when due_at is still hours away — its due_at is only the
    end-of-day default, not a promise about when the action may happen.

    v0.3.44: 用户本人说的话不受这条约束。这道门防的是「她自己答应了就当做到」
    （v0.3.36 那次「我记着呢」），不防用户。用户周六说「星期天那个我已经充了、
    不用提醒」，那是真的提前完成，拦住它只会让提醒照常触发。
    """
    try:
        guard = float(guard_hours)
    except (TypeError, ValueError):
        guard = _PREMATURE_COMPLETE_GUARD_HOURS
    if guard <= 0:
        return None
    due_raw = str(getattr(item, "due_at", "") or "").strip()
    if not due_raw:
        return None
    try:
        due = parse_time(due_raw)
    except (TypeError, ValueError):
        return None
    remaining = due - utc_now()
    if remaining <= timedelta(0):
        return None
    if remaining > timedelta(hours=guard):
        if not _names_pending_period(item):
            return None
        if str(evidence_speaker or "").strip().lower() == "user":
            return None
        return "premature_completion_before_due"
    return None


def reconcile_due_at(time_hint: str, model_due: str, now: datetime | None = None) -> str:
    """用本地时间解析校准模型给的绝对日期。

    模型算日期是弱项：2026-11-06（周五）的「后天星期天」，它给出 11-09（周一）。
    prompt 里明写了「当前日期时间：2026-11-06 20:18（周五）」，它照样错一天。
    原因是日期推演要在 token 之间做进位，模型没有日历概念。

    而 ``resolve_due_at`` 用真实日历算，是准的。所以分工：
    **日期信代码，钟点信模型**。

    - ``time_hint`` 解析不出 → 保留模型给的（可能是「11月8日下午三点」这种绝对表述）；
    - 两者日期一致 → 保留模型的（它可能给了更精确的钟点）；
    - 日期不一致 → 用代码算的日期，保留模型的时分。
    """
    text = str(time_hint or "").strip()
    candidate = str(model_due or "").strip()
    if not text:
        return candidate
    try:
        resolved = resolve_due_at(text, now)
    except (TypeError, ValueError):
        return candidate
    if not resolved:
        return candidate
    try:
        resolved_moment = parse_time(resolved)
    except (TypeError, ValueError):
        return candidate
    if not candidate:
        return resolved
    try:
        model_moment = parse_time(candidate)
    except (TypeError, ValueError):
        return resolved
    if resolved_moment.date() == model_moment.date():
        return candidate
    corrected = model_moment.replace(
        year=resolved_moment.year,
        month=resolved_moment.month,
        day=resolved_moment.day,
    )
    return corrected.isoformat()


def apply_attention_observation(
    ledger: StateLedger,
    observation: AttentionObservation,
    *,
    premature_complete_guard_hours: float | None = None,
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
    if observation.action == "complete" and current is not None:
        if not completion_evidence_matches_item(current, observation.evidence_quote):
            return ledger, False, "unrelated_completion_evidence"
        # v0.3.36: 「提醒型」事项（remember）的完成证据必须来自用户。
        # 她自己的「知道啦，我记着呢，到点提醒你」只能证明**答应**了，
        # 不能证明提醒已经发生——2026-11-06 的「星期天提醒充电」就是被这句话
        # 在建项 21 分钟后关掉的，用户看到的现象是「说了两遍记得也没建」。
        if observation.source != "webui" and current.kind == "remember":
            speaker = str(observation.evidence_speaker or "").strip().lower()
            if speaker == "assistant":
                return ledger, False, "reminder_acknowledged_not_delivered"
        # v0.3.25: code-level gate — an item whose due_at is still hours away
        # has no trustworthy completion evidence yet; model-suggested closes
        # get skipped until the window is near. Human webui edits stay exempt.
        if observation.source != "webui":
            guard = (
                _PREMATURE_COMPLETE_GUARD_HOURS
                if premature_complete_guard_hours is None
                else premature_complete_guard_hours
            )
            rejection = _premature_complete_rejection(
                current,
                guard,
                evidence_speaker=observation.evidence_speaker,
            )
            if rejection:
                return ledger, False, rejection
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
            due_at=reconcile_due_at(observation.time_hint, observation.due_at),
            confidence=observation.confidence,
            explicit=observation.explicit,
            source=observation.source,
            evidence_speaker=observation.evidence_speaker,
            fingerprint=observation.fingerprint
            or attention_fingerprint(observation.content),
            evidence=[_trace(observation, "created", now)],
        )
        # care 是一次性的状态关注：没写截止时间就默认 24 小时后过期，
        # 免得"头晕"这类事项一直挂在清单里（无 due 的条目不会被自动归档）。
        if item.kind == "care" and not item.due_at:
            item.due_at = default_care_due()
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
            # 改时间也要过同一道校准：模型在 update 里同样会把日期算错，
            # 只在 create 校验等于留了一半的洞。
            current.due_at = normalize_due_at(
                reconcile_due_at(
                    current.time_hint,
                    observation.due_at or current.due_at,
                )
                or resolve_due_at(current.time_hint)
            )
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
            # Open-ended items (no explicit deadline) are kept first: they are
            # the long-running things the user actually wants remembered.
            1 if not item.due_at else 0,
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
    user_message: str = "",
    recent_messages: list[str] | None = None,
) -> list[AttentionItem]:
    """Pick the items injected into the reply prompt.

    v0.3.32: ``care`` items (the user's current physical / emotional state)
    share the same ``limit`` slots as everything else, but rank above ordinary
    items — "他说头晕" deserves attention before "周末看电影". Already-overdue
    items still rank first: a promise you are late for must not be pushed out
    by a fresh care item.

    v0.3.35: 加入注入时机（``should_inject``）。有明确时间的事项只在临近窗口内出现，
    除非他主动提起、或已逾期。没有明确时间的事项保持现状。
    """
    current_time = now or utc_now()
    candidates = [
        item
        for item in items
        if is_open_attention(item)
        and (include_proposed or item.status == "open")
        and item.explicit
        and not is_short_lived_attention(item)
        and item.confidence >= (0.62 if item.status == "open" else 0.72)
        and should_inject(item, current_time, user_message, recent_messages)
    ]

    def evidence_timestamp(item: AttentionItem) -> float:
        try:
            return parse_time(item.last_evidence_at).timestamp()
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def score(item: AttentionItem) -> tuple[float, float, float, float, float, str]:
        overdue = is_attention_overdue(item, current_time)
        # 0 = 已逾期，1 = care（用户当下的状态），2 = 其他
        tier = 0.0 if overdue else (1.0 if item.kind == "care" else 2.0)
        open_status = 1.0 if item.status == "open" else 0.0
        explicit = 1.0 if item.explicit else 0.0
        return (
            tier,
            -open_status,
            -explicit,
            -item.confidence,
            -evidence_timestamp(item),
            item.id,
        )

    candidates.sort(key=score)
    return candidates[: max(0, int(limit))]


def attention_exclusion_reason(
    item: AttentionItem,
    *,
    limit_reached: bool = False,
    now: datetime | None = None,
    user_message: str = "",
    recent_messages: list[str] | None = None,
) -> str:
    """这条为什么没进注入。空串 = 它本来就该进。

    只给面板看，不参与任何判定——判定在 ``select_attention_items`` 里，
    两处各写一套迟早会漂移。这里按同一批判据，只是把「被哪一条挡了」说出来。

    存在的理由：面板上半部说「没有待关注事项被注入」，下半部却列着一条，
    用户看到的是自相矛盾。真正的原因（待双方确认）不写出来，就只能猜是 bug。
    """
    if not is_open_attention(item):
        return "已终结"
    if item.status != "open":
        return "待双方确认，不算约定"
    if not item.explicit:
        return "不是明说的约定"
    if is_short_lived_attention(item):
        return "短期状态，不进注入"
    threshold = 0.62 if item.status == "open" else 0.72
    if item.confidence < threshold:
        return "证据不足"
    # 有明确时间、但此刻不在窗口内：不是「不合格」，是「还不到时候」。
    # 刚被主动提醒过也走这里，但原因不同，别混成一句「还未到时间」。
    if not should_inject(item, now, user_message, recent_messages):
        if _reminder_silence_active(item, now):
            return "刚提醒过，暂时不重复"
        if item.due_at:
            return "还未到时间"
    if limit_reached:
        return "被条数上限挤掉"
    return ""


# ---------------------------------------------------------------------------
# 注入时机（v0.3.35）：不是所有事项都该一直在上下文里
#
# 规则只有一句：**该出现的时候出现，不该出现的时候别出声**。
#   1. care（他此刻不舒服/心情不好）——一直注入，它不是「将来的事」；
#   2. 已逾期未完成——继续注入，这时候才最该催；
#   3. 他主动提起了（关键词/语义命中）——立即注入，不然会变成失忆；
#   4. 有明确时间——只在临近窗口内注入，其他时候安静躺在后台；
#   5. 没有明确时间——保持现状（一直注入），先不动它。
# ---------------------------------------------------------------------------

#: 不同事项的提前量（秒）。约定比计划紧，提醒最紧。
INJECTION_LEAD_SECONDS = {
    "commitment": 2 * 3600,
    "plan": 4 * 3600,
    "remember": 4 * 3600,
    "follow_up": 0,  # 只有日期时当天 0 点就出现
    "care": 0,
}

#: 主动提醒之后的静默期（秒）：这段时间里这条不再挂进上下文，
#: 免得她刚说完「该充电了」，下一轮又提一次。他主动问起时不受限。
_REMINDER_SILENCE_SECONDS = 30 * 60

#: 从内容里抽关键词时要丢掉的虚词/动词/时间词。抽出来的名词才适合当「提到没」的判据。
_MENTION_STOPWORDS = frozenset({
    "我们", "你们", "他们", "她们", "自己", "这个", "那个", "这些", "那些",
    "什么", "怎么", "为什么", "是不是", "对不对", "好不好", "要不要", "能不能",
    "一下", "一些", "一点", "一个", "两个", "几个", "很多", "好多",
    "时候", "地方", "东西", "事情", "问题", "情况",
    "明天", "后天", "今天", "昨天", "上午", "下午", "晚上", "中午", "早上", "凌晨",
    "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期天", "周日", "周末",
    "周一", "周二", "周三", "周四", "周五", "周六",
    "记得", "忘记", "提醒", "告诉", "知道", "觉得", "认为", "希望", "想要",
    "可以", "应该", "必须", "需要", "能够", "可能", "也许", "大概",
    "然后", "之后", "之前", "现在", "刚才", "马上", "立刻", "待会", "一会",
})

#: 两个字的纯动词（"说要""要下"）不该当关键词。只过滤两字组合，长词不受影响。
_MENTION_VERB_CHARS = "说要去做给看买吃玩聊拍发写读听唱跳跑走坐站躺睡醒来回出进上下开关打扫洗穿脱戴摘拿放收拾整理清洁擦拖"

#: 当分隔符用的单字：动词/介词/虚词。它们不该粘在名词上
#: （"给电动车充电" 里的「给」、「下盗版」里的「下」）。
#: 注意「还」不在里面——它是「还书」这种实词的一部分，当分隔会把词切坏；
#: 它在「还想买」里做副词时，双向匹配仍能命中「买猫粮」。
_MENTION_SEPARATOR_CHARS = set("说要去做给看买吃玩聊拍发写读听唱跳跑走坐站躺睡醒来回出进上下开关就都把被和跟与在是的了吗吧呢啊呀哦嗯")


def extract_mention_keywords(content: str) -> list[str]:
    """从事项内容里抽「能当话题锚点」的词块。

    做法是**先切再取**：把停用词、人称、单字动词都当分隔符替换成空格，
    剩下连着的才是名词块（"电动车充电""刃零的盗版"里的实体）。

    抽出来的是**词块**，不保证是标准词——所以匹配时必须双向：
    「电动车充电」这个块能匹配用户说的「充电」，靠的是 ``_mention_matches``
    里的片段回查，不是指望这里切得多准。中文不做分词切不准，硬切反而更糟。
    """
    text = re.sub(r"[，。、；：！？…\s（）()「」“”‘’\",.!?;:\-—~～]", " ", str(content or ""))
    # 相对时间整体去掉（「三天后」「两个小时后」「下下周」），它们不是话题锚点
    text = re.sub(
        r"(?:下+个?)?(?:[0-9一二三四五六七八九十两几]+(?:天|周|星期|个月|小时|分钟))(?:后|前|之后|之前)?",
        " ",
        text,
    )
    # 人称与机器人称呼
    text = re.sub(r"(Mando|mando|小怡|我们|你们|他们|她们|自己)", " ", text)
    # 长的停用词先替换（避免「星期天」被拆成「星期」+「天」）
    for word in sorted(_MENTION_STOPWORDS, key=len, reverse=True):
        text = text.replace(word, " ")
    # 单字动词/虚词当分隔
    text = "".join(" " if ch in _MENTION_SEPARATOR_CHARS else ch for ch in text)
    words = [
        match.group()
        for match in re.finditer(r"[\u4e00-\u9fff]{2,6}", text)
        if match.group() not in _MENTION_STOPWORDS
    ]
    return list(dict.fromkeys(words))


def _mention_matches(keyword: str, text: str) -> bool:
    """词块与一句话是否算「谈到同一件事」。

    双向：整块出现在句子里，或句子里 2 字以上的片段落在词块里。
    后者是为了让「电动车充电」被「充电」命中——只做单向匹配的话，
    长词块几乎永远匹配不上。
    """
    if not keyword or not text:
        return False
    if keyword in text:
        return True
    for size in (4, 3, 2):
        if len(text) < size:
            continue
        for index in range(len(text) - size + 1):
            fragment = text[index : index + size]
            if fragment.isascii():
                continue
            if fragment in keyword:
                return True
    return False


def is_mentioned(
    item: AttentionItem,
    user_message: str = "",
    recent_messages: list[str] | None = None,
) -> bool:
    """他这句话（或最近几句）是不是在谈这件事。

    只用当前这一句会漏：他说「对了」下一句才说「充电」。所以把最近几条
    用户消息一起拿来匹配——更接近人「想起」的过程。
    """
    keywords = extract_mention_keywords(item.content)
    if not keywords:
        return False
    texts = [str(user_message or "")] + [str(t) for t in (recent_messages or [])]
    for text in texts:
        if not text:
            continue
        for kw in keywords:
            if _mention_matches(kw, text):
                return True
    return False


def _injection_lead_seconds(kind: str) -> int:
    return int(INJECTION_LEAD_SECONDS.get(kind, 2 * 3600))


def _reminder_silence_active(item: AttentionItem, now: datetime | None = None) -> bool:
    """刚被主动提醒过、还在静默期内。

    判定与面板文案共用这一个函数：两处各写一套，迟早会漂移成
    「面板说还没到时间，实际是刚提醒过」。
    """
    stamp = str(getattr(item, "notified_at", "") or "").strip()
    if not stamp:
        return False
    current = now or utc_now()
    try:
        notified = parse_time(stamp)
    except (TypeError, ValueError):
        return False
    if notified.tzinfo is None:
        notified = notified.replace(tzinfo=current.tzinfo)
    return (current - notified).total_seconds() < _REMINDER_SILENCE_SECONDS


def should_inject(
    item: AttentionItem,
    now: datetime | None = None,
    user_message: str = "",
    recent_messages: list[str] | None = None,
) -> bool:
    """这条此刻该不该进上下文。

    判据顺序有讲究：care 与逾期是「无条件」，提及是「例外放行」，
    窗口是「默认规则」，无时间事项是「保持现状」。
    """
    current = now or utc_now()
    # 1. care 不是「将来的事」，一直注入
    if item.kind == "care":
        return True
    # 2. 他主动提起了：立即注入，不然就是失忆
    if is_mentioned(item, user_message, recent_messages):
        return True
    # 3. 刚被主动提醒过：短时间内别再挂进上下文，否则她刚说完又会说一遍
    if _reminder_silence_active(item, current):
        return False
    # 4. 逾期未完成：这时候最该催，不受窗口限制
    if is_attention_overdue(item, current):
        return True
    # 5. 有明确时间：只在窗口内出现
    if item.due_at:
        try:
            due = parse_time(item.due_at)
        except (TypeError, ValueError):
            return True  # 解析不出来就当没有约束，宁可多给也别漏
        if due.tzinfo is None:
            due = due.replace(tzinfo=current.tzinfo)
        # 00:00 与 23:59 都是「只有日期」的兜底写法（建项时没给钟点），
        # 按整天算——「今晚」这种事不能等到 21:59 才出现。
        if (due.hour == 0 and due.minute == 0) or (
            due.hour == 23 and due.minute >= 59
        ):
            return current.date() >= due.date()
        lead = _injection_lead_seconds(item.kind)
        return (due - current).total_seconds() <= lead
    # 6. 无明确时间：保持现状（一直注入）
    return True


# ---------------------------------------------------------------------------
# 主动提醒（v0.3.42）：到点了就开口，而不是等他说话才带一句
#
# 与注入的关系：注入回答「这轮聊天要不要提到它」，提醒回答「要不要主动开口」。
# 两条路共用 notified_at：提醒过后 30 分钟内不再注入（should_inject 第 3 条）。
# ---------------------------------------------------------------------------


def due_reminder_candidates(
    items: list[AttentionItem],
    now: datetime | None = None,
    *,
    kinds: tuple[str, ...] = ("remember",),
    max_count: int = 2,
    interval_minutes: float = 30.0,
) -> list[AttentionItem]:
    """到点、该提醒、且还没提够次数的事项，按到期时间升序。

    刻意比 `select_attention_items` 更保守：只收**已到点**的（不是临近窗口），
    因为主动开口的打扰成本比上下文里多一行高得多。
    """
    current = now or utc_now()
    allowed = tuple(str(k) for k in kinds if str(k))
    limit_count = max(1, int(max_count))
    interval = max(0.0, float(interval_minutes)) * 60.0
    picked: list[AttentionItem] = []
    for item in items:
        if item.status != "open" or not item.explicit:
            continue
        if item.kind not in allowed or item.kind == "care":
            continue
        if not item.due_at or is_short_lived_attention(item):
            continue
        if item.confidence < 0.62:
            continue
        try:
            due = parse_time(item.due_at)
        except (TypeError, ValueError):
            continue
        if due.tzinfo is None:
            due = due.replace(tzinfo=current.tzinfo)
        if current < due:
            continue
        if item.notify_count >= limit_count:
            continue
        if item.notified_at:
            try:
                last = parse_time(item.notified_at)
            except (TypeError, ValueError):
                last = None
            if last is not None:
                if last.tzinfo is None:
                    last = last.replace(tzinfo=current.tzinfo)
                if (current - last).total_seconds() < interval:
                    continue
        picked.append(item)
    return sorted(picked, key=lambda row: row.due_at)


def mark_reminded(
    ledger: StateLedger, item_id: str, now: datetime | None = None
) -> tuple[StateLedger, bool]:
    """在事项上记一次主动提醒。返回 (新账本, 是否真的改了)。

    调用方应该在**发出消息之前**调用它：先占位再发，避免两个 tick 同时命中同一条。
    发失败就把账本丢掉不用回滚——少提一次可以接受，重复提才烦人。
    """
    target = str(item_id or "").strip()
    if not target:
        return ledger, False
    updated = _copy_ledger(ledger)
    for item in updated.attention_items:
        if item.id != target:
            continue
        stamp = (now or utc_now()).isoformat()
        item.notified_at = stamp
        item.notify_count = max(0, int(item.notify_count)) + 1
        item.updated_at = stamp
        # 刻意不动 version：它是对外的乐观锁版本，复核拿着旧版本号回来时
        # 会因为「版本已过期」被拒。提醒只是内部记账，不该让别人的操作失败。
        return updated, True
    return ledger, False
