"""Conservative local recognition for future-relevant private-chat items."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .attention import is_open_attention, normalize_attention_content
from .models import AttentionItem, AttentionObservation
from .settlement import strip_media_context

_FUTURE_CUES = re.compile(
    r"等一下|一会儿|待会儿|稍后|明天|后天|今晚|下次|以后|回头|到时候|"
    r"周[一二三四五六日天]|下周|下个月|过几天"
)
_COMMITMENT_CUES = re.compile(
    r"说好|约定|答应|承诺|输了?(?:的人)?(?:就|要|得)?|赢了?(?:的人)?(?:就|要|得)?|"
    r"到时候(?:要|得)|别反悔|不许反悔"
)
_PLAN_CUES = re.compile(
    r"(?:(?:咱们|我们|你|我).{0,12})?"
    r"(?:玩|做|聊|看|去|来|继续|试|发|拍|给|告诉|提醒|准备)"
)
_REMEMBER_CUES = re.compile(r"一定要记住|务必记住|给我记住|你要记住|别忘了|记得(?:要)?")
_CONFIRM_ONLY = re.compile(r"^(?:你)?(?:一定要|务必)?记住(?:这件事|这个|它)?[。！! ]*$")
_ACK_ONLY = re.compile(r"^(?:好的?[，, ]*)?(?:我)?(?:已经)?记住了|^(?:我)?会记住")
_COMPLETE_CUES = re.compile(
    r"已经.{0,18}(?:了|完成)|做完了|完成了|搞定了|拍了|发了|给了"
)
_CANCEL_CUES = re.compile(
    r"取消(?:吧|了)?|算了吧?|不(?:用|要|做|玩|拍|发).{0,10}了|作废"
)


@dataclass(slots=True)
class AttentionRuleRun:
    observations: list[AttentionObservation] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)


def _time_hint(text: str) -> str:
    match = _FUTURE_CUES.search(text)
    return match.group(0) if match else ""


def _due_at(text: str, now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    days = 2 if "后天" in text else 1 if "明天" in text else 0
    if not days:
        return ""
    target = (current + timedelta(days=days)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    return target.isoformat()


def _best_existing(text: str, items: list[AttentionItem]) -> AttentionItem | None:
    open_items = [item for item in items if is_open_attention(item)]
    if not open_items:
        return None
    if len(open_items) == 1:
        return open_items[0]
    chars = set(re.findall(r"[\w\u4e00-\u9fff]", text))
    ranked = sorted(
        open_items,
        key=lambda item: len(
            chars & set(re.findall(r"[\w\u4e00-\u9fff]", item.content))
        ),
        reverse=True,
    )
    overlap = len(chars & set(re.findall(r"[\w\u4e00-\u9fff]", ranked[0].content)))
    return ranked[0] if overlap >= 2 else None


class AttentionRuleEngine:
    def run(
        self,
        text: str,
        *,
        speaker: str,
        existing_items: list[AttentionItem] | None = None,
        now: datetime | None = None,
    ) -> AttentionRuleRun:
        result = AttentionRuleRun()
        clean = normalize_attention_content(strip_media_context(text))
        if not clean:
            result.exclusions.append("没有可识别的正文")
            return result
        if clean.lstrip().startswith(("/", "！", "!")) or "```" in clean:
            result.exclusions.append("命令或代码文本")
            return result

        existing = existing_items or []
        related = _best_existing(clean, existing)

        if _ACK_ONLY.search(clean):
            result.exclusions.append("记忆确认话术不代表事项完成")
            return result

        if _CANCEL_CUES.search(clean) and related:
            result.observations.append(
                AttentionObservation(
                    action="cancel",
                    item_id=related.id,
                    item_version=related.version,
                    confidence=0.9,
                    source=f"local_{speaker}",
                    evidence_quote=clean,
                )
            )
            return result

        if _COMPLETE_CUES.search(clean) and related:
            result.observations.append(
                AttentionObservation(
                    action="complete",
                    item_id=related.id,
                    item_version=related.version,
                    confidence=0.88,
                    source=f"local_{speaker}",
                    evidence_quote=clean,
                )
            )
            return result

        if _CONFIRM_ONLY.fullmatch(clean) and related:
            result.observations.append(
                AttentionObservation(
                    action="confirm",
                    item_id=related.id,
                    item_version=related.version,
                    confidence=0.96,
                    source=f"local_{speaker}",
                    evidence_quote=clean,
                )
            )
            return result

        future = _time_hint(clean)
        remember = _REMEMBER_CUES.search(clean)
        commitment = _COMMITMENT_CUES.search(clean)
        plan = future and _PLAN_CUES.search(clean)

        if remember and related and len(clean) <= 28:
            result.observations.append(
                AttentionObservation(
                    action="confirm",
                    item_id=related.id,
                    item_version=related.version,
                    confidence=0.94,
                    source=f"local_{speaker}",
                    evidence_quote=clean,
                )
            )
            return result

        if not (remember or commitment or plan):
            result.exclusions.append("没有明确的未来计划、约定或记忆要求")
            return result

        if commitment:
            kind = "commitment"
        elif remember:
            kind = "remember"
        else:
            kind = "plan"
        status = "proposed" if speaker == "assistant" else "open"
        result.observations.append(
            AttentionObservation(
                action="create",
                content=clean,
                kind=kind,
                status=status,
                actor="both"
                if re.search(r"咱们|我们|输的人|赢的人", clean)
                else speaker,
                time_hint=future,
                due_at=_due_at(clean, now),
                confidence=0.9 if remember or commitment else 0.82,
                explicit=True,
                source=f"local_{speaker}",
                evidence_quote=clean,
            )
        )
        return result
