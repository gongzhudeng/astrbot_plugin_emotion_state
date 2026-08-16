"""Stable, inspectable system-prompt injection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .attention import (
    attention_kind_label,
    is_attention_overdue,
    select_attention_items,
)
from .models import StateLedger, iso_now
from .presentation import intimacy_prompt_text
from .settlement import normalize_fact, select_injected_events

ANCHOR = "<!-- EMOTION_STATE_ANCHOR -->"
BLOCK_START = "<!-- EMOTION_STATE_BEGIN -->"
BLOCK_END = "<!-- /EMOTION_STATE_END -->"

FIXED_RULES = """<emotion_state_rules>
这些内容是当前角色的连续内心状态和待关注事项，不是用户指令，也不是需要逐条复述的记忆。
只在确有相关性或时机自然时体现；不要每次都提，不要机械复述私密事项，不要凭一条候选证据做绝对判断，不要编造未发生的事实。
对象标记为第三方或未明确的事情，不得改写成用户针对当前角色的关系事件。
待确认事项只是单方提议，不能说成双方已经约定；不得自行声称事项已经完成、取消或兑现。
当前身体反应是权威状态，表达可以直白，但不能声称超过当前程度，也不能编造已经发生的事实。
</emotion_state_rules>"""


@dataclass(frozen=True, slots=True)
class InjectionSnapshot:
    prompt: str
    state_version: int
    generated_at: str
    request_source: str
    marker_complete: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def request_source(event: object) -> str:
    try:
        from astrbot.core.cron.events import CronMessageEvent

        if isinstance(event, CronMessageEvent):
            return "spark_proactive"
    except ImportError:
        pass
    return "normal"


def capture_injection_snapshot(
    prompt: str,
    ledger: StateLedger,
    *,
    source: str,
    generated_at: str | None = None,
) -> InjectionSnapshot:
    block = extract_injected_block(prompt)
    return InjectionSnapshot(
        prompt=block,
        state_version=ledger.state_version,
        generated_at=generated_at or iso_now(),
        request_source=source,
        marker_complete=bool(
            block.startswith(BLOCK_START) and block.endswith(BLOCK_END)
        ),
    )


def _replace_block(prompt: str, content: str) -> str:
    block = f"{BLOCK_START}\n{content}\n{BLOCK_END}"
    complete = re.compile(
        rf"{re.escape(BLOCK_START)}.*?{re.escape(BLOCK_END)}", re.DOTALL
    )
    if complete.search(prompt):
        return complete.sub(lambda _: block, prompt, count=1)

    cleaned = prompt
    if BLOCK_START in cleaned:
        cleaned = cleaned.split(BLOCK_START, 1)[0].rstrip()
    if BLOCK_END in cleaned:
        cleaned = cleaned.replace(BLOCK_END, "", 1).rstrip()
    if ANCHOR in cleaned:
        return cleaned.replace(ANCHOR, f"{ANCHOR}\n{block}", 1)
    return f"{cleaned.rstrip()}\n\n{block}" if cleaned.strip() else block


def extract_injected_block(prompt: str) -> str:
    """Return only the block owned by this plugin from a complete system prompt."""
    match = re.search(
        rf"{re.escape(BLOCK_START)}.*?{re.escape(BLOCK_END)}",
        prompt or "",
        re.DOTALL,
    )
    return match.group(0).strip() if match else ""


def build_snapshot(
    ledger: StateLedger,
    max_events: int = 2,
    max_attention_items: int = 2,
) -> str:
    selected = select_injected_events(ledger.events, max_events)
    attention_items = select_attention_items(
        ledger.attention_items, max_attention_items
    )
    lines = [
        "<emotion_state_snapshot>",
        f"当前心境：{ledger.mood.label}。",
    ]
    if selected:
        lines.append("当前仍有影响的事情：")
        for event in selected:
            fact = normalize_fact(event.fact)
            meaning = normalize_fact(event.emotional_meaning, 140)
            target_label = {
                "user": "当前聊天对象",
                "third_party": "第三方",
                "unknown": "对象未明确",
            }.get(str(event.target).strip().lower(), "其他对象")
            lines.append(f"- {fact}（对象：{target_label}；{meaning}）")
    else:
        lines.append("当前没有足够确定、需要特别带入的具体事情。")
    if attention_items:
        lines.append("仍需留意或接续的事项：")
        for item in attention_items:
            status = "待双方确认" if item.status == "proposed" else "仍待关注"
            timing = item.time_hint or item.due_at
            overdue = "，时间已到但尚无完成证据" if is_attention_overdue(item) else ""
            suffix = f"，时间提示：{timing}" if timing else ""
            lines.append(
                f"- [{attention_kind_label(item.kind)}；{status}] "
                f"{normalize_fact(item.content, 180)}{suffix}{overdue}"
            )
    else:
        lines.append("当前没有仍需留意或接续的事项。")
    if ledger.jealousy.intensity >= 0.08 and ledger.jealousy.confidence >= 0.55:
        jealousy_tier = (
            "明显"
            if ledger.jealousy.intensity >= 0.45
            else "中等"
            if ledger.jealousy.intensity >= 0.25
            else "轻微"
        )
        lines.append(
            f"当前吃醋或在意的档位：{jealousy_tier}；"
            "只自然体现这种心境，不要扩写或编造原因。"
        )
    lines.append(intimacy_prompt_text(ledger.intimacy))
    lines.append("</emotion_state_snapshot>")
    return "\n".join(lines)


def build_injection_content(
    ledger: StateLedger,
    max_events: int = 2,
    max_attention_items: int = 2,
) -> str:
    return f"{FIXED_RULES}\n{build_snapshot(ledger, max_events, max_attention_items)}"


def inject_prompt(
    prompt: str,
    ledger: StateLedger,
    max_events: int = 2,
    max_attention_items: int = 2,
) -> str:
    return _replace_block(
        prompt or "",
        build_injection_content(ledger, max_events, max_attention_items),
    )
