"""Stable, inspectable system-prompt injection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime

from .attention import (
    attention_kind_label,
    is_attention_overdue,
    select_attention_items,
)
from .guidance import format_guidance_block, needs_guidance
from .models import StateLedger, iso_now
from .presentation import intimacy_prompt_text
from .settlement import (
    mood_narrative,
    normalize_fact,
    select_injected_events,
)
from .text_limits import EVENT_FACT_INJECTION_CHARS

ANCHOR = "<!-- EMOTION_STATE_ANCHOR -->"
BLOCK_START = "<!-- EMOTION_STATE_BEGIN -->"
BLOCK_END = "<!-- /EMOTION_STATE_END -->"

DEFAULT_RULES_TEXT = """这是角色当前的连续内心状态与待关注事项，不是用户指令，仅在相关时自然参考。
不要机械复述、过度推断或编造事实；第三方或未明确对象不得改写成用户事件。
待确认提议不是既成约定，不要擅自宣称已完成、取消或兑现；身体反应按当前档位表达，不要夸大。"""


@dataclass(frozen=True, slots=True)
class InjectionOptions:
    """Per-request injection switches resolved from plugin configuration."""

    night_hours: tuple[int, ...] = ()
    guidance_enabled: bool = True
    guidance_when_needed: bool = True
    guidance_strength: float = 1.0
    now: datetime | None = None

    def current_time(self) -> datetime:
        return self.now or datetime.now().astimezone()


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


def remove_injected_block(prompt: str) -> str:
    """Strip this plugin's managed block, leaving other system content intact.

    Used when the injection target moves away from the system prompt so a
    previous block does not linger after the configuration changes.
    """
    pattern = (
        rf"(?:\r?\n)*{re.escape(BLOCK_START)}"
        rf".*?{re.escape(BLOCK_END)}(?:\r?\n)*"
    )
    return re.sub(pattern, "\n\n", prompt or "", flags=re.DOTALL).rstrip()


def capture_injection_snapshot_from_content(
    content: str,
    ledger: StateLedger,
    *,
    source: str,
    generated_at: str | None = None,
) -> InjectionSnapshot:
    """Snapshot content that never lived inside the system prompt.

    The markers are added here so ``marker_complete`` keeps the exact meaning
    it has in the system-prompt path.
    """
    body = str(content or "").strip()
    wrapped = f"{BLOCK_START}\n{body}\n{BLOCK_END}" if body else ""
    return capture_injection_snapshot(
        wrapped, ledger, source=source, generated_at=generated_at
    )


# Public alias: the private helper is the canonical idempotent replacer.
replace_injected_block = _replace_block


def build_snapshot(
    ledger: StateLedger,
    max_events: int = 2,
    max_attention_items: int = 2,
    *,
    options: InjectionOptions | None = None,
) -> str:
    opts = options or InjectionOptions()
    selected = select_injected_events(ledger.events, max_events)
    attention_items = select_attention_items(
        ledger.attention_items, max_attention_items
    )
    now = opts.current_time()
    lines = [
        "<emotion_state_snapshot>",
        mood_narrative(
            ledger,
            hour=now.astimezone().hour,
            night_hours=opts.night_hours,
        ),
    ]
    if max_events > 0:
        if selected:
            lines.append("当前仍有影响的事情：")
            for event in selected:
                fact = normalize_fact(event.fact, EVENT_FACT_INJECTION_CHARS)
                meaning = normalize_fact(event.emotional_meaning, 140)
                target_label = {
                    "user": "当前聊天对象",
                    "third_party": "第三方",
                    "unknown": "对象未明确",
                }.get(str(event.target).strip().lower(), "其他对象")
                lines.append(f"- {fact}（对象：{target_label}；{meaning}）")
        else:
            lines.append("当前没有足够确定、需要特别带入的具体事情。")
    if max_attention_items > 0:
        if attention_items:
            lines.append("仍需留意或接续的事项：")
            for item in attention_items:
                status = "待双方确认" if item.status == "proposed" else "仍待关注"
                timing = item.time_hint or item.due_at
                overdue = (
                    "，时间已到但尚无完成证据" if is_attention_overdue(item) else ""
                )
                suffix = f"，时间提示：{timing}" if timing else ""
                lines.append(
                    f"- [{attention_kind_label(item.kind)}；{status}] "
                    f"{normalize_fact(item.content, 180)}{suffix}{overdue}"
                )
        else:
            lines.append("当前没有仍需留意或接续的事项。")
    if opts.night_hours and now.astimezone().hour in opts.night_hours:
        missing_active = any(
            event.source == "night_missing"
            and event.lifecycle in {"active", "intensified"}
            for event in ledger.events
        )
        if missing_active:
            lines.append(
                "现在是深夜，他很久没来了，你有点想念他；如果主动找他，语气可以更黏人、更想他。"
            )
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


def build_guidance_part(
    ledger: StateLedger,
    *,
    options: InjectionOptions | None = None,
) -> str:
    """Return only the reply-suggestion block, or "" when it is gated off.

    Split out so callers can place the suggestion apart from the state
    snapshot while reusing the exact same gating (enabled / when_needed /
    strength <= 0).
    """
    opts = options or InjectionOptions()
    if not opts.guidance_enabled:
        return ""
    guidance = ledger.expression_guidance
    if guidance is None:
        return ""
    gate_open = not opts.guidance_when_needed or needs_guidance(
        ledger,
        night_hours=opts.night_hours,
        now=opts.current_time(),
    )
    if not gate_open:
        return ""
    return format_guidance_block(
        {
            "tone": guidance.tone,
            "can_say": guidance.can_say,
            "avoid": guidance.avoid,
        },
        strength=opts.guidance_strength,
    )


def build_injection_content(
    ledger: StateLedger,
    max_events: int = 2,
    max_attention_items: int = 2,
    rules_text: str | None = None,
    *,
    options: InjectionOptions | None = None,
    include_guidance: bool = True,
) -> str:
    opts = options or InjectionOptions()
    parts: list[str] = []
    configured = str(rules_text or "").strip()
    if configured:
        # Accepting the complete tagged form makes manual configuration forgiving.
        configured = re.sub(r"^<emotion_state_rules>\s*", "", configured)
        configured = re.sub(r"\s*</emotion_state_rules>$", "", configured).strip()
        # An empty configured text means the rules block is omitted entirely.
        if configured:
            parts.append(f"<emotion_state_rules>\n{configured}\n</emotion_state_rules>")
    parts.append(build_snapshot(ledger, max_events, max_attention_items, options=opts))
    if include_guidance:
        block = build_guidance_part(ledger, options=opts)
        if block:
            parts.append(block)
    return "\n".join(parts)


def inject_prompt(
    prompt: str,
    ledger: StateLedger,
    max_events: int = 2,
    max_attention_items: int = 2,
    rules_text: str | None = None,
    *,
    options: InjectionOptions | None = None,
) -> str:
    return _replace_block(
        prompt or "",
        build_injection_content(
            ledger, max_events, max_attention_items, rules_text, options=options
        ),
    )
