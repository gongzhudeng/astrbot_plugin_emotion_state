"""Low-frequency simulated life events grounded in long-term memory.

Once or twice a day the plugin retrieves role-relevant memories (and optional
knowledge-base background), then asks a model to turn them into one small
inner event ("想起他前几天陪妈妈去医院，不知道结果怎么样"). These events are
regular InnerEvents, so they surface in chat naturally and feed Spark's
proactive judge without duplicating its responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any

from .models import StateLedger, clamp
from .settlement import event_score

_MEMORY_QUERY = "用户最近提到的生活近况 家人 朋友 学校 同学 健康 工作 住院 检查 约定"


def parse_time_window(value: str) -> tuple[time, time]:
    """Parse the single configured "HH:MM-HH:MM" generation window."""
    try:
        start_text, end_text = str(value or "").split("-", 1)
        start = time.fromisoformat(start_text.strip())
        end = time.fromisoformat(end_text.strip())
        if start >= end:
            raise ValueError("window start must precede end")
        return start, end
    except (TypeError, ValueError):
        return time(10, 0), time(22, 30)


def draw_slots(
    user_key: str,
    cycle_date: str,
    window_value: str,
    max_events: int,
) -> list[datetime]:
    """Draw deterministic random generation slots inside today's window."""
    bounded = max(0, min(4, int(max_events)))
    if bounded == 0:
        return []
    start, end = parse_time_window(window_value)
    local_now = datetime.now().astimezone()
    day = datetime.fromisoformat(cycle_date).date() if cycle_date else local_now.date()
    start_dt = datetime.combine(day, start, tzinfo=local_now.tzinfo)
    end_dt = datetime.combine(day, end, tzinfo=local_now.tzinfo)
    span_seconds = max(1, int((end_dt - start_dt).total_seconds()))
    digest = hashlib.sha256(f"{user_key}:{cycle_date}:life".encode()).digest()
    count = 1 + (digest[0] % bounded) if bounded > 1 else 1
    slots: list[datetime] = []
    used: set[int] = set()
    for index in range(count):
        offset = int.from_bytes(digest[1 + index * 2 : 3 + index * 2], "big")
        position = offset % span_seconds
        if position in used:
            position = (position + span_seconds // 3) % span_seconds
        used.add(position)
        slots.append(start_dt + timedelta(seconds=position))
    return sorted(slots)


def memory_query(ledger: StateLedger) -> str:
    """Compose the role-view retrieval query for long-term memory."""
    top_event = max(
        (
            event
            for event in ledger.events
            if event.lifecycle in {"active", "intensified"}
            and event.source not in {"night_missing"}
        ),
        key=event_score,
        default=None,
    )
    suffix = f" {top_event.fact[:40]}" if top_event else ""
    return f"{_MEMORY_QUERY}{suffix}"


def build_life_event_prompt(
    ledger: StateLedger,
    *,
    now: datetime,
    memories: list[str],
    kb_context: list[str],
) -> str:
    payload = {
        "time_band": now.astimezone().strftime("%H:%M"),
        "today_temperament": ledger.today_temperament.word,
        "current_events": [{"fact": event.fact[:120]} for event in ledger.events[:5]],
        "memory_materials": [item[:200] for item in memories[:6]],
        "background_materials": [item[:200] for item in kb_context[:4]],
    }
    return (
        "你是角色的内心声音。请根据材料为角色生成一条此刻自然想起的小事，"
        "优先从 memory_materials 里挑一件用户提到过的真实生活小事，"
        "从角色（想念、挂心、好奇、关心）的视角写成内心活动；"
        "background_materials 只用于补充角色的家庭、学校背景，让事情具体。"
        "材料不足时可以生成一条与当前心境一致的普通内心小事，但不得编造具体的外部新闻或大事。"
        '只返回 JSON 对象：{"skip": false, "fact": "...", '
        '"emotional_meaning": "...", "valence": -1~1, "intensity": 0~1}。'
        "fact 不超过 60 字，第一人称内心活动；valence/intensity 反映这件事带来的情绪。"
        '若此刻确实不适合生成任何小事，返回 {"skip": true}。\n'
        f"材料：{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_life_event_response(text: str) -> dict[str, Any] | None:
    """Parse one model life-event response; None means skip."""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    if not isinstance(data, dict) or bool(data.get("skip", False)):
        return None
    fact = str(data.get("fact", "")).strip()[:120]
    if not fact:
        raise ValueError("life event response missing fact")
    return {
        "fact": fact,
        "emotional_meaning": str(data.get("emotional_meaning", "")).strip()[:180],
        "valence": clamp(float(data.get("valence", 0.0)), -1.0, 1.0),
        "intensity": clamp(float(data.get("intensity", 0.3))),
    }


def event_slot_due(ledger: StateLedger, now: datetime) -> bool:
    """True when at least one today slot has arrived and not been consumed."""
    local_now = now.astimezone()
    if ledger.life_event_slots_date != local_now.date().isoformat():
        return False
    for raw in ledger.life_event_slots:
        try:
            slot = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            continue
        if slot.tzinfo is None:
            slot = slot.replace(tzinfo=timezone.utc)
        slot = slot.astimezone(local_now.tzinfo)
        if slot <= local_now and local_now - slot <= timedelta(hours=6):
            return True
    return False
