"""Endogenous mood dynamics: circadian rhythm, temperament, drift, and missing.

All functions are deterministic given their inputs so that settlement stays
auditable and restart-safe. Nothing here calls models; the intrinsic layer only
shifts the mood baseline or emits low-intensity observations.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, time, timedelta, timezone

from .models import (
    EventObservation,
    IntrinsicParams,
    StateLedger,
    TemperamentState,
    clamp,
    iso_now,
)

TEMPERAMENT_SHIFTS: dict[str, tuple[float, float, float]] = {
    # word -> (valence, energy, tension) baseline shifts
    "平静": (0.0, 0.0, 0.0),
    "轻快": (0.12, 0.10, -0.02),
    "慵懒": (-0.02, -0.15, 0.03),
    "专注": (0.05, 0.08, 0.05),
    "温柔": (0.10, 0.02, -0.03),
    "内敛": (-0.02, -0.08, 0.04),
    "多愁善感": (-0.18, -0.05, 0.10),
    "黏人": (0.10, 0.06, 0.10),
    "雀跃": (0.18, 0.15, 0.02),
    "安静": (-0.05, -0.10, -0.02),
}


def parse_night_range(value: str) -> tuple[int, ...]:
    """Parse a "HH:MM-HH:MM" range into the set of night hours, wrapping midnight."""
    try:
        start_text, end_text = str(value or "").split("-", 1)
        start = time.fromisoformat(start_text.strip())
        end = time.fromisoformat(end_text.strip())
    except (TypeError, ValueError):
        return (22, 23, 0, 1)
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    span = (end_minutes - start_minutes) % 1440
    if span == 0:
        span = 1440
    hours = {(start_minutes + minute) % 1440 // 60 for minute in range(0, span, 15)}
    return tuple(sorted(hours))


def is_night_hour(hour: int, night_hours: tuple[int, ...] | frozenset[int]) -> bool:
    return int(hour) in night_hours


def circadian_shift(
    hour: int,
    night_hours: tuple[int, ...] | frozenset[int],
    strength: float,
) -> tuple[float, float, float]:
    """Return the (valence, energy, tension) baseline shift for the current hour."""
    if not night_hours or not is_night_hour(hour, night_hours):
        return (0.0, 0.0, 0.0)
    factor = clamp(float(strength), 0.0, 2.0)
    return (-0.15 * factor, -0.10 * factor, 0.08 * factor)


def _seeded_noise(seed_text: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return tuple((byte / 127.5) - 1.0 for byte in (digest[0], digest[1], digest[2]))  # type: ignore[return-value]


def drift_target(
    user_key: str,
    now: datetime,
    amplitude: float,
) -> tuple[float, float, float]:
    """Return a bounded, slowly wandering pseudo-random mood target offset."""
    bounded = clamp(float(amplitude), 0.0, 0.2)
    if bounded <= 0.0:
        return (0.0, 0.0, 0.0)
    bucket = int(now.timestamp() // 3600)
    noise = _seeded_noise(f"{user_key}:{bucket}")
    return (noise[0] * bounded, noise[1] * bounded, noise[2] * bounded)


def draw_temperament(
    user_key: str,
    cycle_date: str,
    words: tuple[str, ...] | list[str],
) -> TemperamentState:
    """Draw today's temperament deterministically from the date and user key."""
    candidates = [str(word).strip() for word in words if str(word).strip()] or ["平静"]
    index = int(
        hashlib.sha256(f"{user_key}:{cycle_date}".encode()).hexdigest(), 16
    ) % len(candidates)
    word = candidates[index]
    valence_shift, energy_shift, tension_shift = TEMPERAMENT_SHIFTS.get(
        word, (0.0, 0.0, 0.0)
    )
    return TemperamentState(
        word=word,
        date=cycle_date,
        valence_shift=valence_shift,
        energy_shift=energy_shift,
        tension_shift=tension_shift,
    )


def ensure_temperament(
    ledger: StateLedger,
    now: datetime,
    params: IntrinsicParams,
) -> bool:
    """Refresh the daily temperament draw in place when the calendar day changed."""
    if not params.temperament_enabled:
        return False
    today = now.astimezone().date().isoformat()
    if ledger.today_temperament.date == today and ledger.today_temperament.word:
        return False
    ledger.today_temperament = draw_temperament(
        ledger.user_key, today, params.temperament_words
    )
    return True


def intrinsic_baseline_shift(
    ledger: StateLedger,
    now: datetime,
    params: IntrinsicParams,
) -> tuple[float, float, float]:
    """Return the total (valence, energy, tension) baseline shift for this moment."""
    local_hour = now.astimezone().hour
    valence, energy, tension = circadian_shift(
        local_hour, params.night_hours, params.night_strength
    )
    temperament = ledger.today_temperament
    if params.temperament_enabled and temperament.word:
        valence += temperament.valence_shift
        energy += temperament.energy_shift
        tension += temperament.tension_shift
    return (valence, energy, tension)


def decay_mood_offset(
    ledger: StateLedger,
    elapsed_hours: float,
    half_life_minutes: float = 15.0,
) -> bool:
    """Decay the short-term mood offset in place; returns True when it changed."""
    offset = ledger.mood_offset
    before = (offset.valence, offset.energy, offset.tension)
    if all(abs(value) < 0.005 for value in before):
        if any(before):
            ledger.mood_offset = type(offset)()
            return True
        return False
    factor = math.pow(0.5, max(0.0, elapsed_hours) * 60.0 / max(1.0, half_life_minutes))
    offset.valence = clamp(offset.valence * factor, -0.35, 0.35)
    offset.energy = clamp(offset.energy * factor, -0.35, 0.35)
    offset.tension = clamp(offset.tension * factor, -0.35, 0.35)
    offset.updated_at = iso_now()
    return True


def maybe_night_missing_observation(
    ledger: StateLedger,
    now: datetime,
    params: IntrinsicParams,
) -> EventObservation | None:
    """Emit one low-intensity missing event when alone at night for too long."""
    threshold = max(0.0, float(params.night_missing_after_hours))
    if threshold <= 0.0 or not params.night_hours:
        return None
    local_hour = now.astimezone().hour
    if not is_night_hour(local_hour, params.night_hours):
        return None
    last_seen = ledger.last_user_message_ts
    if last_seen <= 0.0:
        return None
    now_ts = now.timestamp()
    quiet_hours = (now_ts - last_seen) / 3600.0
    if quiet_hours < threshold:
        return None
    fingerprint = f"night_missing:{now.astimezone().date().isoformat()}"
    if any(
        event.fingerprint == fingerprint and event.lifecycle != "archived"
        for event in ledger.events
    ):
        return None
    return EventObservation(
        action="create",
        fact="夜深了他还没来找我，房间安静得有点不习惯",
        emotional_meaning="夜里独处太久，开始想念聊天对象，有点低落和黏人",
        target="user",
        category="episodic",
        valence=-0.30 * max(params.sensitivity.attachment, 0.2),
        intensity=0.22,
        confidence=0.62,
        source="night_missing",
        fingerprint=fingerprint,
        note=f"夜间静置约 {quiet_hours:.1f} 小时",
        tags=["night_missing", "missing"],
    )


def next_life_event_window_check(now: datetime, slots: list[str]) -> datetime | None:
    """Return the first slot time that has arrived but is not obviously stale."""
    current = now.astimezone()
    due: list[datetime] = []
    for raw in slots:
        try:
            slot = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            continue
        if slot.tzinfo is None:
            slot = slot.replace(tzinfo=timezone.utc)
        slot = slot.astimezone(current.tzinfo)
        if slot <= current and current - slot <= timedelta(hours=6):
            due.append(slot)
    return min(due) if due else None
