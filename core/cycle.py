"""生理期状态：按角色存一个开始日，其余全部现算，不建状态机。

设计约定（与内心世界其它层的关系）：

* 它是**身体状态**，不是一件事——因此不进 ``ledger.events``、不进 ``attention_items``，
  只作为一条独立的身体事实进入注入（``injector.build_snapshot``）。
* **空就是空**：非经期、经前、未配置开始日时，文案一律返回空串，调用方 ``if line:`` 才
  追加。绝不输出「身体状态：正常」这类占位——占位会让模型把「没有」当成一条信息。
* 三处文案的用途不同，**不能混用**：

  - ``body_line``：给对话注入，允许出现「经期第 N 天」。她可以说，也可以不说，
    由模型按人设决定。
  - ``schedule_hint``：给日程**生成模板**，同样允许出现「经期第 N 天」，
    因为模板是模型读的说明书。
  - 但最终生成的**日程文本**里不该出现这几个字——它只该体现在活动内容上
    （煮红糖姜茶、早点睡），所以生成提示词里另有一句约束，见 busy_schedule。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "PHASE_PERIOD",
    "PHASE_FOLLICULAR",
    "PHASE_OVULATION",
    "PHASE_LUTEAL",
    "CycleSettings",
    "phase_of",
    "body_line",
    "mood_delta",
    "schedule_hint",
    "parse_start",
    "load_settings",
    "save_settings",
    "update_role",
]

PHASE_PERIOD = "period"
PHASE_FOLLICULAR = "follicular"
PHASE_OVULATION = "ovulation"
PHASE_LUTEAL = "luteal"

CYCLE_LENGTH_MIN = 21
CYCLE_LENGTH_MAX = 35
PERIOD_LENGTH_MIN = 3
PERIOD_LENGTH_MAX = 7

#: 黄体期里只有最后这几天才值得说；前面那十几天和平时没有区别。
LUTEAL_NOTICE_DAYS = 7

#: 文案里不允许出现的词。它们会让「身体感觉」变成「病历术语」，
#: 而且一旦进入日程文本就会被每轮注入带回来（见模块 docstring）。
FORBIDDEN_WORDS = ("月经", "生理期", "例假", "大姨妈")


def _clamp_int(value: object, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


@dataclass(frozen=True, slots=True)
class CycleSettings:
    """一个角色的周期设置。``last_period_start`` 为空表示整套系统静默。"""

    last_period_start: str = ""
    cycle_length: int = 28
    period_length: int = 5

    def normalized(self) -> "CycleSettings":
        return CycleSettings(
            last_period_start=str(self.last_period_start or "").strip(),
            cycle_length=_clamp_int(self.cycle_length, CYCLE_LENGTH_MIN, CYCLE_LENGTH_MAX, 28),
            period_length=_clamp_int(self.period_length, PERIOD_LENGTH_MIN, PERIOD_LENGTH_MAX, 5),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "CycleSettings":
        if not isinstance(data, dict):
            return cls()
        return cls(
            last_period_start=str(data.get("last_period_start") or "").strip(),
            cycle_length=data.get("cycle_length", 28),
            period_length=data.get("period_length", 5),
        ).normalized()


def parse_start(raw: object) -> date | None:
    """解析开始日；非法或未来日期一律按「未配置」处理，不抛异常。"""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def phase_of(
    today: date,
    settings: CycleSettings | None,
) -> tuple[str, int] | None:
    """返回 (阶段, 周期第几天)；未配置或开始日在未来时返回 None。

    第 1 天是开始日。超过一个周期就按长度回卷——周期是重复的，
    不需要为「推迟」建额外状态。
    """
    if settings is None:
        return None
    settings = settings.normalized()
    start = parse_start(settings.last_period_start)
    if start is None or start > today:
        return None
    length = settings.cycle_length
    day = ((today - start).days % length) + 1
    if day <= settings.period_length:
        return PHASE_PERIOD, day
    if day <= 13:
        return PHASE_FOLLICULAR, day
    if day <= 16:
        return PHASE_OVULATION, day
    return PHASE_LUTEAL, day


def _luteal_notice(day: int, length: int) -> bool:
    return day >= max(17, length - LUTEAL_NOTICE_DAYS + 1)


def body_line(phase_day: tuple[str, int] | None, settings: CycleSettings | None = None) -> str:
    """对话注入用的身体事实；没有值得说的就返回空串。"""
    if phase_day is None:
        return ""
    phase, day = phase_day
    length = (settings or CycleSettings()).normalized().cycle_length
    if phase == PHASE_PERIOD:
        if day <= 2:
            return f"经期第{day}天，小腹坠着，腰有点酸，人没什么劲"
        return f"经期第{day}天，有点乏，腰还有点酸"
    if phase == PHASE_LUTEAL and _luteal_notice(day, length):
        return "经前这几天，身上有点胀，容易烦"
    return ""


def mood_delta(phase_day: tuple[str, int] | None, settings: CycleSettings | None = None) -> dict[str, float]:
    """对情绪结算的微调。幅度刻意很小：只让某种心情更容易出现，不换人设。"""
    zero = {"energy": 0.0, "tension": 0.0, "neg_mul": 1.0}
    if phase_day is None:
        return zero
    phase, day = phase_day
    length = (settings or CycleSettings()).normalized().cycle_length
    if phase == PHASE_PERIOD:
        if day <= 2:
            return {"energy": -0.10, "tension": 0.05, "neg_mul": 1.10}
        return {"energy": -0.05, "tension": 0.0, "neg_mul": 1.05}
    if phase == PHASE_LUTEAL and _luteal_notice(day, length):
        return {"energy": -0.03, "tension": 0.10, "neg_mul": 1.15}
    return zero


def schedule_hint(
    phase_day: tuple[str, int] | None,
    level: str = "light",
    settings: CycleSettings | None = None,
) -> str:
    """给日程生成模板的一句话说明。level 决定影响多深，不写具体活动。"""
    if phase_day is None:
        return ""
    phase, day = phase_day
    length = (settings or CycleSettings()).normalized().cycle_length
    is_period = phase == PHASE_PERIOD
    is_luteal = phase == PHASE_LUTEAL and _luteal_notice(day, length)
    if not (is_period or is_luteal):
        return ""
    level = str(level or "light").strip().lower()
    if level not in ("light", "medium", "heavy"):
        level = "light"
    if is_period:
        if level == "heavy":
            return f"经期第{day}天，人很乏；今天以休息为主"
        if level == "medium":
            return f"经期第{day}天，容易累；减少外出和剧烈活动，多安排在家"
        return f"经期第{day}天，容易累；饮食可以暖一点，喝些热的"
    if level == "heavy":
        return "经前这几天，身上发胀，人容易乏；今天以休息为主"
    if level == "medium":
        return "经前这几天，身上发胀，容易烦；减少外出，多安排轻松的事"
    return "经前这几天，身上有点胀，容易烦；可以安排些轻松的事"


# ---------------------------------------------------------------------------
# 存取：按角色分键，原子写。读失败一律当成「没配置」，不抛给调用方。
# ---------------------------------------------------------------------------


def load_settings(path: Path) -> dict[str, CycleSettings]:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    roles = payload.get("roles") if isinstance(payload, dict) else None
    if not isinstance(roles, dict):
        return {}
    return {
        str(role_id): CycleSettings.from_dict(raw)
        for role_id, raw in roles.items()
    }


def save_settings(path: Path, roles: dict[str, CycleSettings]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "roles": {role_id: settings.normalized().to_dict() for role_id, settings in roles.items()},
    }
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, file_path)


def update_role(path: Path, role_id: str, settings: CycleSettings) -> CycleSettings:
    """更新单个角色并落盘，返回规范化后的设置。"""
    roles = load_settings(path)
    normalized = settings.normalized()
    roles[str(role_id)] = normalized
    save_settings(path, roles)
    return normalized
