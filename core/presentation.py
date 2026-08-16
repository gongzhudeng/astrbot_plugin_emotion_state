"""Chinese presentation helpers for stable internal state enums."""

from __future__ import annotations

from .intimacy import body_reaction_stage
from .models import IntimacyState

_BODY_REACTION_STAGE_LABELS = {
    "not_noticeable": "身体平静，没有明显性反应",
    "slightly_aware": "身体开始发热，敏感度和性唤起轻微上升",
    "warmly_receptive": "身体反应明显，发热和敏感感持续增强",
    "open_and_receptive": "身体反应强烈，敏感度和性唤起处于高位",
    "warming": "身体开始发热，性唤起正在升起",
    "close": "身体反应较明显，敏感感持续",
    "aroused": "性唤起已经明显升高",
}
_BODY_REACTION_TIER_LABELS = {
    "not_noticeable": "平静",
    "slightly_aware": "轻微",
    "warmly_receptive": "明显",
    "open_and_receptive": "强烈",
}


def intimacy_stage_label(stage: str) -> str:
    return _BODY_REACTION_STAGE_LABELS.get(str(stage), "未知身体反应")


def intimacy_prompt_text(state: IntimacyState) -> str:
    stage = body_reaction_stage(state.body_sensitivity, state.sexual_arousal)
    tier = _BODY_REACTION_TIER_LABELS[stage]
    return (
        f"当前身体反应档位：{tier}。{intimacy_stage_label(stage)}。"
        "只按当前档位自然表达，不要夸大或编造已经发生的事实。"
    )
