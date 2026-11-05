"""未注入的待关注事项：把「为什么没进」说出来。"""

from __future__ import annotations

from astrbot_plugin_emotion_state.core.attention import attention_exclusion_reason
from astrbot_plugin_emotion_state.core.models import AttentionItem


def test_proposed_is_excluded_as_unconfirmed() -> None:
    item = AttentionItem(content="他可能会去下盗版", status="proposed", explicit=False, confidence=0.7)
    assert attention_exclusion_reason(item) == "待双方确认，不算约定"


def test_not_explicit_is_excluded() -> None:
    item = AttentionItem(content="她猜他要去买书", status="open", explicit=False, confidence=0.9)
    assert attention_exclusion_reason(item) == "不是明说的约定"


def test_low_confidence_is_excluded() -> None:
    item = AttentionItem(content="他好像说过要早睡", status="open", explicit=True, confidence=0.4)
    assert attention_exclusion_reason(item) == "证据不足"


def test_short_lived_is_excluded() -> None:
    # 「等一下 + 动作」这种一次性的即时承诺，不该占着长期事项的槽位。
    item = AttentionItem(
        content="等一下拍给你看",
        status="open",
        explicit=True,
        confidence=0.9,
    )
    assert attention_exclusion_reason(item) == "短期状态，不进注入"


def test_normal_item_has_no_reason() -> None:
    item = AttentionItem(content="周末一起看电影", status="open", explicit=True, confidence=0.9)
    assert attention_exclusion_reason(item) == ""


def test_limit_reached_only_when_otherwise_eligible() -> None:
    item = AttentionItem(content="周末一起看电影", status="open", explicit=True, confidence=0.9)
    assert attention_exclusion_reason(item, limit_reached=True) == "被条数上限挤掉"

    proposed = AttentionItem(content="可能要去", status="proposed", explicit=False)
    # 先命中的原因优先：被上限挤掉的前提是它本来该进。
    assert attention_exclusion_reason(proposed, limit_reached=True) == "待双方确认，不算约定"


def test_closed_item_reports_finished() -> None:
    item = AttentionItem(content="已经做完的事", status="completed", explicit=True, confidence=0.9)
    assert attention_exclusion_reason(item) == "已终结"
