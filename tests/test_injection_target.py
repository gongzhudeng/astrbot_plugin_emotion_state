from __future__ import annotations

import pytest
from astrbot.core.provider.entities import ProviderRequest

from astrbot_plugin_emotion_state.core.injector import (
    ANCHOR,
    BLOCK_END,
    BLOCK_START,
    InjectionOptions,
    build_guidance_part,
    build_injection_content,
    capture_injection_snapshot_from_content,
    remove_injected_block,
)
from astrbot_plugin_emotion_state.core.models import (
    ExpressionGuidance,
    MoodState,
    StateLedger,
)
from astrbot_plugin_emotion_state.main import EmotionStatePlugin

OPEN_OPTIONS = InjectionOptions(guidance_enabled=True, guidance_when_needed=False)


def _ledger() -> StateLedger:
    ledger = StateLedger(user_key="tester")
    ledger.mood = MoodState(valence=-0.4, label="低落")
    ledger.expression_guidance = ExpressionGuidance(
        tone="语气放轻放慢",
        can_say="可以说出心情不好",
        avoid="强撑着说没事",
        model_generated=True,
    )
    return ledger


def test_default_includes_guidance():
    ledger = _ledger()
    plain = build_injection_content(ledger, 2, 2, "规则", options=OPEN_OPTIONS)
    explicit = build_injection_content(
        ledger, 2, 2, "规则", options=OPEN_OPTIONS, include_guidance=True
    )
    assert plain == explicit
    assert "<emotion_state_guidance>" in plain


def test_excluding_guidance_keeps_snapshot_and_rules():
    ledger = _ledger()
    content = build_injection_content(
        ledger, 2, 2, "规则", options=OPEN_OPTIONS, include_guidance=False
    )
    assert "<emotion_state_rules>" in content
    assert "<emotion_state_snapshot>" in content
    assert "<emotion_state_guidance>" not in content


def test_guidance_part_matches_the_full_block():
    ledger = _ledger()
    full = build_injection_content(ledger, 2, 2, "规则", options=OPEN_OPTIONS)
    part = build_guidance_part(ledger, options=OPEN_OPTIONS)
    assert part
    assert part in full


def test_guidance_part_respects_the_gate():
    ledger = _ledger()
    closed = InjectionOptions(guidance_enabled=False)
    assert build_guidance_part(ledger, options=closed) == ""
    assert "<emotion_state_guidance>" not in build_injection_content(
        ledger, options=closed
    )


def test_remove_injected_block_clears_only_own_block():
    prompt = f"persona\n\n{BLOCK_START}\nold\n{BLOCK_END}\n\nkeep me"
    cleaned = remove_injected_block(prompt)
    assert BLOCK_START not in cleaned
    assert BLOCK_END not in cleaned
    assert "persona" in cleaned
    assert "keep me" in cleaned


def test_remove_injected_block_keeps_anchor():
    cleaned = remove_injected_block(
        f"persona {ANCHOR}\n{BLOCK_START}\nold\n{BLOCK_END}"
    )
    assert ANCHOR in cleaned


def test_snapshot_from_content_is_marker_complete():
    snapshot = capture_injection_snapshot_from_content(
        "body text", _ledger(), source="spark_proactive"
    )
    assert snapshot.marker_complete is True
    assert snapshot.prompt.startswith(BLOCK_START)
    assert snapshot.prompt.endswith(BLOCK_END)
    assert snapshot.request_source == "spark_proactive"


def test_snapshot_from_empty_content_is_empty():
    snapshot = capture_injection_snapshot_from_content("", _ledger(), source="normal")
    assert snapshot.prompt == ""


class _FakeEvent:
    unified_msg_origin = "private:target"
    message_str = "hello"

    def is_private_chat(self) -> bool:
        return True


class _FakeService:
    def __init__(self, ledger: StateLedger) -> None:
        self.ledger = ledger

    async def get(self, user_key: str, settle: bool = True) -> StateLedger:
        return self.ledger


def _plugin(**config: object) -> EmotionStatePlugin:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {
        "enabled": True,
        "expression_guidance_enabled": True,
        "expression_guidance_when_needed": False,
        "expression_strength": 1.0,
        **config,
    }
    plugin.service = _FakeService(_ledger())
    plugin._last_injected_prompt = {}
    plugin._last_injection_snapshot = {}
    return plugin


@pytest.mark.asyncio
async def test_hook_default_target_keeps_system_prompt():
    plugin = _plugin()
    req = ProviderRequest(system_prompt="persona")

    await plugin.on_llm_request(_FakeEvent(), req)

    assert BLOCK_START in req.system_prompt
    assert not list(req.extra_user_content_parts or [])
    assert plugin._last_injection_snapshot["private:target"].marker_complete is True


@pytest.mark.asyncio
async def test_hook_unknown_target_falls_back_to_system():
    plugin = _plugin(emotion_injection_target="乱七八糟")
    req = ProviderRequest(system_prompt="persona")

    await plugin.on_llm_request(_FakeEvent(), req)

    assert BLOCK_START in req.system_prompt
    assert not list(req.extra_user_content_parts or [])


@pytest.mark.asyncio
async def test_hook_user_target_moves_whole_block_and_cleans_system():
    plugin = _plugin(emotion_injection_target="临时用户上下文末尾")
    req = ProviderRequest(
        system_prompt=f"persona\n\n{BLOCK_START}\nstale\n{BLOCK_END}"
    )

    await plugin.on_llm_request(_FakeEvent(), req)

    assert BLOCK_START not in req.system_prompt
    assert "persona" in req.system_prompt
    parts = list(req.extra_user_content_parts or [])
    assert len(parts) == 1
    assert getattr(parts[0], "_no_save", False) is True
    assert "<emotion_state_snapshot>" in parts[0].text
    assert "<emotion_state_guidance>" in parts[0].text
    assert plugin._last_injection_snapshot["private:target"].marker_complete is True

    await plugin.on_llm_request(_FakeEvent(), req)
    assert len(list(req.extra_user_content_parts or [])) == 1


@pytest.mark.asyncio
async def test_hook_split_target_keeps_state_in_system():
    plugin = _plugin(emotion_injection_target="拆分（状态留系统，建议进用户）")
    req = ProviderRequest(system_prompt="persona")

    await plugin.on_llm_request(_FakeEvent(), req)

    assert "<emotion_state_snapshot>" in req.system_prompt
    assert "<emotion_state_guidance>" not in req.system_prompt
    parts = list(req.extra_user_content_parts or [])
    assert len(parts) == 1
    assert parts[0].text.startswith("<emotion_state_guidance>")
