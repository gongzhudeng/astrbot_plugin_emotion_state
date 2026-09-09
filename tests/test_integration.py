from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from astrbot_plugin_busy_schedule.core.generator import ScheduleGenerator
from astrbot_plugin_busy_schedule.main import _rebuild_system_prompt
from astrbot_plugin_emotion_state.core.daily import (
    build_daily_prompt,
    parse_daily_response,
)
from astrbot_plugin_emotion_state.core.injector import (
    ANCHOR,
    BLOCK_END,
    BLOCK_START,
    InjectionSnapshot,
    build_injection_content,
    inject_prompt,
)
from astrbot_plugin_emotion_state.core.models import (
    AttentionItem,
    DiaryEntry,
    InnerEvent,
    StateLedger,
)
from astrbot_plugin_emotion_state.core.provider_gateway import ProviderGateway
from astrbot_plugin_emotion_state.main import EmotionStatePlugin, sync_plugin_logo
from quart import Quart

from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.star_handler import star_handlers_registry

CACHE_START = "<!-- BUSY_SCHEDULE_CACHE -->"
CACHE_END = "<!-- /BUSY_SCHEDULE_CACHE -->"
CUSTOM_START = "<!-- BUSY_SCHEDULE_CUSTOM -->"
CUSTOM_END = "<!-- /BUSY_SCHEDULE_CUSTOM -->"


def busy_inject(prompt: str) -> str:
    return _rebuild_system_prompt(
        prompt,
        {
            "daily": "<character_static>今日穿搭、天气、完整日程</character_static>",
            "custom": "<character_custom>自定义动态内容</character_custom>",
        },
    )


def assert_final_prompt_order(prompt: str) -> None:
    positions = [
        prompt.index(CACHE_START),
        prompt.index(CACHE_END),
        prompt.index(CUSTOM_START),
        prompt.index(CUSTOM_END),
        prompt.index(ANCHOR),
        prompt.index(BLOCK_START),
        prompt.index(BLOCK_END),
    ]
    assert positions == sorted(positions)
    assert prompt.count(ANCHOR) == 1
    assert prompt.count(BLOCK_START) == 1
    assert prompt.count(BLOCK_END) == 1


def test_prompt_order_is_stable_for_both_hook_execution_orders() -> None:
    ledger = StateLedger(user_key="private:order")

    busy_first = inject_prompt(busy_inject("persona"), ledger)
    emotion_first = busy_inject(inject_prompt("persona", ledger))

    assert_final_prompt_order(busy_first)
    assert_final_prompt_order(emotion_first)
    assert busy_inject(emotion_first) == emotion_first
    assert inject_prompt(busy_first, ledger) == busy_first


def test_plugin_logo_syncs_into_dashboard_page(tmp_path: Path) -> None:
    root = tmp_path / "plugin"
    root.mkdir()
    target = tmp_path / "pages" / "dashboard" / "logo.png"

    # 图标缺失时静默跳过
    assert sync_plugin_logo(root, target) is False

    # 首次同步：写入副本
    (root / "logo.png").write_bytes(b"logo-v1")
    assert sync_plugin_logo(root, target) is True
    assert target.read_bytes() == b"logo-v1"

    # 内容一致时不重复写
    assert sync_plugin_logo(root, target) is False

    # 插件图标更新后（即使体积不同）重新同步
    (root / "logo.png").write_bytes(b"logo-v2-with-a-longer-payload")
    assert sync_plugin_logo(root, target) is True
    assert target.read_bytes() == b"logo-v2-with-a-longer-payload"


def test_dashboard_uses_ready_bridge_and_separate_query_params() -> None:
    dashboard_dir = Path(__file__).parents[1] / "pages" / "dashboard"
    dashboard = (dashboard_dir / "app.js").read_text(encoding="utf-8")
    markup = (dashboard_dir / "index.html").read_text(encoding="utf-8")
    styles = (dashboard_dir / "styles.css").read_text(encoding="utf-8")

    assert "bridge.ready()" in dashboard
    assert 'api("page/state", { params: { session_id: state.sessionId } })' in dashboard
    assert (
        'api("page/injection", { params: { session_id: state.sessionId } })'
        in dashboard
    )
    assert "`${endpoint(path)}?" not in dashboard
    assert "readSessionId" in dashboard
    assert "writeSessionId" in dashboard
    assert "withTimeout" in dashboard
    assert "载入失败：" in dashboard
    assert "renderDiaries" in dashboard
    assert "pageSize: 10" in dashboard
    assert 'id="diary-search"' in markup
    assert 'id="diary-month"' in markup
    assert 'id="mood-dials"' in markup
    # 情绪注入的两个视图各自独立成面板，不再挤在同一个双栏卡里
    assert "情绪注入 · 历史快照" in markup
    assert "情绪注入 · 实时预览" in markup
    assert 'id="prompt"' in markup
    assert 'id="preview-prompt"' in markup
    assert "injection-grid" not in markup
    assert "[hidden]{display:none !important}" in styles
    # 回复建议面板：右列第三行，占用待关注事项下方空区
    assert 'id="guidance-state"' in markup
    assert 'id="guidance-tone"' in markup
    assert 'id="guidance-rows"' in markup
    assert 'id="guidance-note"' in markup
    assert "guidance-card" in markup
    assert "renderGuidance" in dashboard
    assert r".guidance-card" in styles
    assert "grid-template-rows:auto auto auto" in styles
    assert "align-items:start" in styles


def test_emotion_commands_keep_group_and_standalone_routes_separate() -> None:
    handlers = star_handlers_registry.get_handlers_by_module_name(
        "astrbot_plugin_emotion_state.main"
    )
    command_filters = {
        metadata.handler_name: event_filter
        for metadata in handlers
        for event_filter in metadata.event_filters
        if isinstance(event_filter, CommandFilter)
    }
    group_filters = [
        event_filter
        for metadata in handlers
        for event_filter in metadata.event_filters
        if isinstance(event_filter, CommandGroupFilter)
    ]

    group_injection = command_filters["group_injection"]
    standalone_injection = command_filters["cmd_injection"]
    assert "情绪管理 注入" in group_injection.get_complete_command_names()
    assert group_injection.get_handler_md().extras_configs["sub_command"] is True
    assert standalone_injection.get_complete_command_names()[0] == "情绪注入"
    assert "sub_command" not in standalone_injection.get_handler_md().extras_configs
    assert group_injection.get_handler_md() is not standalone_injection.get_handler_md()

    emotion_group = next(
        item for item in group_filters if item.group_name == "情绪管理"
    )
    assert group_injection in emotion_group.sub_command_filters
    assert any(
        metadata.handler_name == "emotion_group"
        and emotion_group in metadata.event_filters
        for metadata in handlers
    )


def test_daily_prompt_preserves_persona_and_first_person_contract() -> None:
    prompt = build_daily_prompt(
        StateLedger(user_key="private:diary"),
        "2026-03-14",
        {"messages": ["用户说今天辛苦了"]},
        {"low_weight_mood_word": "温柔"},
        4000,
        "句子偏短，偶尔嘴硬",
    )

    assert "第一人称" in prompt
    assert "已有的人格" in prompt
    assert "不得编造" in prompt
    assert "句子偏短，偶尔嘴硬" in prompt
    assert "只能影响措辞" in prompt


class FakeProvider:
    def __init__(
        self, provider_id: str, *, text: str = "", error: Exception | None = None
    ):
        self._id = provider_id
        self._text = text
        self._error = error
        self.calls = 0

    def meta(self):
        return SimpleNamespace(id=self._id)

    async def text_chat(self, **_kwargs):
        self.calls += 1
        if self._error:
            raise self._error
        return SimpleNamespace(completion_text=self._text)


class FakeContext:
    def __init__(self, providers: dict[str, FakeProvider]):
        self.providers = providers

    def get_provider_by_id(self, provider_id: str):
        return self.providers.get(provider_id)

    def get_using_provider(self, _session_id: str = ""):
        return self.providers.get("session")


@pytest.mark.asyncio
async def test_provider_chain_uses_configured_order_and_falls_back() -> None:
    first = FakeProvider("first", error=RuntimeError("offline"))
    second = FakeProvider("second", text="")
    third = FakeProvider("third", text='{"ok":true}')
    gateway = ProviderGateway(
        FakeContext({"first": first, "second": second, "third": third}),
        {"analysis_provider_ids": ["first", "second", "third"]},
    )

    text, provider_id = await gateway.complete("prompt", "private:provider")

    assert (text, provider_id) == ('{"ok":true}', "third")
    assert [first.calls, second.calls, third.calls] == [1, 1, 1]


@pytest.mark.asyncio
async def test_provider_chain_retries_when_daily_response_is_incomplete() -> None:
    first = FakeProvider(
        "first",
        text='{"diary":"有日记，但没有摘要"}',
    )
    second = FakeProvider(
        "second",
        text='{"diary":"完整日记","day_summary":"完整摘要","confidence":0.8}',
    )
    gateway = ProviderGateway(
        FakeContext({"first": first, "second": second}),
        {"daily_provider_ids": ["first", "second"]},
    )

    text, provider_id = await gateway.complete(
        "daily prompt",
        "private:provider",
        task="daily",
        validate=lambda value: parse_daily_response(value, 600),
    )

    assert (text, provider_id) == (second._text, "second")
    assert [first.calls, second.calls] == [1, 1]


@pytest.mark.asyncio
async def test_provider_chains_are_independent() -> None:
    quick = FakeProvider("quick", text="quick result")
    deep = FakeProvider("deep", text="deep result")
    gateway = ProviderGateway(
        FakeContext({"quick": quick, "deep": deep}),
        {
            "review_provider_ids": ["quick"],
            "daily_provider_ids": ["deep"],
        },
    )

    review = await gateway.complete("review", "private:provider", task="review")
    daily = await gateway.complete("daily", "private:provider", task="daily")

    assert review == ("quick result", "quick")
    assert daily == ("deep result", "deep")
    assert [quick.calls, deep.calls] == [1, 1]


def test_legacy_unified_provider_chain_remains_compatible() -> None:
    gateway = ProviderGateway(
        FakeContext({}),
        {"analysis_provider_ids": ["legacy"]},
    )

    assert gateway.configured_ids("review") == ["legacy"]
    assert gateway.configured_ids("daily") == ["legacy"]


@pytest.mark.asyncio
async def test_provider_chain_uses_session_provider_only_when_list_is_empty() -> None:
    session = FakeProvider("session", text="session result")
    gateway = ProviderGateway(FakeContext({"session": session}), {})

    text, provider_id = await gateway.complete("prompt", "private:provider")

    assert (text, provider_id) == ("session result", "session")


class FakeService:
    def __init__(self, ledger: StateLedger | None = None):
        self.ledger = ledger
        self.get_calls: list[tuple[str, bool]] = []

    async def get(self, user_key: str, settle: bool = True):
        self.get_calls.append((user_key, settle))
        return self.ledger or StateLedger(user_key=user_key)


class DeleteCaptureService(FakeService):
    def __init__(self, ledger: StateLedger, *, changed: bool, reason: str):
        super().__init__(ledger)
        self.changed = changed
        self.reason = reason
        self.delete_calls: list[tuple[str, str, str]] = []

    async def delete_item(
        self,
        user_key: str,
        kind: str,
        item_id: str,
    ):
        self.delete_calls.append((user_key, kind, item_id))
        return self.ledger, self.changed, self.reason


class FakeEvent:
    unified_msg_origin = "private:command"
    message_str = "/情绪注入"

    def is_private_chat(self) -> bool:
        return True

    def plain_result(self, text: str) -> str:
        return text


@pytest.mark.asyncio
async def test_group_injection_command_returns_only_actual_emotion_block() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = FakeService()
    full_prompt = inject_prompt(
        f"PERSONA\n{ANCHOR}\n<!-- BUSY_SCHEDULE_ACTIVITY -->",
        StateLedger(user_key="private:command"),
    )
    plugin._last_injection_snapshot = {}
    plugin._last_injected_prompt = {
        "private:command": full_prompt[
            full_prompt.index(BLOCK_START) : full_prompt.index(BLOCK_END)
            + len(BLOCK_END)
        ]
    }

    results = [item async for item in plugin.group_injection(FakeEvent())]

    assert len(results) == 1
    assert results[0].startswith("【最近一次模型请求的历史快照（只读，已实际发送）】")
    assert results[0].count(BLOCK_START) == 1
    assert results[0].count(BLOCK_END) == 1
    assert "PERSONA" not in results[0]
    assert "BUSY_SCHEDULE_ACTIVITY" not in results[0]


@pytest.mark.asyncio
async def test_injection_command_returns_only_actual_emotion_block() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = FakeService()
    full_prompt = inject_prompt(
        f"PERSONA\n{ANCHOR}\n<!-- BUSY_SCHEDULE_ACTIVITY -->",
        StateLedger(user_key="private:command"),
    )
    plugin._last_injection_snapshot = {}
    plugin._last_injected_prompt = {
        "private:command": full_prompt[
            full_prompt.index(BLOCK_START) : full_prompt.index(BLOCK_END)
            + len(BLOCK_END)
        ]
    }

    results = [item async for item in plugin.cmd_injection(FakeEvent())]

    assert len(results) == 1
    assert results[0].startswith("【最近一次模型请求的历史快照（只读，已实际发送）】")
    assert results[0].count(BLOCK_START) == 1
    assert results[0].count(BLOCK_END) == 1
    assert "PERSONA" not in results[0]
    assert "BUSY_SCHEDULE_ACTIVITY" not in results[0]


@pytest.mark.asyncio
async def test_injection_command_labels_preview_before_first_request() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = FakeService()
    plugin._last_injection_snapshot = {}
    plugin._last_injected_prompt = {}

    results = [item async for item in plugin.cmd_injection(FakeEvent())]

    assert len(results) == 1
    assert results[0].startswith("【最近一次模型请求的历史快照（只读，已实际发送）】")
    assert "尚未产生请求装配记录。可使用“/情绪注入 预览”查看未发送预览。" in results[0]
    assert "当前账本实时预览" not in results[0]
    assert BLOCK_START not in results[0]


@pytest.mark.asyncio
async def test_injection_command_marks_older_request_snapshot_as_stale() -> None:
    ledger = StateLedger(user_key="private:command", state_version=9)
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = FakeService(ledger)
    actual_prompt = inject_prompt("", StateLedger(user_key="private:command"))
    actual_block = actual_prompt[
        actual_prompt.index(BLOCK_START) : actual_prompt.index(BLOCK_END)
        + len(BLOCK_END)
    ]
    plugin._last_injected_prompt = {"private:command": actual_block}
    plugin._last_injection_snapshot = {
        "private:command": InjectionSnapshot(
            prompt=actual_block,
            state_version=7,
            generated_at="2026-08-04T13:41:28+00:00",
            request_source="spark_proactive",
            marker_complete=True,
        )
    }

    results = [item async for item in plugin.cmd_injection(FakeEvent())]

    assert len(results) == 1
    assert "Spark 主动对话" in results[0]
    assert "历史请求时点，比当前实时账本落后 2 个版本" in results[0]
    assert "状态版本：7（历史请求时点，比当前实时账本落后 2 个版本）" in results[0]
    assert "请求装配时间：" in results[0]
    assert "预览生成时间：" not in results[0]
    assert "北京时间 UTC+08:00" in results[0]
    assert "2026-08-04 21:41:28" in results[0]
    assert "2026-08-04T13:41:28+00:00" not in results[0]


@pytest.mark.asyncio
async def test_request_snapshot_and_live_preview_share_stable_reaction_tier() -> None:
    ledger = StateLedger(user_key="private:cache", state_version=7)
    ledger.intimacy.body_sensitivity = 0.85
    ledger.intimacy.sexual_arousal = 0.75
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {"enabled": True}
    plugin.service = FakeService(ledger)
    plugin._last_injection_snapshot = {}
    plugin._last_injected_prompt = {}
    request = ProviderRequest(system_prompt="persona")

    await plugin.on_llm_request(FakeEvent(), request)
    actual = plugin._actual_injection_snapshot("private:command")

    ledger.state_version = 8
    ledger.intimacy.body_sensitivity = 1.0
    ledger.intimacy.sexual_arousal = 0.89
    preview = plugin._preview_injection_snapshot(ledger)

    assert actual is not None
    assert actual.prompt == preview.prompt
    assert "当前身体反应档位：强烈" in actual.prompt
    assert "0.85" not in actual.prompt
    assert "0.75" not in actual.prompt
    assert "1.00" not in preview.prompt
    assert "0.89" not in preview.prompt
    assert plugin.service.get_calls == [("private:command", True)]


@pytest.mark.asyncio
async def test_schedule_context_uses_settled_previous_day_without_private_state() -> (
    None
):
    ledger = StateLedger(
        user_key="private:schedule",
        diaries=[
            DiaryEntry(
                cycle_date="2026-03-13",
                diary="完整私密日记不应输出",
                day_summary=(
                    "昨天完成了项目，心里轻松了不少。"
                    "吃醋强度升高，因为一条私密事件感到不安。"
                    "亲密阶段变为 close。"
                ),
            )
        ],
    )
    ledger.mood.label = "温和愉快"
    ledger.mood.energy = 0.56
    ledger.mood.tension = 0.18
    ledger.intimacy.stage = "close"
    ledger.jealousy.intensity = 0.5
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.service = FakeService(ledger)

    context = await plugin._schedule_context("private:schedule", date(2026, 3, 14))

    assert "温和愉快" in context
    assert "昨天完成了项目" in context
    assert "完整私密日记" not in context
    assert "close" not in context
    assert "吃醋" not in context
    assert "私密事件" not in context
    assert "亲密阶段" not in context
    assert "软参考" in context


@pytest.mark.asyncio
async def test_schedule_context_requires_exact_previous_day_settlement() -> None:
    ledger = StateLedger(
        user_key="private:schedule",
        diaries=[
            DiaryEntry(
                cycle_date="2026-03-12",
                diary="旧日记",
                day_summary="这条旧摘要不能作为今天的心情。",
            )
        ],
    )
    ledger.mood.label = "温和愉快"
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.service = FakeService(ledger)

    context = await plugin._schedule_context("private:schedule", date(2026, 3, 14))

    assert context == "暂无已结算心情参考。"
    assert "旧摘要" not in context


@pytest.mark.asyncio
async def test_busy_schedule_prompt_fills_emotion_context() -> None:
    generator = ScheduleGenerator.__new__(ScheduleGenerator)
    generator.context = SimpleNamespace()
    generator.data_mgr = SimpleNamespace(get=lambda _target_date: None)
    generator.config = {
        "日程生成": {
            "prompt_template": "日期={date_str}\n心情={emotion_context}",
            "pool": {},
        }
    }
    generator._get_persona_desc = lambda _umo: _async_value("人设")
    generator._get_emotion_context = lambda _umo, _target_date: _async_value(
        "已结算心情"
    )
    generator._get_recent_chats = lambda _umo: _async_value("无近期对话")
    generator._get_rag_context = lambda _umo: _async_value("")
    generator._get_history_schedules = lambda _target_date: "无历史日程"
    generator._get_yesterday_last_activity = lambda _target_date: ""

    prompt = await generator._build_prompt(date(2026, 3, 14), umo="private:schedule")

    # busy_schedule dropped the random mood-color line and the reference
    # wrapper: the settled emotion text now renders as-is.
    assert "心情=已结算心情" in prompt
    assert "{emotion_context}" not in prompt


async def _async_value(value: str) -> str:
    return value


@pytest.mark.asyncio
async def test_busy_schedule_emotion_context_has_safe_fallback() -> None:
    generator = ScheduleGenerator.__new__(ScheduleGenerator)
    generator.context = SimpleNamespace()

    missing = await generator._get_emotion_context(
        "private:schedule", date(2026, 3, 14)
    )
    assert missing == "暂无已结算心情参考。"

    async def emotion_context(umo, target_date):
        assert umo == "private:schedule"
        assert target_date == date(2026, 3, 14)
        return "已结算心情"

    generator.context._emotion_state_schedule_context = emotion_context
    assert (
        await generator._get_emotion_context("private:schedule", date(2026, 3, 14))
        == "已结算心情"
    )

    async def failing_context(_umo, _target_date):
        raise RuntimeError("emotion offline")

    generator.context._emotion_state_schedule_context = failing_context
    assert (
        await generator._get_emotion_context("private:schedule", date(2026, 3, 14))
        == "暂无已结算心情参考。"
    )


class MemorySummaryCaptureService:
    def __init__(self):
        self.ledger = StateLedger(user_key="private:media-summary")
        self.calls = []

    async def get(self, _user_key: str):
        return self.ledger

    async def apply_memory_summary(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ledger, []


@pytest.mark.asyncio
async def test_pure_media_summary_advances_without_emotion_actions() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = MemorySummaryCaptureService()

    await plugin._consume_memory_summary(
        {
            "session_id": "private:media-summary",
            "end_index": 12,
            "emotion_source_has_text": False,
            "emotional_observations": [
                {
                    "action": "create",
                    "fact": "模型从视频猜测出的互动",
                    "category": "episodic",
                    "confidence": 0.9,
                }
            ],
            "mood_adjustment": {"valence": 1.0, "confidence": 1.0},
        }
    )

    args, kwargs = plugin.service.calls[0]
    assert args[1] == 12
    assert args[2] == []
    assert kwargs["mood_adjustment"] == {}


@pytest.mark.asyncio
async def test_pure_media_summary_can_complete_matching_attention() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = MemorySummaryCaptureService()

    await plugin._consume_memory_summary(
        {
            "session_id": "private:media-summary",
            "end_index": 13,
            "emotion_source_has_text": False,
            "attention_observations": [
                {
                    "action": "complete",
                    "item_id": "photo-item",
                    "item_version": 2,
                    "confidence": 0.95,
                    "evidence_quote": "[图片消息]",
                    "evidence_speaker": "assistant",
                }
            ],
        }
    )

    args, kwargs = plugin.service.calls[0]
    assert args[2] == []
    assert kwargs["mood_adjustment"] == {}
    observation = kwargs["attention_observations"][0]
    assert observation.action == "complete"
    assert observation.item_id == "photo-item"
    assert observation.evidence_quote == "[图片消息]"


def test_attention_history_response_distinguishes_empty_from_invalid() -> None:
    parse = EmotionStatePlugin._parse_attention_reconciliation_response

    assert parse('{"attention_observations": []}') == []
    assert parse("not json") is None
    assert parse('{"attention_observations": "invalid"}') is None
    assert parse(
        '{"attention_observations": ['
        '{"action":"cancel","item_id":"a"},'
        '{"action":"complete","item_id":"b"}'
        "]}"
    ) == [{"action": "complete", "item_id": "b"}]


@pytest.mark.asyncio
async def test_attention_history_backfill_without_history_remains_retryable(
    monkeypatch,
) -> None:
    class BackfillService:
        def __init__(self):
            self.ledger = StateLedger(
                user_key="private:no-history",
                attention_items=[
                    AttentionItem(
                        id="pending",
                        content="发一张照片给我",
                        status="open",
                        explicit=True,
                        confidence=0.9,
                    )
                ],
            )
            self.reconciliation_calls = 0

        async def get(self, _user_key):
            return self.ledger

        async def reconcile_attention_history(self, *_args, **_kwargs):
            self.reconciliation_calls += 1
            self.ledger.attention_reconciliation_version = 1
            return self.ledger, []

    async def no_sleep(_seconds):
        return None

    service = BackfillService()
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.service = service
    plugin.store = SimpleNamespace(user_keys=lambda: ["private:no-history"])
    plugin.context = SimpleNamespace(
        _livingmemory_get_attention_history=lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr("astrbot_plugin_emotion_state.main.asyncio.sleep", no_sleep)

    await plugin._run_attention_history_backfill()

    assert service.reconciliation_calls == 0
    assert service.ledger.attention_reconciliation_version == 0


@pytest.mark.asyncio
async def test_private_summary_forwards_versioned_merge_and_mood() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = MemorySummaryCaptureService()

    await plugin._consume_memory_summary(
        {
            "session_id": "private:media-summary",
            "summary_id": "private:media-summary:27",
            "end_index": 14,
            "emotion_source_has_text": True,
            "emotional_observations": [
                {
                    "action": "merge",
                    "event_id": "event-1",
                    "event_version": 3,
                    "fact": "他已经去上班了",
                    "emotional_meaning": "这件日常小事让我惦记",
                    "category": "episodic",
                    "valence": 0.2,
                    "intensity": 0.35,
                    "confidence": 0.8,
                    "uncertain": False,
                }
            ],
            "mood_adjustment": {
                "valence": 0.4,
                "energy": 0.55,
                "tension": 0.1,
                "confidence": 0.8,
            },
        }
    )

    args, kwargs = plugin.service.calls[0]
    observation = args[2][0]
    assert observation.action == "merge"
    assert observation.event_id == "event-1"
    assert observation.event_version == 3
    assert observation.category == "episodic"
    assert kwargs["summary_id"] == "private:media-summary:27"
    assert kwargs["mood_adjustment"]["valence"] == 0.4


@pytest.mark.asyncio
async def test_prompt_context_is_read_only_and_matches_injection_content() -> None:
    ledger = StateLedger(user_key="private:judge", state_version=7)
    ledger.mood.label = "平静"
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {"enabled": True, "max_injected_events": 2}
    plugin.service = FakeService(ledger)
    plugin._last_injection_snapshot = {}
    plugin._last_injected_prompt = {}

    before = ledger.to_dict()
    context = await plugin._prompt_context("private:judge")

    assert context == build_injection_content(ledger, 2)
    assert plugin.service.get_calls == [("private:judge", False)]
    assert ledger.to_dict() == before
    assert plugin._last_injection_snapshot == {}
    assert plugin._last_injected_prompt == {}


@pytest.mark.asyncio
async def test_state_api_explains_single_selected_event(monkeypatch) -> None:
    ledger = StateLedger(
        user_key="private:command",
        events=[
            InnerEvent(
                id="selected",
                fingerprint="selected-fingerprint",
                fact="仍有影响的事情",
                emotional_meaning="持续影响",
                category="psychological",
                lifecycle="active",
                intensity=0.8,
                confidence=0.9,
                occurrence_count=3,
            ),
            InnerEvent(
                id="candidate",
                fact="尚未确认的事情",
                emotional_meaning="等待更多证据",
                category="psychological",
                lifecycle="candidate",
                intensity=0.9,
                confidence=0.9,
            ),
            InnerEvent(
                id="transient",
                fact="短时情绪",
                emotional_meaning="短时影响",
                category="transient",
                lifecycle="active",
                intensity=0.9,
                confidence=0.9,
            ),
        ],
    )
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {"max_injected_events": 2}
    plugin.service = FakeService(ledger)
    plugin.context = SimpleNamespace()
    plugin.gateway = SimpleNamespace(configured_ids=lambda _task: [])
    plugin._last_injection_snapshot = {}
    plugin._last_injected_prompt = {}

    async def settled(_user_key):
        return ledger

    async def no_spark(_user_key):
        return {}

    monkeypatch.setattr(plugin, "_settle_proactive_state", settled)
    monkeypatch.setattr(plugin, "_spark_snapshot", no_spark)
    monkeypatch.setattr(plugin, "_api_user_key", lambda: ledger.user_key)

    payload = await plugin.state_api()
    diagnostics = payload["data"]["diagnostics"]
    evaluations = {
        item["id"]: item for item in diagnostics["injection_event_evaluations"]
    }

    assert diagnostics["selected_injection_event_count"] == 1
    assert diagnostics["max_injected_events"] == 2
    assert evaluations["selected"]["selected"] is True
    assert evaluations["selected"]["fingerprint"] == "selected-fingerprint"
    assert evaluations["selected"]["occurrence_count"] == 3
    assert evaluations["candidate"]["exclusion_reason"] == "lifecycle:candidate"
    assert evaluations["transient"]["exclusion_reason"] == "transient_category"


class AttentionCaptureService(FakeService):
    def __init__(self, ledger: StateLedger):
        super().__init__(ledger)
        self.attention_calls = []

    async def observe_attention(self, _user_key, observation):
        self.attention_calls.append(observation)
        return self.ledger, False, "not_called_in_normal_chat"


@pytest.mark.asyncio
async def test_normal_chat_has_no_local_attention_write_path() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {"enabled": True}
    plugin.service = AttentionCaptureService(StateLedger(user_key="private:command"))

    assert not hasattr(plugin, "on_llm_response")
    assert plugin.service.attention_calls == []


@pytest.mark.asyncio
async def test_memory_summary_forwards_versioned_attention_action() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {}
    plugin.service = MemorySummaryCaptureService()

    await plugin._consume_memory_summary(
        {
            "session_id": "private:media-summary",
            "summary_id": "private:media-summary:28",
            "end_index": 15,
            "emotion_source_has_text": True,
            "attention_observations": [
                {
                    "action": "confirm",
                    "item_id": "attention-1",
                    "item_version": 2,
                    "kind": "commitment",
                    "confidence": 0.9,
                    "evidence_quote": "你一定要记住",
                    "evidence_speaker": "user",
                }
            ],
        }
    )

    _, kwargs = plugin.service.calls[0]
    observation = kwargs["attention_observations"][0]
    assert observation.action == "confirm"
    assert observation.item_id == "attention-1"
    assert observation.item_version == 2
    assert observation.evidence_quote == "你一定要记住"


@pytest.mark.asyncio
async def test_state_text_uses_chinese_intimacy_and_attention_labels() -> None:
    ledger = StateLedger(
        user_key="private:command",
        attention_items=[
            AttentionItem(
                content="等一下咱们玩角色扮演",
                kind="plan",
                status="open",
                time_hint="等一下",
                confidence=0.9,
            )
        ],
    )
    ledger.intimacy.body_sensitivity = 0.85
    ledger.intimacy.sexual_arousal = 0.8
    ledger.intimacy.stage = "not_noticeable"
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {"persona_intimacy_tier": "非常亲密"}
    plugin.service = FakeService(ledger)

    output = await plugin._state_text(FakeEvent())

    assert "[计划/仍待关注]" in output
    assert "等一下咱们玩角色扮演" in output
    assert "人设亲密基线：非常亲密" in output
    assert "当前身体反应：身体反应强烈" in output
    assert "接受亲密行为的意愿" not in output
    assert "克制程度" not in output
    assert "open_and_receptive" not in output


@pytest.mark.asyncio
async def test_delete_item_api_forwards_stable_item_id() -> None:
    app = Quart(__name__)
    ledger = StateLedger(user_key="private:api", state_version=10)
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.service = DeleteCaptureService(ledger, changed=True, reason="archived")

    async with app.test_request_context(
        "/",
        method="POST",
        json={
            "session_id": "private:api",
            "kind": "event",
            "id": "event-1",
            "state_version": 9,
            "item_version": 4,
        },
    ):
        response = await plugin.delete_item_api()

    assert plugin.service.delete_calls == [("private:api", "event", "event-1")]
    assert response == {
        "status": "ok",
        "data": {
            "kind": "event",
            "id": "event-1",
            "status": "archived",
            "state_version": 10,
            "mood": ledger.to_dict()["mood"],
        },
    }


@pytest.mark.asyncio
async def test_delete_item_api_explains_missing_item() -> None:
    app = Quart(__name__)
    ledger = StateLedger(user_key="private:api", state_version=11)
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.service = DeleteCaptureService(
        ledger,
        changed=False,
        reason="item_not_found",
    )

    async with app.test_request_context(
        "/",
        method="POST",
        json={
            "session_id": "private:api",
            "kind": "attention",
            "id": "attention-1",
        },
    ):
        response = await plugin.delete_item_api()

    assert response["status"] == "error"
    assert response["reason"] == "item_not_found"
    assert response["state_version"] == 11
    assert response["message"] == "对象不存在，可能已被清理"


@pytest.mark.asyncio
async def test_injection_api_marks_actual_and_preview_sources_separately() -> None:
    app = Quart(__name__)
    ledger = StateLedger(user_key="private:api", state_version=9)
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {"enabled": True, "max_injected_attention_items": 9}
    plugin.service = FakeService(ledger)
    actual_prompt = inject_prompt(
        "persona", StateLedger(user_key="private:api", state_version=7)
    )
    actual_block = actual_prompt[
        actual_prompt.index(BLOCK_START) : actual_prompt.index(BLOCK_END)
        + len(BLOCK_END)
    ]
    plugin._last_injection_snapshot = {
        "private:api": InjectionSnapshot(
            prompt=actual_block,
            state_version=7,
            generated_at="2026-08-04T13:41:28+00:00",
            request_source="normal",
            marker_complete=True,
        )
    }
    plugin._last_injected_prompt = {}

    async with app.test_request_context("/?session_id=private:api"):
        response = await plugin.injection_api()

    data = response["data"]
    assert data["actual_available"] is True
    assert data["preview_only"] is False
    assert data["source"] == "last_actual_request"
    assert data["actual"]["source"] == "last_actual_request"
    assert data["actual"]["preview_only"] is False
    assert data["actual"]["stale"] is True
    assert data["preview"]["source"] == "current_ledger_preview"
    assert data["preview"]["preview_only"] is True
    assert data["preview"]["actual_available"] is False
    assert data["preview"]["state_version"] == 9
    assert data["preview"]["prompt"]


@pytest.mark.asyncio
async def test_attention_limit_is_clamped_for_state_rendering() -> None:
    ledger = StateLedger(
        user_key="private:command",
        attention_items=[
            AttentionItem(content=f"事项 {index}", confidence=0.9) for index in range(6)
        ],
    )
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.config = {"max_injected_attention_items": 99}
    plugin.service = FakeService(ledger)

    output = await plugin._state_text(FakeEvent())

    assert "后台未终结待关注事项（共 6 条）" in output
    assert "本次模型提示词实际最多注入 4 条，当前选中 4 条" in output


@pytest.mark.asyncio
async def test_emotion_terminate_cleans_registered_callbacks() -> None:
    plugin = EmotionStatePlugin.__new__(EmotionStatePlugin)
    plugin.context = SimpleNamespace(
        _emotion_state_memory_summary=plugin._consume_memory_summary,
        _emotion_state_review_context=plugin._review_context,
        _emotion_state_schedule_context=plugin._schedule_context,
        _emotion_state_get_prompt_context=plugin._prompt_context,
    )
    plugin._settlement_task = None
    plugin._proactive_task = None
    plugin._review_tasks = set()

    await plugin.terminate()

    assert not hasattr(plugin.context, "_emotion_state_memory_summary")
    assert not hasattr(plugin.context, "_emotion_state_review_context")
    assert not hasattr(plugin.context, "_emotion_state_schedule_context")
    assert not hasattr(plugin.context, "_emotion_state_get_prompt_context")


def test_private_and_command_boundaries() -> None:
    assert EmotionStatePlugin._is_private(FakeEvent()) is True
    assert EmotionStatePlugin._is_command(" /情绪状态") is True
    assert EmotionStatePlugin._is_command("！情绪状态") is True
    assert EmotionStatePlugin._is_command("今天心情不错") is False
