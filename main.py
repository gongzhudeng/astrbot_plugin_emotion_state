"""AstrBot adapter for the private emotion-state ledger."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quart import request

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.message.components import Image, Plain, Reply
from astrbot.core.provider.entities import ProviderRequest

from .core.attention import (
    attention_kind_label,
    attention_status_label,
    is_attention_overdue,
    select_attention_items,
)
from .core.daily import (
    build_daily_prompt,
    local_daily_fallback,
    logical_day,
    parse_boundary,
    parse_daily_response,
)
from .core.image_renderer import EmotionStateImageRenderer
from .core.injector import (
    InjectionSnapshot,
    build_injection_content,
    build_snapshot,
    capture_injection_snapshot,
    inject_prompt,
    request_source,
)
from .core.intimacy import (
    body_reaction_stage,
    persona_intimacy_multiplier,
    persona_intimacy_tier_label,
)
from .core.models import (
    AttentionObservation,
    DiaryEntry,
    EventObservation,
    StateLedger,
)
from .core.presentation import intimacy_stage_label
from .core.provider_gateway import ProviderGateway
from .core.rules import LocalRuleEngine
from .core.service import EmotionStateService
from .core.settlement import (
    acknowledge_proactive_reply,
    archive_legacy_transient_events,
    is_legacy_transient_event,
    jealousy_evidence,
    select_injected_events,
    select_injected_events_with_reasons,
    settle_intimacy,
    settle_jealousy,
    settle_mood_proposal,
    settle_proactive_evidence,
    settle_transient_mood,
    settle_unanswered_proactive,
)
from .core.storage import LedgerStore

PLUGIN_NAME = "astrbot_plugin_emotion_state"


@register(
    PLUGIN_NAME,
    "灵犀 · 内心世界",
    "私聊专用的连续情绪、心事、每日回顾与亲密状态系统。",
    "v0.1.5",
    "https://github.com/gongzhudeng/astrbot_plugin_emotion_state",
)
class EmotionStatePlugin(Star):
    """Coordinates the private-chat boundary and domain services."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self.store = LedgerStore(self.data_dir)
        self.service = EmotionStateService(
            self.store,
            self._float_config("decay_half_life_hours", 72.0),
            self._int_config("max_active_psychological_events", 6),
            self._int_config("max_open_attention_items", 8),
            self._int_config("max_active_episodic_events", 6),
        )
        self.rules = LocalRuleEngine(self._list_config("custom_rules"))
        self.gateway = ProviderGateway(context, config)
        self.image_renderer = EmotionStateImageRenderer(Path(__file__).resolve().parent)
        self._last_injected_prompt: dict[str, str] = {}
        self._last_injection_snapshot: dict[str, InjectionSnapshot] = {}
        self._review_tasks: set[asyncio.Task[Any]] = set()
        self._settlement_task: asyncio.Task[Any] | None = None
        self._proactive_task: asyncio.Task[Any] | None = None
        self._attention_backfill_task: asyncio.Task[Any] | None = None
        self.context._emotion_state_memory_summary = self._consume_memory_summary
        self.context._emotion_state_review_context = self._review_context
        self.context._emotion_state_schedule_context = self._schedule_context
        self.context._emotion_state_get_prompt_context = self._prompt_context
        self._register_web_apis()

    def _config(self, key: str, default: Any = None) -> Any:
        value = self.config.get(key, default)
        return default if value is None else value

    def _list_config(self, key: str) -> list[Any]:
        value = self._config(key, [])
        if isinstance(value, list):
            return value
        return []

    def _float_config(self, key: str, default: float) -> float:
        try:
            return float(self._config(key, default))
        except (TypeError, ValueError):
            return default

    def _int_config(self, key: str, default: int) -> int:
        try:
            return int(self._config(key, default))
        except (TypeError, ValueError):
            return default

    def _attention_injection_limit(self) -> int:
        return max(0, min(4, self._int_config("max_injected_attention_items", 2)))

    @staticmethod
    def _is_private(event: AstrMessageEvent) -> bool:
        return bool(event.is_private_chat())

    @staticmethod
    def _is_command(text: str) -> bool:
        return text.lstrip().startswith(("/", "！", "!"))

    @staticmethod
    def _message_context(event: AstrMessageEvent) -> str:
        """Treat a reply-only chain as quoted evidence, not a new user statement."""
        try:
            components = event.get_messages()
        except (AttributeError, TypeError):
            components = getattr(getattr(event, "message_obj", None), "message", [])
        has_reply = any(isinstance(item, Reply) for item in components or [])
        has_plain = any(
            isinstance(item, Plain) and str(item.text or "").strip()
            for item in components or []
        )
        return "quoted" if has_reply and not has_plain else "user"

    @staticmethod
    def _rule_text(event: AstrMessageEvent, fallback: str) -> str:
        """Use newly authored plain text while excluding quoted reply content."""
        try:
            components = event.get_messages()
        except (AttributeError, TypeError):
            components = getattr(getattr(event, "message_obj", None), "message", [])
        if not any(isinstance(item, Reply) for item in components or []):
            return fallback
        direct = " ".join(
            str(item.text or "").strip()
            for item in components or []
            if isinstance(item, Plain) and str(item.text or "").strip()
        )
        return direct or fallback

    def _user_key(self, event: AstrMessageEvent) -> str:
        return str(event.unified_msg_origin or "")

    async def initialize(self) -> None:
        if self._config("daily_settlement_enabled", False):
            self._settlement_task = asyncio.create_task(self._settlement_loop())
        if self._config("proactive_silence_enabled", True):
            self._proactive_task = asyncio.create_task(
                self._proactive_settlement_loop()
            )
        self._attention_backfill_task = asyncio.create_task(
            self._run_attention_history_backfill()
        )

    async def terminate(self) -> None:
        if (
            getattr(self.context, "_emotion_state_memory_summary", None)
            == self._consume_memory_summary
        ):
            delattr(self.context, "_emotion_state_memory_summary")
        if (
            getattr(self.context, "_emotion_state_review_context", None)
            == self._review_context
        ):
            delattr(self.context, "_emotion_state_review_context")
        if (
            getattr(self.context, "_emotion_state_schedule_context", None)
            == self._schedule_context
        ):
            delattr(self.context, "_emotion_state_schedule_context")
        if (
            getattr(self.context, "_emotion_state_get_prompt_context", None)
            == self._prompt_context
        ):
            delattr(self.context, "_emotion_state_get_prompt_context")
        if self._settlement_task:
            self._settlement_task.cancel()
        if self._proactive_task:
            self._proactive_task.cancel()
        attention_backfill_task = getattr(self, "_attention_backfill_task", None)
        if attention_backfill_task:
            attention_backfill_task.cancel()
        for task in list(self._review_tasks):
            task.cancel()
        self._review_tasks.clear()

    @filter.event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=20)
    async def on_private_message(self, event: AstrMessageEvent) -> None:
        if not self._config("enabled", True) or not self._is_private(event):
            return
        text = str(event.message_str or "").strip()
        if not text or self._is_command(text):
            return
        key = self._user_key(event)
        ledger = await self.service.advance_watermark(key)
        await self._settle_proactive_state(key)
        watermark = ledger.message_watermark
        rule_text = self._rule_text(event, text)
        run = self.rules.run(
            rule_text,
            context=self._message_context(event),
            watermark=watermark,
        )
        for observation in run.candidates[:3]:
            _, applied, _ = await self.service.observe(key, observation)
            if not applied:
                break
        if run.transient_signals:
            await self.service.mutate(
                key,
                "transient_mood",
                lambda current: settle_transient_mood(
                    current, run.transient_signals[:3]
                ),
                {"signals": len(run.transient_signals[:3])},
            )

        has_jealousy_evidence, jealousy_source, jealousy_strength = jealousy_evidence(
            text
        )
        await self.service.mutate_if_changed(
            key,
            "jealousy_settlement",
            lambda current: settle_jealousy(
                current,
                evidence=has_jealousy_evidence,
                source=jealousy_source,
                strength=jealousy_strength,
            ),
            {"source": jealousy_source, "evidence": has_jealousy_evidence},
        )

        if self._config("intimacy_enabled", False):
            keywords = [
                str(item)
                for item in self._list_config("intimacy_keywords")
                if str(item)
            ]
            matched = [item for item in keywords if item in text]
            tier = persona_intimacy_tier_label(
                self._config("persona_intimacy_tier", "很亲密")
            )
            await self.service.mutate_if_changed(
                key,
                "intimacy_settlement",
                lambda current: settle_intimacy(
                    current,
                    relevant=bool(matched),
                    strength=min(1.0, 0.3 + len(matched) * 0.2),
                    sensitivity_multiplier=persona_intimacy_multiplier(tier),
                    decay_half_life_hours=self._float_config(
                        "intimacy_decay_half_life_hours", 8.0
                    ),
                ),
                {"matched_rules": matched, "persona_intimacy_tier": tier},
            )

        if self._config("model_review_enabled", False) and any(
            item.uncertain for item in run.candidates
        ):
            baseline = await self.service.get(key)
            self._schedule_review(key, text, watermark, baseline.state_version)

    @filter.on_llm_request(priority=-10_000)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._config("enabled", True) or not self._is_private(event):
            return
        key = self._user_key(event)
        if not key:
            return
        ledger = await self.service.get(key)
        req.system_prompt = inject_prompt(
            req.system_prompt or "",
            ledger,
            self._int_config("max_injected_events", 2),
            self._attention_injection_limit(),
        )
        snapshot = capture_injection_snapshot(
            req.system_prompt,
            ledger,
            source=request_source(event),
        )
        self._last_injected_prompt[key] = snapshot.prompt
        self._last_injection_snapshot[key] = snapshot

    async def _review_context(self, user_key: str) -> dict[str, Any]:
        return await self.service.get_review_context(str(user_key).strip())

    async def _prompt_context(self, user_key: str) -> str:
        """Return the current prompt block without settling or recording a request."""
        key = str(user_key or "").strip()
        if not key or not self._config("enabled", True):
            return ""
        ledger = await self.service.get(key, settle=False)
        return build_injection_content(
            ledger,
            self._int_config("max_injected_events", 2),
            self._attention_injection_limit(),
        )

    @staticmethod
    def _schedule_summary(value: str) -> str:
        sensitive = re.compile(
            r"吃醋|嫉妒|占有欲|亲密(?:阶段|状态|数值)?|性唤起|身体敏感|"
            r"亲近意愿|克制程度|私密事件|内部状态|intimacy|jealous",
            re.I,
        )
        sentences = re.split(r"(?<=[。！？!?；;])\s*|\n+", str(value or ""))
        public = [
            " ".join(sentence.split())
            for sentence in sentences
            if sentence.strip() and not sensitive.search(sentence)
        ]
        return "".join(public)[:180]

    async def _schedule_context(
        self, user_key: str, target_date: str | date | None = None
    ) -> str:
        """Expose only settled, low-sensitivity mood context to schedule generation."""
        key = str(user_key or "").strip()
        if not key:
            return "暂无已结算心情参考。"
        ledger = await self.service.get(key)
        if not ledger.diaries:
            return "暂无已结算心情参考。"

        target = None
        if isinstance(target_date, datetime):
            target = target_date.date()
        elif isinstance(target_date, date):
            target = target_date
        elif target_date:
            try:
                target = date.fromisoformat(str(target_date)[:10])
            except ValueError:
                target = None
        target = target or date.today()
        previous_day = (target - timedelta(days=1)).isoformat()
        diary = next(
            (
                item
                for item in reversed(ledger.diaries)
                if item.cycle_date == previous_day
            ),
            None,
        )
        if diary is None:
            return "暂无已结算心情参考。"

        def level(value: float) -> str:
            if value >= 0.68:
                return "较高"
            if value >= 0.38:
                return "中等"
            return "较低"

        summary = self._schedule_summary(diary.day_summary)
        return (
            f"已结算的今日心情：{ledger.mood.label}；"
            f"能量{level(ledger.mood.energy)}，紧张程度{level(ledger.mood.tension)}。\n"
            f"最近回顾摘要：{summary or '暂无摘要'}\n"
            "这是软参考，只用于安排没有明确约定的时段和穿搭氛围；"
            "不得覆盖已确认计划、天气、安全或日程格式约束。"
        )

    @staticmethod
    def _parse_attention_reconciliation_response(
        value: str,
    ) -> list[dict[str, Any]] | None:
        """Extract the small, versioned action list emitted by the backfill review."""
        text = str(value or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        candidates = [text]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            actions = (
                payload.get("attention_observations", [])
                if isinstance(payload, dict)
                else payload
            )
            if isinstance(actions, list):
                return [
                    item
                    for item in actions
                    if isinstance(item, dict)
                    and str(item.get("action", "")).strip().lower() == "complete"
                ]
        return None

    async def _run_attention_history_backfill(self) -> None:
        """Reconcile old open items once when LivingMemory can provide source messages."""
        await asyncio.sleep(2)
        history_lookup = getattr(
            self.context, "_livingmemory_get_attention_history", None
        )
        if not callable(history_lookup):
            logger.info(
                "[EmotionState] attention history backfill skipped: LivingMemory unavailable"
            )
            return

        for user_key in self.store.user_keys():
            try:
                ledger = await self.service.get(user_key)
                if ledger.attention_reconciliation_version >= 1:
                    continue
                open_items = [
                    item
                    for item in ledger.attention_items
                    if item.status in {"proposed", "open"}
                ]
                if not open_items:
                    await self.service.reconcile_attention_history(
                        user_key, [], reconciliation_version=1
                    )
                    continue

                since = min(item.created_at for item in open_items)
                history = history_lookup(user_key, since=since, limit=600)
                if asyncio.iscoroutine(history):
                    history = await history
                if not isinstance(history, list) or not history:
                    logger.info(
                        "[EmotionState] attention history backfill retained %s: no source history",
                        user_key,
                    )
                    continue

                items = [
                    {
                        **item.review_view(),
                        "created_at": item.created_at,
                    }
                    for item in open_items[:12]
                ]
                history_json = json.dumps(history, ensure_ascii=False)[:12000]
                prompt = (
                    "你是待关注事项历史对账器。仅返回 JSON 对象"
                    "{\"attention_observations\":[...]}。\n"
                    "逐条比对目录和真实聊天记录。只有能确认完整事项已经发生时才输出"
                    "action=complete，并原样填写item_id、item_version、evidence_quote、"
                    "evidence_speaker和confidence(>=0.78)。用户或角色的具体完成说明，"
                    "以及与一次性发送事项匹配的[图片消息]/[语音消息]/[视频消息]/[文件消息]"
                    "都可作为证据。无关媒体、部分进展、泛泛的“弄好了/发了/好了”不得完成。"
                    "不要创建、取消、替代或更新事项；证据不足时输出空数组。\n"
                    f"目录：{json.dumps(items, ensure_ascii=False)}\n"
                    f"聊天记录：{history_json}"
                )
                response, provider_id = await self.gateway.complete(
                    prompt,
                    session_id=user_key,
                    task="review",
                )
                actions = self._parse_attention_reconciliation_response(response)
                if actions is None:
                    logger.warning(
                        "[EmotionState] attention history backfill retained %s: invalid review response",
                        user_key,
                    )
                    continue
                observations = self._parse_attention_observations(
                    actions, source="livingmemory_summary"
                )
                _, reasons = await self.service.reconcile_attention_history(
                    user_key, observations, reconciliation_version=1
                )
                logger.info(
                    "[EmotionState] attention history backfill settled: session=%s provider=%s reasons=%s",
                    user_key,
                    provider_id,
                    ",".join(reasons) or "none",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[EmotionState] attention history backfill failed for %s: %s",
                    user_key,
                    exc,
                )

    @staticmethod
    def _parse_attention_observations(
        value: Any,
        *,
        source: str,
    ) -> list[AttentionObservation]:
        if not isinstance(value, list):
            return []
        actions = {"create", "confirm", "update", "complete", "cancel", "supersede"}
        result: list[AttentionObservation] = []
        for raw in value[:6]:
            if not isinstance(raw, dict):
                continue
            action = str(raw.get("action", "create")).strip().lower()
            item_id = str(raw.get("item_id", "")).strip()[:80]
            content = str(raw.get("content", "")).strip()[:240]
            evidence_quote = str(raw.get("evidence_quote", "")).strip()[:240]
            evidence_speaker = str(raw.get("evidence_speaker", "")).strip().lower()[:24]
            explicit = bool(raw.get("explicit", False))
            if action not in actions or not evidence_quote:
                continue
            try:
                item_version = (
                    int(raw["item_version"])
                    if raw.get("item_version") is not None
                    else None
                )
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if action == "create":
                status = str(raw.get("status", "open")).strip().lower()
                if (
                    not content
                    or not evidence_quote
                    or confidence < (0.62 if status == "proposed" else 0.82)
                    or (
                        status != "proposed"
                        and (not explicit or evidence_speaker != "user")
                    )
                    or (
                        status == "proposed"
                        and evidence_speaker not in {"user", "assistant", "both"}
                    )
                ):
                    continue
            elif (
                not item_id
                or item_version is None
                or confidence < 0.78
                or (
                    action != "complete"
                    and evidence_speaker != "user"
                )
                or (
                    action == "complete"
                    and evidence_speaker not in {"user", "assistant", "both"}
                )
            ):
                continue
            result.append(
                AttentionObservation(
                    action=action,
                    item_id=item_id,
                    item_version=item_version,
                    content=content,
                    kind=str(raw.get("kind", "follow_up")),
                    status=str(raw.get("status", "open")),
                    actor=str(raw.get("actor", "both"))[:40],
                    time_hint=str(raw.get("time_hint", ""))[:80],
                    due_at=str(raw.get("due_at", ""))[:64],
                    confidence=confidence,
                    explicit=explicit,
                    source=source,
                    evidence_quote=evidence_quote,
                    evidence_speaker=evidence_speaker,
                    note=str(raw.get("note", ""))[:240],
                )
            )
        return result

    async def _consume_memory_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Consume optional LivingMemory suggestions through authoritative settlement."""
        user_key = str(payload.get("session_id", "")).strip()
        try:
            end_index = int(payload.get("end_index", 0))
        except (TypeError, ValueError):
            return {"status": "invalid_end_index"}
        if not user_key or end_index <= 0:
            return {"status": "invalid_payload"}
        summary_id = str(payload.get("summary_id", "")).strip()[:160]
        ledger = await self.service.get(user_key)

        source_has_text = bool(payload.get("emotion_source_has_text", True))
        observations = (
            payload.get("emotional_observations", []) if source_has_text else []
        )
        normalized: list[EventObservation] = []
        if isinstance(observations, list):
            for raw in observations[:6]:
                if not isinstance(raw, dict):
                    continue
                try:
                    action = str(raw.get("action", "create")).strip().lower()
                    if action not in {
                        "create",
                        "merge",
                        "intensify",
                        "ease",
                        "dormant",
                        "archive",
                        "retain",
                    }:
                        continue
                    event_id = str(raw.get("event_id", "")).strip()[:80]
                    if action != "create" and not event_id:
                        continue
                    try:
                        event_version = (
                            int(raw["event_version"])
                            if raw.get("event_version") is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        event_version = None
                    if action != "create" and event_version is None:
                        continue
                    normalized.append(
                        EventObservation(
                            action=action,
                            event_id=event_id,
                            event_version=event_version,
                            fact=str(raw.get("fact", ""))[:240],
                            emotional_meaning=str(
                                raw.get(
                                    "emotional_meaning",
                                    "长期记忆总结发现了持续的互动影响",
                                )
                            )[:240],
                            target=str(raw.get("target", "unknown"))[:80],
                            target_basis=str(raw.get("target_basis", ""))[:120],
                            evidence_quote=str(raw.get("evidence_quote", ""))[:240],
                            evidence_speaker=str(raw.get("evidence_speaker", ""))
                            .strip()
                            .lower()[:24],
                            category=str(raw.get("category", "concrete")),
                            valence=float(raw.get("valence", 0.0)),
                            intensity=float(raw.get("intensity", 0.35)),
                            confidence=float(raw.get("confidence", 0.5)),
                            source="livingmemory_summary",
                            message_watermark=ledger.message_watermark,
                            uncertain=bool(raw.get("uncertain", True)),
                            note=str(raw.get("note", ""))[:240],
                            tags=[str(tag)[:40] for tag in raw.get("tags", [])[:5]]
                            if isinstance(raw.get("tags", []), list)
                            else [],
                        )
                    )
                except (TypeError, ValueError):
                    continue

        attention_observations = self._parse_attention_observations(
            payload.get("attention_observations", []),
            source="livingmemory_summary",
        )
        try:
            expected_version = int(
                payload.get("emotion_review_state_version", ledger.state_version)
            )
        except (TypeError, ValueError):
            expected_version = ledger.state_version
        try:
            expected_message_watermark = int(
                payload.get("emotion_review_message_watermark")
            )
        except (TypeError, ValueError):
            expected_message_watermark = None
        mood_adjustment = payload.get("mood_adjustment", {}) if source_has_text else {}
        if not isinstance(mood_adjustment, dict):
            mood_adjustment = {}
        result_ledger, reasons = await self.service.apply_memory_summary(
            user_key,
            end_index,
            normalized,
            expected_version,
            expected_message_watermark,
            attention_observations=attention_observations,
            mood_adjustment=mood_adjustment,
            episodic_limit=self._int_config("max_active_episodic_events", 6),
            summary_id=summary_id or None,
        )
        status = "duplicate" if "duplicate_summary_id" in reasons else "applied"
        if "already_processed" in reasons:
            status = "legacy_duplicate"
        logger.info(
            "[EmotionState] memory summary settled: session=%s summary_id=%s "
            "end_index=%s status=%s reasons=%s state_version=%s",
            user_key,
            summary_id or "legacy",
            end_index,
            status,
            ",".join(reasons) or "none",
            result_ledger.state_version,
        )
        return {
            "status": status,
            "reasons": reasons,
            "state_version": result_ledger.state_version,
        }

    def _schedule_review(
        self,
        user_key: str,
        text: str,
        watermark: int,
        baseline_state_version: int,
    ) -> None:
        async def review() -> None:
            try:
                prompt = (
                    "请只返回 JSON 数组，判断以下私聊消息是否形成需要持续关注的心事。"
                    "考虑玩笑、引用、对象和前后文，不要把普通激动误判为性唤起。"
                    f"消息：{text[:1000]}"
                )
                response, provider_id = await self.gateway.complete(
                    prompt, user_key, task="review"
                )
                payload = json.loads(response)
                if not isinstance(payload, list):
                    return
                expected_version = baseline_state_version
                for item in payload[:3]:
                    if not isinstance(item, dict):
                        continue
                    ledger = await self.service.get(user_key)
                    if ledger.state_version != expected_version:
                        return
                    observation = EventObservation(
                        action=str(item.get("action", "create")),
                        fact=str(item.get("fact") or text[:240]),
                        emotional_meaning=str(
                            item.get("emotional_meaning") or "待复核的互动影响"
                        ),
                        target=str(item.get("target", "unknown")),
                        target_basis=str(item.get("target_basis", "")),
                        evidence_quote=str(item.get("evidence_quote", "")),
                        evidence_speaker=str(item.get("evidence_speaker", "")),
                        valence=float(item.get("valence", 0.0)),
                        intensity=float(item.get("intensity", 0.35)),
                        confidence=float(item.get("confidence", 0.5)),
                        source=f"model:{provider_id}",
                        message_watermark=watermark,
                        expected_state_version=expected_version,
                        uncertain=bool(item.get("uncertain", True)),
                    )
                    updated, applied, _ = await self.service.observe(
                        user_key, observation
                    )
                    if not applied:
                        return
                    expected_version = updated.state_version
            except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.debug("[EmotionState] asynchronous review skipped: %s", exc)

        task = asyncio.create_task(review())
        self._review_tasks.add(task)
        task.add_done_callback(self._review_tasks.discard)

    async def _daily_context(self, user_key: str) -> dict[str, Any]:
        callback = getattr(self.context, "_livingmemory_get_daily_context", None)
        if not callable(callback):
            return {}
        try:
            result = callback(
                user_key,
                self._int_config("diary_tail_messages", 12),
                self._int_config("diary_input_max_chars", 4000),
            )
            return await result if asyncio.iscoroutine(result) else result
        except Exception as exc:
            logger.warning("[EmotionState] LivingMemory daily context failed: %s", exc)
            return {}

    def _schedule_facts(self) -> dict[str, Any]:
        callback = getattr(self.context, "_busy_schedule_get_facts", None)
        if not callable(callback):
            return {}
        try:
            result = callback()
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.warning("[EmotionState] Busy Schedule facts failed: %s", exc)
            return {}

    async def _spark_snapshot(self, user_key: str) -> dict[str, Any]:
        callback = getattr(self.context, "_spark_get_proactive_state", None)
        if not callable(callback):
            return {}
        try:
            result = callback(user_key)
            result = await result if asyncio.iscoroutine(result) else result
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.warning("[EmotionState] Spark state lookup failed: %s", exc)
            return {}

    async def _settle_proactive_state(self, user_key: str) -> StateLedger:
        snapshot = await self._spark_snapshot(user_key)
        if not snapshot.get("available"):
            return await self.service.get(user_key, settle=False)
        now_ts = datetime.now().astimezone().timestamp()
        evidence = snapshot.get("evidence")
        has_structured_evidence = int(
            snapshot.get("schema_version", 0) or 0
        ) >= 1 and isinstance(evidence, list)

        if has_structured_evidence:

            def settle(current: StateLedger) -> StateLedger:
                return settle_proactive_evidence(
                    current,
                    evidence=evidence,
                    now_ts=now_ts,
                    threshold_minutes=self._float_config(
                        "proactive_silence_threshold_minutes", 180.0
                    ),
                    max_intensity=self._float_config(
                        "proactive_silence_max_intensity", 0.55
                    ),
                    max_stage=self._int_config("proactive_silence_max_stages", 3),
                    episodic_limit=self._int_config("max_active_episodic_events", 6),
                )

            updated, _ = await self.service.mutate_if_changed(
                user_key,
                "spark_proactive_evidence_settlement",
                settle,
                {"evidence_count": len(evidence)},
            )
            return updated

        proactive_ts = float(snapshot.get("last_proactive_reply_ts", 0.0) or 0.0)
        user_reply_ts = float(snapshot.get("last_user_reply_ts", 0.0) or 0.0)

        def settle_legacy(current: StateLedger) -> StateLedger:
            current = settle_unanswered_proactive(
                current,
                proactive_ts=proactive_ts,
                user_reply_ts=user_reply_ts,
                now_ts=now_ts,
                threshold_minutes=self._float_config(
                    "proactive_silence_threshold_minutes", 180.0
                ),
                max_intensity=self._float_config(
                    "proactive_silence_max_intensity", 0.55
                ),
                max_stage=self._int_config("proactive_silence_max_stages", 3),
            )
            return acknowledge_proactive_reply(
                current,
                proactive_ts=proactive_ts,
                user_reply_ts=user_reply_ts,
            )

        updated, _ = await self.service.mutate_if_changed(
            user_key,
            "spark_proactive_settlement",
            settle_legacy,
            {
                "proactive_ts": proactive_ts,
                "user_reply_ts": user_reply_ts,
            },
        )
        return updated

    async def _proactive_settlement_loop(self) -> None:
        while True:
            try:
                for user_key in await asyncio.to_thread(self.store.user_keys):
                    await self._settle_proactive_state(user_key)
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[EmotionState] proactive settlement failed: %s", exc)
                await asyncio.sleep(60)

    async def _run_daily_review(
        self, user_key: str, cycle_date: str, force: bool = False
    ) -> tuple[StateLedger, str]:
        ledger = await self.service.get(user_key)
        if not force and any(
            entry.cycle_date == cycle_date for entry in ledger.diaries
        ):
            return ledger, "already_settled"

        review = local_daily_fallback(ledger, self._int_config("diary_max_chars", 600))
        provider_id = "local_fallback"
        if self._config("diary_enabled", True):
            schedule_facts = self._schedule_facts()
            palette = [
                str(item)
                for item in self._list_config("daily_mood_palette")
                if str(item)
            ]
            if palette:
                index = sum(ord(char) for char in cycle_date) % len(palette)
                schedule_facts["low_weight_mood_word"] = palette[index]
            prompt = build_daily_prompt(
                ledger,
                cycle_date,
                await self._daily_context(user_key),
                schedule_facts,
                self._int_config("diary_input_max_chars", 4000),
                self._config("diary_style_hint", ""),
            )
            try:
                response, provider_id = await self.gateway.complete(
                    prompt,
                    user_key,
                    task="daily",
                    validate=lambda text: parse_daily_response(
                        text, self._int_config("diary_max_chars", 600)
                    ),
                )
                review = parse_daily_response(
                    response, self._int_config("diary_max_chars", 600)
                )
            except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[EmotionState] Daily review used local fallback for %s: %s",
                    user_key,
                    exc,
                )

        entry = DiaryEntry(
            cycle_date=cycle_date,
            diary=review["diary"],
            day_summary=review["day_summary"],
            mood_proposal=review["next_mood_proposal"],
            confidence=review["confidence"],
            provider_id=provider_id,
            input_watermark=ledger.message_watermark,
        )
        observations = review.get("event_observations", [])
        expected_version = ledger.state_version
        if isinstance(observations, list):
            for raw in observations[:3]:
                if not isinstance(raw, dict):
                    continue
                current = await self.service.get(user_key)
                if current.state_version != expected_version:
                    break
                try:
                    observation = EventObservation(
                        action=str(raw.get("action", "create")),
                        fact=str(raw.get("fact", ""))[:240],
                        emotional_meaning=str(
                            raw.get("emotional_meaning", "每日回顾发现的持续影响")
                        )[:240],
                        target=str(raw.get("target", "unknown"))[:80],
                        target_basis=str(raw.get("target_basis", ""))[:120],
                        evidence_quote=str(raw.get("evidence_quote", ""))[:240],
                        evidence_speaker=str(raw.get("evidence_speaker", ""))[:24],
                        valence=float(raw.get("valence", 0.0)),
                        intensity=float(raw.get("intensity", 0.35)),
                        confidence=float(raw.get("confidence", 0.5)),
                        source=f"daily_review:{provider_id}",
                        message_watermark=entry.input_watermark,
                        expected_state_version=expected_version,
                        uncertain=bool(raw.get("uncertain", True)),
                    )
                except (TypeError, ValueError):
                    continue
                updated, applied, _ = await self.service.observe(user_key, observation)
                if not applied:
                    break
                expected_version = updated.state_version

        def commit(current: StateLedger) -> StateLedger:
            proposal_is_current = current.state_version == expected_version
            current = settle_mood_proposal(
                current,
                entry.mood_proposal if proposal_is_current else {},
                entry.confidence if proposal_is_current else 0.0,
            )
            current.diaries = [
                item for item in current.diaries if item.cycle_date != cycle_date
            ]
            current.diaries.append(entry)
            retention = max(1, self._int_config("diary_retention_days", 90))
            current.diaries = current.diaries[-retention:]
            current.logical_day = cycle_date
            return current

        updated = await self.service.mutate(
            user_key,
            "daily_review",
            commit,
            {"cycle_date": cycle_date, "provider_id": provider_id},
        )
        return updated, provider_id

    @filter.command_group("情绪管理", alias={"emotion-management"})
    def emotion_group(self):
        """情绪与亲密状态管理指令组，集中查看状态、回顾、规则和注入信息。"""
        pass

    def _actual_injection_snapshot(self, key: str) -> InjectionSnapshot | None:
        snapshot = getattr(self, "_last_injection_snapshot", {}).get(key)
        if snapshot:
            return snapshot
        prompt = getattr(self, "_last_injected_prompt", {}).get(key)
        if not prompt:
            return None
        return InjectionSnapshot(
            prompt=prompt,
            state_version=-1,
            generated_at="",
            request_source="legacy",
            marker_complete=bool(prompt),
        )

    def _preview_injection_snapshot(self, ledger: StateLedger) -> InjectionSnapshot:
        prompt = inject_prompt(
            "",
            ledger,
            self._int_config("max_injected_events", 2),
            self._attention_injection_limit(),
        )
        return capture_injection_snapshot(prompt, ledger, source="preview")

    @staticmethod
    def _beijing_display_time(value: str) -> str:
        if not value:
            return "未知"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            beijing = parsed.astimezone(timezone(timedelta(hours=8)))
        except ValueError:
            return value
        return f"{beijing:%Y-%m-%d %H:%M:%S}（北京时间 UTC+08:00）"

    @classmethod
    def _snapshot_meta(
        cls,
        snapshot: InjectionSnapshot,
        *,
        current_version: int,
    ) -> str:
        source_labels = {
            "normal": "普通对话",
            "spark_proactive": "Spark 主动对话",
            "legacy": "旧缓存（来源未知）",
            "preview": "当前账本实时预览",
        }
        version = str(snapshot.state_version) if snapshot.state_version >= 0 else "未知"
        generated_at = cls._beijing_display_time(snapshot.generated_at)
        time_label = (
            "预览生成时间" if snapshot.request_source == "preview" else "请求装配时间"
        )
        freshness = "版本未知"
        if snapshot.state_version >= 0:
            delta = current_version - snapshot.state_version
            freshness = (
                "与当前账本版本一致"
                if delta == 0
                else f"历史请求时点，比当前实时账本落后 {delta} 个版本"
            )
        return (
            f"来源：{source_labels.get(snapshot.request_source, snapshot.request_source)}；"
            f"状态版本：{version}（{freshness}）；{time_label}：{generated_at}"
        )

    async def _injection_text(
        self, event: AstrMessageEvent, mode: str = ""
    ) -> str | None:
        if not self._is_private(event):
            return None
        key = self._user_key(event)
        ledger = await self.service.get(key)
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode in {"预览", "preview"}:
            preview = self._preview_injection_snapshot(ledger)
            return "\n".join(
                [
                    "【当前账本实时预览（只读，尚未发送）】",
                    "此内容仅用于诊断，不会因此再次注入或发送给模型。",
                    self._snapshot_meta(preview, current_version=ledger.state_version),
                    preview.prompt,
                ]
            )

        actual = self._actual_injection_snapshot(key)
        lines = [
            "【最近一次模型请求的历史快照（只读，已实际发送）】",
            "此命令只读取历史副本，不会因此再次注入或发送给模型。",
        ]
        if actual:
            lines.extend(
                [
                    self._snapshot_meta(actual, current_version=ledger.state_version),
                    actual.prompt,
                ]
            )
        else:
            lines.append("尚未产生请求装配记录。可使用“/情绪注入 预览”查看未发送预览。")
        return "\n".join(lines)

    async def _state_text(self, event: AstrMessageEvent) -> str | None:
        if not self._is_private(event):
            return None
        ledger = await self.service.get(self._user_key(event))
        return self._format_state(ledger)

    async def _emotion_view_png(self, event: AstrMessageEvent) -> bytes | None:
        if not self._is_private(event):
            return None
        ledger = await self.service.get(self._user_key(event))
        events = select_injected_events(
            ledger.events, self._int_config("max_injected_events", 2)
        )
        attention_items = select_attention_items(
            ledger.attention_items,
            self._attention_injection_limit(),
            include_proposed=False,
        )
        stage = body_reaction_stage(
            ledger.intimacy.body_sensitivity,
            ledger.intimacy.sexual_arousal,
        )
        renderer = getattr(self, "image_renderer", None)
        if renderer is None:
            renderer = EmotionStateImageRenderer(Path(__file__).resolve().parent)
        return await asyncio.to_thread(
            renderer.render,
            ledger,
            events,
            attention_items,
            intimacy_stage_label(stage),
            datetime.now().astimezone(),
            self._config("emotion_view_theme", "自动"),
        )

    async def _settle_text(self, event: AstrMessageEvent) -> str | None:
        if not self._is_private(event):
            return None
        ledger = await self.service.settle_now(self._user_key(event))
        return f"已完成确定性结算。\n{self._format_state(ledger)}"

    async def _rebuild_text(self, event: AstrMessageEvent) -> str | None:
        if not self._is_private(event):
            return None
        key = self._user_key(event)
        ledger = await self.service.get(key)
        return build_snapshot(
            ledger,
            self._int_config("max_injected_events", 2),
            self._attention_injection_limit(),
        )

    async def _diary_rerun_text(self, event: AstrMessageEvent) -> str | None:
        if not self._is_private(event):
            return None
        boundary = parse_boundary(str(self._config("daily_settlement_time", "07:00")))
        cycle_date = logical_day(datetime.now(), boundary).isoformat()
        ledger, provider_id = await self._run_daily_review(
            self._user_key(event), cycle_date, force=True
        )
        diary = ledger.diaries[-1] if ledger.diaries else None
        text = diary.diary if diary else "未生成日记"
        return f"已重跑 {cycle_date} 每日回顾。\nProvider：{provider_id}\n\n{text}"

    async def _provider_chain_text(self, event: AstrMessageEvent) -> str | None:
        if not self._is_private(event):
            return None
        key = self._user_key(event)
        lines = []
        for task, label in (("review", "低频复核"), ("daily", "每日深度回顾")):
            configured = self.gateway.configured_ids(task)
            resolved = [
                self.gateway.provider_id(item)
                for item in self.gateway.providers(key, task)
            ]
            lines.append(
                f"{label}配置："
                + (" → ".join(configured) if configured else "当前会话模型")
            )
            lines.append(
                f"{label}可用：" + (" → ".join(resolved) if resolved else "无")
            )
        return "\n".join(lines)

    async def _rule_test_text(
        self, event: AstrMessageEvent, sample: str = ""
    ) -> str | None:
        if not self._is_private(event):
            return None
        text = sample.strip() or str(event.message_str or "")
        return json.dumps(self.rules.run(text).to_dict(), ensure_ascii=False, indent=2)

    async def _cleanup_transient_text(
        self, event: AstrMessageEvent, mode: str = ""
    ) -> str | None:
        if not self._is_private(event):
            return None
        key = self._user_key(event)
        ledger = await self.service.get(key)
        candidates = [item for item in ledger.events if is_legacy_transient_event(item)]
        if mode.strip() not in {"执行", "apply", "confirm"}:
            preview = "\n".join(f"- {item.fact[:120]}" for item in candidates[:20])
            return (
                f"可归档的旧瞬时事件：{len(candidates)} 条。\n"
                + (preview or "没有符合条件的记录。")
                + "\n确认后使用：/情绪管理 清理瞬时 执行"
            )

        archived_ids: list[str] = []

        def archive(current: StateLedger) -> StateLedger:
            nonlocal archived_ids
            updated, archived_ids = archive_legacy_transient_events(current)
            return updated

        _, changed = await self.service.mutate_if_changed(
            key,
            "archive_transient_migration",
            archive,
            {"candidate_count": len(candidates)},
        )
        return f"已归档 {len(archived_ids) if changed else 0} 条旧瞬时事件。"

    @emotion_group.command("注入", alias={"prompt"})
    async def group_injection(self, event: AstrMessageEvent, mode: str = ""):
        result = await self._injection_text(event, mode)
        if result is not None:
            yield event.plain_result(result)

    @filter.command(
        "情绪注入",
        alias={"心境注入", "emotion prompt"},
        desc="默认查看最近一次实际发送副本；传入“预览”查看当前账本的只读未发送预览。",
    )
    async def cmd_injection(self, event: AstrMessageEvent, mode: str = ""):
        result = await self._injection_text(event, mode)
        if result is not None:
            yield event.plain_result(result)

    @emotion_group.command("状态", alias={"state"})
    async def group_state(self, event: AstrMessageEvent):
        result = await self._state_text(event)
        if result is not None:
            yield event.plain_result(result)

    @emotion_group.command("查看", alias={"view"})
    async def group_emotion_view(self, event: AstrMessageEvent):
        try:
            png = await self._emotion_view_png(event)
            if png is not None:
                yield event.chain_result([Image.fromBytes(png)])
        except Exception as exc:  # noqa: BLE001
            logger.error("[EmotionState] emotion view render failed: %s", exc)
            yield event.plain_result("情绪图片生成失败，请稍后重试。")

    @filter.command(
        "情绪状态",
        alias={"心境", "emotion state"},
        desc="查看当前心境、有效心事和独立亲密状态。",
    )
    async def cmd_state(self, event: AstrMessageEvent):
        result = await self._state_text(event)
        if result is not None:
            yield event.plain_result(result)

    @filter.command(
        "情绪查看",
        alias={"emotion view"},
        desc="以本地渲染图片查看当前情绪、身体反应、心事和待关注事项。",
    )
    async def cmd_emotion_view(self, event: AstrMessageEvent):
        try:
            png = await self._emotion_view_png(event)
            if png is not None:
                yield event.chain_result([Image.fromBytes(png)])
        except Exception as exc:  # noqa: BLE001
            logger.error("[EmotionState] emotion view render failed: %s", exc)
            yield event.plain_result("情绪图片生成失败，请稍后重试。")

    @emotion_group.command("清理瞬时", alias={"cleanup-transient"})
    async def group_cleanup_transient(self, event: AstrMessageEvent, mode: str = ""):
        result = await self._cleanup_transient_text(event, mode)
        if result is not None:
            yield event.plain_result(result)

    @emotion_group.command("结算", alias={"settle"})
    async def group_settle(self, event: AstrMessageEvent):
        result = await self._settle_text(event)
        if result is not None:
            yield event.plain_result(result)

    @filter.command(
        "情绪结算",
        alias={"手动结算", "emotion settle"},
        desc="立即执行一次不调用模型的本地确定性衰减与结算。",
    )
    async def cmd_settle(self, event: AstrMessageEvent):
        result = await self._settle_text(event)
        if result is not None:
            yield event.plain_result(result)

    @emotion_group.command("重建", alias={"rebuild"})
    async def group_rebuild(self, event: AstrMessageEvent):
        result = await self._rebuild_text(event)
        if result is not None:
            yield event.plain_result(result)

    @filter.command(
        "情绪快照重建",
        alias={"重建情绪快照", "emotion rebuild"},
        desc="按当前账本重新生成情绪快照，不改动最近请求装配记录。",
    )
    async def cmd_rebuild(self, event: AstrMessageEvent):
        result = await self._rebuild_text(event)
        if result is not None:
            yield event.plain_result(result)

    @emotion_group.command("日记", alias={"diary"})
    async def group_diary_rerun(self, event: AstrMessageEvent):
        result = await self._diary_rerun_text(event)
        if result is not None:
            yield event.plain_result(result)

    @filter.command(
        "情绪日记重跑",
        alias={"日记重跑", "emotion diary rerun"},
        desc="使用每日深度回顾模型链强制重跑当前逻辑日回顾。",
    )
    async def cmd_diary_rerun(self, event: AstrMessageEvent):
        result = await self._diary_rerun_text(event)
        if result is not None:
            yield event.plain_result(result)

    @emotion_group.command("模型链", alias={"providers"})
    async def group_provider_chain(self, event: AstrMessageEvent):
        result = await self._provider_chain_text(event)
        if result is not None:
            yield event.plain_result(result)

    @filter.command(
        "情绪模型链",
        alias={"心境模型链", "emotion providers"},
        desc="检查低频复核链和每日深度回顾链的配置与可用 Provider。",
    )
    async def cmd_provider_chain(self, event: AstrMessageEvent):
        result = await self._provider_chain_text(event)
        if result is not None:
            yield event.plain_result(result)

    @emotion_group.command("规则试跑", alias={"rule-test"})
    async def group_rule_test(self, event: AstrMessageEvent, text: str = ""):
        result = await self._rule_test_text(event, text)
        if result is not None:
            yield event.plain_result(result)

    @filter.command(
        "情绪规则试跑",
        alias={"规则试跑", "emotion rule test"},
        desc="测试样例文本的规则命中结果，不写入真实状态。",
    )
    async def cmd_rule_test(self, event: AstrMessageEvent, text: str = ""):
        result = await self._rule_test_text(event, text)
        if result is not None:
            yield event.plain_result(result)

    def _format_state(self, ledger: StateLedger) -> str:
        events = select_injected_events(
            ledger.events, self._int_config("max_injected_events", 2)
        )
        max_attention_items = self._attention_injection_limit()
        selected_attention = select_attention_items(
            ledger.attention_items,
            max_attention_items,
            include_proposed=False,
        )
        selected_ids = {item.id for item in selected_attention}
        all_attention = [
            item
            for item in ledger.attention_items
            if item.status in {"proposed", "open"}
        ]
        lines = [
            f"当前心境：{ledger.mood.label}",
            f"心境数值：偏向 {ledger.mood.valence:+.2f}，能量 {ledger.mood.energy:.2f}，紧张 {ledger.mood.tension:.2f}",
            f"状态版本：{ledger.state_version}，消息水位：{ledger.message_watermark}",
            "当前事情：",
        ]
        lines.extend(
            f"- {item.fact}（{item.category}/{item.lifecycle}，{item.intensity:.2f}）"
            for item in events
        )
        attention_items = all_attention
        lines.append(f"后台未终结待关注事项（共 {len(attention_items)} 条）：")
        lines.extend(
            "- "
            f"[{attention_kind_label(item.kind)}/{attention_status_label(item.status)}] "
            f"{item.content}"
            + (
                f"（{item.time_hint or item.due_at}）"
                if item.time_hint or item.due_at
                else ""
            )
            + ("（已到时间但尚无完成证据）" if is_attention_overdue(item) else "")
            for item in attention_items
        )
        if not attention_items:
            lines.append("- 暂无")
        lines.append(
            f"本次模型提示词实际最多注入 {max_attention_items} 条，当前选中 {len(selected_ids)} 条："
        )
        lines.extend(f"- {item.content}" for item in selected_attention)
        if not selected_attention:
            lines.append("- 暂无")
        lines.append(
            f"吃醋状态：强度 {ledger.jealousy.intensity:.2f}，"
            f"置信度 {ledger.jealousy.confidence:.2f}，"
            f"最近证据 {ledger.jealousy.last_evidence_at or '无'}"
        )
        stage = body_reaction_stage(
            ledger.intimacy.body_sensitivity,
            ledger.intimacy.sexual_arousal,
        )
        tier = persona_intimacy_tier_label(
            self._config("persona_intimacy_tier", "很亲密")
        )
        lines.append(f"人设亲密基线：{tier}")
        lines.append(f"当前身体反应：{intimacy_stage_label(stage)}")
        lines.append(
            "身体反应数值："
            f"身体敏感度 {ledger.intimacy.body_sensitivity:.2f}，"
            f"性唤起 {ledger.intimacy.sexual_arousal:.2f}"
        )
        return "\n".join(lines)

    async def _settlement_loop(self) -> None:
        boundary = parse_boundary(str(self._config("daily_settlement_time", "07:00")))
        while True:
            try:
                now = datetime.now()
                current_day = logical_day(now, boundary)
                for user_key in await asyncio.to_thread(self.store.user_keys):
                    ledger = await self.service.get(user_key)
                    if not ledger.logical_day:
                        await self.service.mutate(
                            user_key,
                            "logical_day_initialized",
                            lambda current: self._set_logical_day(
                                current, current_day.isoformat()
                            ),
                        )
                        continue
                    if ledger.logical_day == current_day.isoformat():
                        continue
                    await self._run_daily_review(
                        user_key, ledger.logical_day, force=False
                    )
                    await self.service.mutate(
                        user_key,
                        "logical_day_advanced",
                        lambda current: self._set_logical_day(
                            current, current_day.isoformat()
                        ),
                    )
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[EmotionState] settlement loop failed: %s", exc)
                await asyncio.sleep(60)

    @staticmethod
    def _set_logical_day(ledger: StateLedger, value: str) -> StateLedger:
        ledger.logical_day = value
        return ledger

    def _register_web_apis(self) -> None:
        register = getattr(self.context, "register_web_api", None)
        if not callable(register):
            return
        register(
            f"/{PLUGIN_NAME}/page/sessions",
            self.sessions_api,
            ["GET"],
            "Emotion private sessions",
        )
        register(
            f"/{PLUGIN_NAME}/page/state",
            self.state_api,
            ["GET"],
            "Emotion state",
        )
        register(
            f"/{PLUGIN_NAME}/page/rules/test",
            self.rules_api,
            ["POST"],
            "Emotion rule test",
        )
        register(
            f"/{PLUGIN_NAME}/page/injection",
            self.injection_api,
            ["GET"],
            "Emotion injection preview",
        )
        register(
            f"/{PLUGIN_NAME}/page/items/delete",
            self.delete_item_api,
            ["POST"],
            "Archive an emotion state item",
        )

    @staticmethod
    def _api_user_key() -> str:
        return str(request.args.get("session_id", "")).strip()

    async def sessions_api(self):
        try:
            sessions = await asyncio.to_thread(self.store.sessions)
        except (OSError, ValueError, TypeError) as exc:
            logger.error("[EmotionState] session enumeration failed: %s", exc)
            return {"status": "error", "message": "私聊会话读取失败，请查看服务日志"}
        return {"status": "ok", "data": {"sessions": sessions}}

    async def state_api(self):
        user_key = self._api_user_key()
        if not user_key:
            return {"status": "error", "message": "session_id is required"}
        ledger = await self._settle_proactive_state(user_key)
        proactive = await self._spark_snapshot(user_key)
        proactive_ts = float(proactive.get("last_proactive_reply_ts", 0.0) or 0.0)
        waiting_seconds = (
            max(0.0, datetime.now().astimezone().timestamp() - proactive_ts)
            if proactive.get("awaiting_user_reply") and proactive_ts > 0
            else 0.0
        )
        review_chain = self.gateway.configured_ids("review")
        daily_chain = self.gateway.configured_ids("daily")
        actual = self._actual_injection_snapshot(user_key)
        event_limit = self._int_config("max_injected_events", 2)
        selected_events, event_exclusions = select_injected_events_with_reasons(
            ledger.events, event_limit
        )
        attention_limit = self._attention_injection_limit()
        selected_attention = select_attention_items(
            ledger.attention_items,
            attention_limit,
            include_proposed=False,
        )
        stage = body_reaction_stage(
            ledger.intimacy.body_sensitivity,
            ledger.intimacy.sexual_arousal,
        )
        tier = persona_intimacy_tier_label(
            self._config("persona_intimacy_tier", "很亲密")
        )
        return {
            "status": "ok",
            "data": {
                "ledger": ledger.to_dict(),
                "presentation": {
                    "persona_intimacy_tier": tier,
                    "body_reaction_stage": intimacy_stage_label(stage),
                    "intimacy_stage": intimacy_stage_label(stage),
                    "attention_items": [
                        {
                            **item.review_view(),
                            "kind_label": attention_kind_label(item.kind),
                            "status_label": attention_status_label(item.status),
                            "overdue": is_attention_overdue(item),
                        }
                        for item in reversed(ledger.attention_items)
                        if item.status in {"proposed", "open"}
                    ],
                    "all_open_attention_items": [
                        item.to_dict()
                        for item in ledger.attention_items
                        if item.status in {"proposed", "open"}
                    ],
                    "selected_injection_attention_items": [
                        item.to_dict() for item in selected_attention
                    ],
                    "selected_injection_attention_count": len(selected_attention),
                    "max_injected_attention_items": attention_limit,
                },
                "diagnostics": {
                    "busy_schedule": callable(
                        getattr(self.context, "_busy_schedule_get_facts", None)
                    ),
                    "livingmemory": callable(
                        getattr(self.context, "_livingmemory_get_daily_context", None)
                    ),
                    "spark": bool(proactive.get("available")),
                    "spark_awaiting_user_reply": bool(
                        proactive.get("awaiting_user_reply")
                    ),
                    "spark_waiting_minutes": round(waiting_seconds / 60.0, 1),
                    "spark_applied_stage": ledger.proactive_applied_stage,
                    "review_provider_chain": review_chain
                    or ["current_session_provider"],
                    "daily_provider_chain": daily_chain or ["current_session_provider"],
                    "selected_injection_event_count": len(selected_events),
                    "max_injected_events": event_limit,
                    "injection_event_evaluations": [
                        {
                            "id": event.id,
                            "fingerprint": event.fingerprint,
                            "category": event.category,
                            "lifecycle": event.lifecycle,
                            "intensity": event.intensity,
                            "confidence": event.confidence,
                            "created_at": event.created_at,
                            "updated_at": event.updated_at,
                            "last_stimulated_at": event.last_stimulated_at,
                            "occurrence_count": event.occurrence_count,
                            "selected": event.id not in event_exclusions,
                            "exclusion_reason": event_exclusions.get(event.id, ""),
                        }
                        for event in ledger.events
                    ],
                    "last_actual_prompt_available": bool(actual and actual.prompt),
                    "last_actual_prompt_state_version": (
                        actual.state_version if actual else None
                    ),
                    "last_actual_prompt_generated_at": (
                        actual.generated_at if actual else ""
                    ),
                    "last_actual_prompt_source": (
                        actual.request_source if actual else ""
                    ),
                    "last_actual_prompt_stale": bool(
                        actual
                        and actual.state_version >= 0
                        and actual.state_version != ledger.state_version
                    ),
                },
            },
        }

    async def delete_item_api(self):
        payload = await request.get_json(silent=True) or {}
        user_key = str(payload.get("session_id", "")).strip()
        kind = str(payload.get("kind", "")).strip().lower()
        item_id = str(payload.get("id", "")).strip()
        if not user_key:
            return {"status": "error", "message": "session_id is required"}
        if kind not in {"event", "attention"}:
            return {"status": "error", "message": "kind must be event or attention"}
        ledger, changed, reason = await self.service.delete_item(
            user_key,
            kind,
            item_id,
        )
        if not changed:
            messages = {
                "invalid_item_reference": "对象 ID 无效",
                "item_not_found": "对象不存在，可能已被清理",
                "item_already_archived": "对象已经归档",
            }
            return {
                "status": "error",
                "message": messages.get(reason, f"删除失败：{reason}"),
                "reason": reason,
                "state_version": ledger.state_version,
            }
        return {
            "status": "ok",
            "data": {
                "kind": kind,
                "id": item_id,
                "status": reason,
                "state_version": ledger.state_version,
                "mood": ledger.to_dict()["mood"],
            },
        }

    async def rules_api(self):
        payload = await request.get_json(silent=True) or {}
        sample = str(payload.get("text", ""))
        return {
            "status": "ok",
            "data": {
                "run": self.rules.run(sample).to_dict(),
                "errors": self.rules.validate(),
            },
        }

    async def injection_api(self):
        user_key = self._api_user_key()
        if not user_key:
            return {"status": "error", "message": "session_id is required"}
        ledger = await self.service.get(user_key)
        actual = self._actual_injection_snapshot(user_key)
        preview = self._preview_injection_snapshot(ledger)
        actual_payload = actual.to_dict() if actual else None
        actual_available = bool(actual and actual.prompt)
        if actual_payload:
            actual_payload["source"] = "last_actual_request"
            actual_payload["actual_available"] = actual_available
            actual_payload["preview_only"] = False
            actual_payload["stale"] = bool(
                actual.state_version >= 0
                and actual.state_version != ledger.state_version
            )
        preview_payload = preview.to_dict()
        preview_payload["source"] = "current_ledger_preview"
        preview_payload["actual_available"] = False
        preview_payload["preview_only"] = True
        preview_payload["stale"] = False
        return {
            "status": "ok",
            "data": {
                "kind": "actual" if actual_available else "none",
                "prompt": actual.prompt if actual_available else "",
                "source": "last_actual_request" if actual_available else "none",
                "actual_available": actual_available,
                "preview_only": not actual_available,
                "current_state_version": ledger.state_version,
                "actual": actual_payload,
                "preview": preview_payload,
            },
        }
