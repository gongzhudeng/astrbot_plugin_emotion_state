"""Provider selection and ordered fallback for model-assisted tasks."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any, Literal

from astrbot.api import logger

try:
    from astrbot.core.provider.provider import Provider
except ImportError:
    Provider = object  # type: ignore[misc,assignment]


ProviderTask = Literal["review", "daily", "guidance", "life_event"]

# Tasks that fall back to the review chain when their own chain is not configured.
_REVIEW_FALLBACK_TASKS = ("guidance", "life_event")


class ProviderGateway:
    def __init__(self, context: Any, config: Any) -> None:
        self.context = context
        self.config = config

    def _ids(self, key: str) -> list[str]:
        value = self.config.get(key, []) if hasattr(self.config, "get") else []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value or [] if str(item).strip()]

    def _has_key(self, key: str) -> bool:
        try:
            return key in self.config
        except (TypeError, AttributeError):
            return self.config.get(key, None) is not None

    def configured_ids(self, task: ProviderTask = "review") -> list[str]:
        key = f"{task}_provider_ids"
        if task in _REVIEW_FALLBACK_TASKS:
            # New tasks ride the same ordered chain mechanic: own chain first,
            # then the review chain, then (in providers()) the session model.
            if self._has_key(key):
                own = self._ids(key)
                if own:
                    return own
            return self.configured_ids("review")
        if self._has_key(key):
            return self._ids(key)

        unified = self._ids("analysis_provider_ids")
        if unified:
            return unified

        # Read legacy split fields only when no current model chain has been saved.
        primary = str(self.config.get("analysis_provider_id", "") or "").strip()
        fallback = self._ids("analysis_fallback_provider_ids")
        return [item for item in [primary, *fallback] if item]

    def _configured_ids(self) -> list[str]:
        """Keep the previous review-chain accessor available to integrations."""
        return self.configured_ids("review")

    def providers(
        self, session_id: str = "", task: ProviderTask = "review"
    ) -> list[Any]:
        result: list[Any] = []
        configured_ids = self.configured_ids(task)
        for provider_id in configured_ids:
            try:
                provider = self.context.get_provider_by_id(provider_id)
            except Exception as exc:
                logger.warning(
                    "[EmotionState] Provider %s is unavailable: %s",
                    provider_id,
                    exc,
                )
                continue
            if provider is not None:
                result.append(provider)
        if not configured_ids and session_id:
            try:
                provider = self.context.get_using_provider(session_id)
            except TypeError:
                provider = self.context.get_using_provider()
            if provider is not None:
                result.append(provider)
        deduped: list[Any] = []
        seen: set[int] = set()
        for provider in result:
            if id(provider) not in seen:
                seen.add(id(provider))
                deduped.append(provider)
        return deduped

    @staticmethod
    def provider_id(provider: Any) -> str:
        try:
            return str(provider.meta().id)
        except Exception:
            return type(provider).__name__

    def _model_limits(self, task: ProviderTask) -> tuple[int, float]:
        if task == "daily":
            retries_key, timeout_key = (
                "daily_model_max_retries",
                "daily_model_timeout_seconds",
            )
            default_retries, default_timeout = 1, 180.0
        else:
            # review / guidance / life_event share the review call limits.
            retries_key, timeout_key = (
                "review_model_max_retries",
                "review_model_timeout_seconds",
            )
            default_retries, default_timeout = 0, 30.0
        try:
            retries = int(self.config.get(retries_key, default_retries))
        except (TypeError, ValueError):
            retries = default_retries
        try:
            timeout = float(self.config.get(timeout_key, default_timeout))
        except (TypeError, ValueError):
            timeout = default_timeout
        return max(0, min(10, retries)), max(1.0, min(3600.0, timeout))

    @staticmethod
    def _supports_request_max_retries(provider: Any) -> bool:
        target = getattr(provider, "text_chat", None)
        side_effect = getattr(target, "side_effect", None)
        if callable(side_effect):
            target = side_effect
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return True
        return "request_max_retries" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    async def complete(
        self,
        prompt: str,
        session_id: str = "",
        system_prompt: str = "",
        contexts: list[dict[str, Any]] | None = None,
        task: ProviderTask = "review",
        validate: Callable[[str], Any] | None = None,
    ) -> tuple[str, str]:
        providers = self.providers(session_id, task)
        if not providers:
            raise RuntimeError("没有可用的情绪分析 Provider")
        max_retries, timeout_seconds = self._model_limits(task)
        max_attempts = max_retries + 1
        last_error = "模型返回空内容"
        for index, provider in enumerate(providers):
            provider_id = self.provider_id(provider)
            for attempt in range(max_attempts):
                started = time.perf_counter()
                try:
                    kwargs: dict[str, Any] = (
                        {"contexts": contexts} if contexts else {"prompt": prompt}
                    )
                    if not contexts:
                        kwargs.update(
                            {
                                "session_id": session_id,
                                "system_prompt": system_prompt or None,
                            }
                        )
                    if self._supports_request_max_retries(provider):
                        kwargs["request_max_retries"] = 1
                    response = await asyncio.wait_for(
                        provider.text_chat(**kwargs),
                        timeout=timeout_seconds,
                    )
                    text = str(getattr(response, "completion_text", "") or "").strip()
                    if not text:
                        last_error = f"{provider_id} 返回空内容"
                        logger.warning(
                            "[EmotionState] Provider %s returned empty output "
                            "(attempt %s/%s)",
                            provider_id,
                            attempt + 1,
                            max_attempts,
                        )
                        continue
                    if validate is not None:
                        try:
                            validate(text)
                        except Exception as exc:
                            last_error = f"{provider_id}: {exc}"
                            logger.warning(
                                "[EmotionState] Provider %s returned invalid task output; switching provider: %s",
                                provider_id,
                                exc,
                            )
                            break
                    logger.info(
                        "[EmotionState] Provider %s succeeded for %s (attempt %s/%s, %.1fs)",
                        provider_id,
                        task,
                        attempt + 1,
                        max_attempts,
                        time.perf_counter() - started,
                    )
                    return text, provider_id
                except asyncio.TimeoutError:
                    last_error = f"{provider_id} 超时（{timeout_seconds:.1f}s）"
                    logger.warning(
                        "[EmotionState] Provider %s timed out for %s (attempt %s/%s, limit=%.1fs)",
                        provider_id,
                        task,
                        attempt + 1,
                        max_attempts,
                        timeout_seconds,
                    )
                except Exception as exc:
                    last_error = f"{provider_id}: {exc}"
                    logger.warning(
                        "[EmotionState] Provider %s failed for %s (attempt %s/%s): %s",
                        provider_id,
                        task,
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                if attempt < max_attempts - 1:
                    logger.info(
                        "[EmotionState] Retrying provider %s for %s immediately",
                        provider_id,
                        task,
                    )
        raise RuntimeError(f"所有情绪模型均失败: {last_error}")
