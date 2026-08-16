"""Provider selection and ordered fallback for model-assisted tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from astrbot.api import logger

try:
    from astrbot.core.provider.provider import Provider
except ImportError:
    Provider = object  # type: ignore[misc,assignment]


ProviderTask = Literal["review", "daily"]


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
        key = "daily_provider_ids" if task == "daily" else "review_provider_ids"
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
        last_error = "模型返回空内容"
        for index, provider in enumerate(providers):
            provider_id = self.provider_id(provider)
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
                response = await provider.text_chat(**kwargs)
                text = str(getattr(response, "completion_text", "") or "").strip()
                if not text:
                    last_error = f"{provider_id} 返回空内容"
                    continue
                if validate is not None:
                    validate(text)
                return text, provider_id
            except Exception as exc:
                last_error = f"{provider_id}: {exc}"
                logger.warning(
                    "[EmotionState] Provider %s failed (%s/%s): %s",
                    provider_id,
                    index + 1,
                    len(providers),
                    exc,
                )
        raise RuntimeError(f"所有情绪模型均失败: {last_error}")
