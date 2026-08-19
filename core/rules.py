"""Safe local rule DSL for producing evidence, not direct long-term facts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .models import EventObservation
from .text_limits import EVENT_FACT_STORAGE_CHARS, bound_complete_text
from .settlement import infer_attack_target, strip_media_context

DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "id": "positive",
        "words": ["开心", "高兴", "好喜欢", "谢谢你", "被治愈"],
        "valence": 0.7,
        "intensity": 0.5,
        "meaning": "这件事让人感到被在意或愉快",
    },
    {
        "id": "negative",
        "words": ["难过", "委屈", "失望", "烦死了", "不舒服", "生气"],
        "valence": -0.7,
        "intensity": 0.6,
        "meaning": "这件事带来明显的不舒服或负担",
    },
    {
        "id": "repair",
        "words": ["对不起", "抱歉", "我错了", "别难过", "抱抱"],
        "valence": 0.2,
        "intensity": 0.45,
        "meaning": "对方正在尝试修复或安慰这段互动",
    },
    {
        "id": "flirt",
        "words": ["想你", "喜欢你", "亲一下", "抱一下", "想和你"],
        "valence": 0.65,
        "intensity": 0.5,
        "meaning": "互动中出现亲近或暧昧倾向",
    },
    {
        "id": "teasing",
        "words": ["笨蛋", "傻瓜", "哼", "才不告诉你"],
        "valence": 0.25,
        "intensity": 0.25,
        "meaning": "更像熟悉关系中的玩笑或撒娇，需要结合上下文",
    },
    {
        "id": "abuse",
        "words": ["滚开", "去死", "恶心", "废物"],
        "valence": -0.8,
        "intensity": 0.72,
        "meaning": "可能构成真实攻击，需要结合对象和引用关系复核",
    },
]


@dataclass(slots=True)
class RuleMatch:
    rule_id: str
    matched: str
    reason: str
    excluded: bool = False
    exclusion_reason: str = ""
    observation: EventObservation | None = None


@dataclass(slots=True)
class RuleRun:
    matches: list[RuleMatch] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    candidates: list[EventObservation] = field(default_factory=list)
    transient_signals: list[EventObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": [
                {
                    "rule_id": item.rule_id,
                    "matched": item.matched,
                    "reason": item.reason,
                    "excluded": item.excluded,
                    "exclusion_reason": item.exclusion_reason,
                }
                for item in self.matches
            ],
            "exclusions": self.exclusions,
            "transient_signals": [asdict(item) for item in self.transient_signals],
            "candidates": [asdict(item) for item in self.candidates],
        }


class LocalRuleEngine:
    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        self.rules = self._compile_rules([*DEFAULT_RULES, *(rules or [])])

    @staticmethod
    def _compile_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compiled = []
        for index, raw in enumerate(rules):
            if not isinstance(raw, dict) or raw.get("enabled", True) is False:
                continue
            item = dict(raw)
            item["id"] = str(item.get("id") or f"custom_{index}")
            item["priority"] = int(item.get("priority", 0))
            item["match_type"] = str(item.get("match_type", "literal"))
            item["words"] = [
                str(value) for value in item.get("words", []) if str(value)
            ]
            if not item["words"] and not item.get("pattern"):
                continue
            if item["match_type"] == "regex":
                try:
                    item["compiled"] = re.compile(str(item.get("pattern") or ""), re.I)
                except re.error as exc:
                    item["compile_error"] = str(exc)
            compiled.append(item)
        return sorted(compiled, key=lambda value: value["priority"], reverse=True)

    @staticmethod
    def _excluded(text: str, rule: dict[str, Any], context: str) -> str:
        lowered = text.lower()
        if "```" in text or re.search(
            r"(^|\n)\s*(日志|系统消息|assistant:|bot:)", lowered
        ):
            return "代码、日志或系统/Bot文本"
        if rule["id"] == "abuse" and context in {"quoted", "third_party", "bot"}:
            return f"攻击性表达上下文标记为 {context}"
        if any(str(word).lower() in lowered for word in rule.get("exclude_words", [])):
            return "命中排除词"
        return ""

    @staticmethod
    def _has_concrete_context(text: str, matched: str) -> bool:
        """A short emotional phrase is transient unless user-authored text has a concrete cue."""
        semantic_text = strip_media_context(text)
        remainder = semantic_text.replace(matched, "", 1).strip()
        concrete_cues = (
            r"因为|由于|记得|生日|礼物|答应|约定|帮我|陪我|会议|工作|"
            r"考试|项目|见面|聊天|同事|朋友|家人|回复|消息"
        )
        return bool(remainder and re.search(concrete_cues, remainder, re.I))

    @staticmethod
    def _category_for(rule: dict[str, Any], text: str, matched: str) -> str:
        configured = str(rule.get("category", "")).strip().lower()
        if configured in {"transient", "psychological", "concrete"}:
            return configured
        has_concrete_context = LocalRuleEngine._has_concrete_context(text, matched)
        if rule["id"] in {"flirt", "positive", "repair", "teasing"}:
            return "concrete" if has_concrete_context else "transient"
        return "concrete"

    def run(self, text: str, *, context: str = "user", watermark: int = 0) -> RuleRun:
        result = RuleRun()
        clean = str(text or "").strip()
        semantic_text = strip_media_context(clean)
        if not clean:
            result.exclusions.append("空消息")
            return result
        if not semantic_text:
            result.exclusions.append("没有可用于情绪判断的用户正文")
            return result
        for rule in self.rules:
            if rule.get("compile_error"):
                result.exclusions.append(
                    f"规则 {rule['id']} 正则无效: {rule['compile_error']}"
                )
                continue
            matched = None
            if rule["match_type"] == "regex" and rule.get("compiled"):
                found = rule["compiled"].search(semantic_text)
                matched = found.group(0) if found else None
            else:
                matched = next(
                    (
                        word
                        for word in rule["words"]
                        if word.lower() in semantic_text.lower()
                    ),
                    None,
                )
            if not matched:
                continue
            exclusion = self._excluded(semantic_text, rule, context)
            if exclusion:
                result.matches.append(
                    RuleMatch(rule["id"], matched, "命中但被排除", True, exclusion)
                )
                result.exclusions.append(f"{rule['id']}: {exclusion}")
                continue
            valence = float(rule.get("valence", 0.0))
            intensity = float(rule.get("intensity", 0.35))
            confidence = float(
                rule.get("confidence", 0.58 if rule["id"] == "teasing" else 0.7)
            )
            category = self._category_for(rule, semantic_text, matched)
            configured_target = str(rule.get("target", "")).strip()
            target = configured_target or "user"
            target_basis = "configured_rule_target" if configured_target else ""
            if rule["id"] == "abuse" and not configured_target:
                target, target_basis = infer_attack_target(semantic_text)
                if target != "user":
                    category = "transient"
            observation = EventObservation(
                action=str(rule.get("action", "create")),
                fact=bound_complete_text(semantic_text, EVENT_FACT_STORAGE_CHARS),
                emotional_meaning=str(
                    rule.get("meaning") or "这段互动可能影响当前心境"
                ),
                target=target,
                target_basis=target_basis,
                evidence_quote=semantic_text[:240],
                evidence_speaker="user",
                category=category,
                valence=valence,
                intensity=intensity,
                confidence=confidence,
                source="local_rule",
                message_watermark=watermark,
                uncertain=confidence < 0.62,
                note=f"rule={rule['id']}; matched={matched}",
                tags=[rule["id"]],
            )
            result.matches.append(
                RuleMatch(
                    rule["id"],
                    matched,
                    "产生瞬时信号" if category == "transient" else "产生候选证据",
                    observation=observation,
                )
            )
            if category == "transient":
                result.transient_signals.append(observation)
            else:
                result.candidates.append(observation)
        return result

    def validate(self) -> list[dict[str, str]]:
        return [
            {"id": item["id"], "error": item["compile_error"]}
            for item in self.rules
            if item.get("compile_error")
        ]
