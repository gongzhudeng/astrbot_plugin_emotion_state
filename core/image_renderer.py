"""Glassmorphism Pillow renderer for the emotion view command (E6)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import AttentionItem, InnerEvent, StateLedger, parse_time
from .style_kit import Canvas, c, font


class EmotionStateImageRenderer:
    """Render one long-form mood sheet without browser dependencies."""

    width = 1080
    day_start_hour = 7
    night_start_hour = 19

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = Path(plugin_dir)

    def resolve_theme(self, mode: object, now: datetime) -> str:
        normalized = str(mode or "").strip().casefold()
        if normalized in {"亮色", "白天", "白天模式", "day", "light"}:
            return "day"
        if normalized in {"暗色", "夜间", "夜间模式", "night", "dark"}:
            return "night"
        return (
            "day"
            if self.day_start_hour <= now.hour < self.night_start_hour
            else "night"
        )

    def render(
        self,
        ledger: StateLedger,
        events: Sequence[InnerEvent],
        attention_items: Sequence[AttentionItem],
        body_stage: str,
        now: datetime,
        mode: object = "自动",
    ) -> bytes:
        """Render the mood sheet as encoded PNG bytes.

        Args:
            ledger: Persisted emotional state (mood and intimacy values).
            events: Inner events still echoing in the mind.
            attention_items: Items that still need follow-up.
            body_stage: Raw body reaction stage label.
            now: Current local time used for highlighting and theme selection.
            mode: Automatic, light, or dark display mode.

        Returns:
            Encoded RGB PNG bytes.
        """
        theme = self.resolve_theme(mode, now)
        return self._render(ledger, events, attention_items, body_stage, now, theme)

    # -- palette -----------------------------------------------------------
    @staticmethod
    def _palette(theme: str) -> dict[str, Any]:
        if theme == "night":
            return {
                "stops": [(0, c("#1E1420")), (0.5, c("#241626")), (1, c("#141222"))],
                "glows": [
                    ("#A85474", 880, 300, 300, 80),
                    ("#6A54A0", 140, 1300, 300, 55),
                    ("#4A7A62", 900, 1900, 300, 45),
                ],
                "ink": "#F2E4EC",
                "sub": "#A892A2",
                "rose": "#E89CB4",
                "lavender": "#AC9EE8",
                "mint": "#7CC9A2",
                "gold": "#D8BC80",
                "blue": "#9BA8DC",
                "tint": (56, 40, 60),
                "talpha": 135,
                "line": (255, 255, 255, 46),
                "border": (255, 255, 255, 44),
                "inner": (224, 160, 180, 40),
                "shadow_a": 95,
            }
        return {
            "stops": [(0, c("#FDEEF3")), (0.5, c("#F8E9F4")), (1, c("#EFF0FA"))],
            "glows": [
                ("#F2B4C8", 880, 300, 300, 90),
                ("#C4B0EC", 140, 1300, 300, 60),
                ("#A8D8C0", 900, 1900, 300, 50),
            ],
            "ink": "#4A3644",
            "sub": "#978092",
            "rose": "#D67A96",
            "lavender": "#A08CD8",
            "mint": "#6FBF97",
            "gold": "#C9A050",
            "blue": "#91A9DD",
            "tint": (255, 255, 255),
            "talpha": 135,
            "line": (255, 255, 255, 240),
            "border": (255, 255, 255, 255),
            "inner": (214, 134, 158, 44),
            "shadow_a": 40,
        }

    # -- label helpers (kept from the previous renderer) ----------------------
    @staticmethod
    def _target_label(target: str) -> str:
        normalized = str(target or "").strip()
        if normalized.casefold() in {"", "unknown", "none", "未明确"}:
            return ""
        if normalized.casefold() in {
            "user",
            "current_user",
            "current chat user",
            "当前用户",
            "当前聊天对象",
        }:
            return "当前聊天对象"
        return normalized[:16]

    @staticmethod
    def _kind_label(kind: str) -> str:
        return {
            "commitment": "约定",
            "plan": "计划",
            "remember": "记住",
            "follow_up": "后续跟进",
        }.get(str(kind), "待关注")

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "proposed": "待确认",
            "open": "仍待关注",
        }.get(str(status), "仍待关注")

    @staticmethod
    def _body_tier_label(stage: str) -> str:
        value = str(stage or "")
        if "强烈" in value or value == "open_and_receptive":
            return "强烈"
        if "明显" in value or value == "warmly_receptive":
            return "明显"
        if "轻微" in value or value == "slightly_aware":
            return "轻微"
        return "平静"

    @staticmethod
    def _event_time(event: InnerEvent, now: datetime) -> str:
        try:
            value = parse_time(event.last_stimulated_at or event.updated_at)
            if now.tzinfo is not None:
                value = value.astimezone(now.tzinfo)
            return f"{value:%m 月 %d 日 · %H:%M}"
        except (TypeError, ValueError, OSError):
            return "最近更新"

    # -- drawing helpers -------------------------------------------------------
    def _card(self, cv: Canvas, box, pal, radius=28):
        cv.shadow(box, radius, 22, 12, alpha=pal["shadow_a"])
        cv.glass(
            box,
            radius=radius,
            tint=pal["tint"],
            alpha=pal["talpha"],
            outline=pal["line"],
            owidth=1.5,
        )

    def _event_metrics(self, cv: Canvas, event: InnerEvent):
        """Measure one event card: (fact_lines, meaning_lines, card_height)."""
        fact_f, mf = font(24, 450), font(22, 450)
        fact_lines = cv.wrap(event.fact, fact_f, 760)
        meaning_lines = (
            cv.wrap(event.emotional_meaning, mf, 740)
            if event.emotional_meaning.strip()
            else []
        )
        box_h = 54 + len(meaning_lines) * 32 if meaning_lines else 0
        fact_top = 96
        box_y = fact_top + len(fact_lines) * 36 + 16
        card_h = (
            box_y + box_h + 20
            if meaning_lines
            else fact_top + len(fact_lines) * 36 + 20
        )
        return fact_lines, meaning_lines, card_h

    def _attention_metrics(self, cv: Canvas, item: AttentionItem):
        content_f = font(25, 600)
        lines = cv.wrap(item.content, content_f, 760)
        return lines, 104 + len(lines) * 36

    # -- sections ---------------------------------------------------------------
    def _draw_header(self, cv: Canvas, pal, now: datetime):
        ink, sub, rose = pal["ink"], pal["sub"], pal["rose"]
        avatar = self.plugin_dir / "logo.png"
        cv.avatar(avatar, 540, 138, 110, c(rose), 3)
        cv.ring(540, 138, 80, c(rose, 110), 1.5)
        cv.text(
            540, 226, "内心世界 · 此刻心绪", font(40, 800, "serif"), ink, anchor="ma"
        )
        cv.text(
            540,
            288,
            f"{now:%Y 年 %m 月 %d 日 · %H:%M}",
            font(20, 450),
            sub,
            anchor="ma",
        )
        cv.star4(408, 148, 9, c(rose, 200))
        cv.star4(672, 130, 7, c(pal["lavender"], 200))

    def _draw_mood(self, cv: Canvas, top, ledger: StateLedger, pal):
        ink, sub = pal["ink"], pal["sub"]
        rose, lavender = pal["rose"], pal["lavender"]
        card_h = 300
        self._card(cv, (58, top, 1022, top + card_h), pal)
        valence = max(-1.0, min(1.0, ledger.mood.valence))
        score = int(round(valence * 100))
        disp = f"{score:+d}"
        ext = max(12, int(abs(valence) * 300))
        track = pal["border"]
        cv.ring(200, top + 150, 92, track, 9)
        gauge = (108, top + 58, 292, top + 242)
        cv.arc_gradient(gauge, -90, -90 + ext, c(rose), c(lavender), 9)
        cv.arc_caps(gauge, -90, -90 + ext, c(rose), 9)
        cv.text(200, top + 138, disp, font(40, 700, "num"), ink, anchor="mm")
        cv.text(200, top + 176, "情绪数值", font(16, 450), sub, anchor="ma")
        cv.text(372, top + 46, "此刻的心境", font(18, 450), sub)
        cv.text(372, top + 78, ledger.mood.label or "平静", font(46, 800), ink)
        values = [
            ("情绪偏向", abs(valence), disp, rose),
            (
                "能量",
                ledger.mood.energy,
                f"{round(ledger.mood.energy * 100)}",
                lavender,
            ),
            (
                "紧张",
                ledger.mood.tension,
                f"{round(ledger.mood.tension * 100)}",
                pal["mint"],
            ),
        ]
        y = top + 150
        for label, frac, text, col in values:
            cv.text(372, y + 2, label, font(17, 450), sub)
            cv.rrect((480, y + 8, 900, y + 16), 4, fill=track)
            cv.rrect(
                (480, y + 8, 480 + int(420 * max(0.0, min(1.0, frac))), y + 16),
                4,
                fill=c(col),
            )
            cv.text(938, y + 2, text, font(20, 700, "num"), ink, anchor="ra")
            y += 44
        return card_h

    def _draw_body(self, cv: Canvas, top, ledger: StateLedger, body_stage: str, pal):
        ink, sub = pal["ink"], pal["sub"]
        rose, lavender = pal["rose"], pal["lavender"]
        self._card(cv, (58, top, 1022, top + 104), pal, radius=26)
        cv.dot(116, top + 52, 34, c(rose, 40))
        cv.heart(116, top + 52, 28, c(rose))
        cv.text(176, top + 18, "身体反应", font(17, 450), sub)
        cv.text(176, top + 46, self._body_tier_label(body_stage), font(26, 700), ink)
        sens = round(ledger.intimacy.body_sensitivity * 100)
        arousal = round(ledger.intimacy.sexual_arousal * 100)
        chip_f = font(17, 600)
        box = cv.pill(
            988,
            top + 34,
            f"敏感度 {sens}",
            chip_f,
            rose,
            c(rose, 40),
            padx=15,
            pady=8,
            anchor="ra",
        )
        cv.pill(
            box[0] - 12,
            top + 34,
            f"唤起 {arousal}",
            chip_f,
            lavender,
            c(lavender, 40),
            padx=15,
            pady=8,
            anchor="ra",
        )
        return 104

    def _draw_section_head(self, cv: Canvas, y, title, count, accent, pal):
        ink = pal["ink"]
        cv.text(84, y, title, font(28, 700), ink)
        cv.pill(
            988,
            y - 4,
            count,
            font(17, 600),
            accent,
            c(accent, 44),
            padx=14,
            pady=7,
            anchor="ra",
        )
        return y + 56

    def _draw_events(self, cv: Canvas, y, events, pal, now: datetime):
        ink, sub = pal["ink"], pal["sub"]
        rose = pal["rose"]
        y = self._draw_section_head(
            cv, y, "仍在心里回响的事", f"{len(events)} 件", rose, pal
        )
        title_f, fact_f, mf = font(25, 700), font(24, 450), font(22, 450)
        if not events:
            cv.rrect(
                (58, y, 1022, y + 90),
                22,
                fill=(*pal["tint"], 60),
                outline=pal["line"],
                width=1.5,
            )
            cv.text(82, y + 29, "现在没有持续影响心境的事情", font(24, 450), sub)
            return y + 90 + 24
        for i, event in enumerate(events):
            fact_lines, meaning_lines, card_h = self._event_metrics(cv, event)
            self._card(cv, (58, y, 1022, y + card_h), pal, radius=26)
            cv.dot(120, y + 64, 26, c(rose, 46))
            cv.text(120, y + 64, f"{i + 1}", font(21, 700, "num"), rose, anchor="mm")
            cv.text(166, y + 26, self._event_time(event, now), font(17, 450), sub)
            target = self._target_label(event.target)
            if target:
                cv.pill(
                    988,
                    y + 24,
                    target,
                    font(15, 600),
                    rose,
                    c(rose, 36),
                    padx=13,
                    pady=6,
                    anchor="ra",
                )
            cv.text(166, y + 56, f"当前有影响的事情 · {i + 1:02d}", title_f, ink)
            ly = y + 96
            for line in fact_lines:
                cv.text(166, ly, line, fact_f, ink)
                ly += 36
            if meaning_lines:
                box_y = ly + 16
                box_h = 54 + len(meaning_lines) * 32
                cv.rrect((146, box_y, 990, box_y + box_h), 16, fill=pal["inner"])
                cv.rrect((146, box_y, 150, box_y + box_h), 2, fill=c(rose))
                cv.text(170, box_y + 14, "留在心里的感觉", font(15, 600), rose)
                my = box_y + 40
                for line in meaning_lines:
                    cv.text(170, my, line, mf, ink)
                    my += 32
            y += card_h + 24
        return y

    def _draw_attention(self, cv: Canvas, y, items, pal):
        ink, sub = pal["ink"], pal["sub"]
        lavender = pal["lavender"]
        y = self._draw_section_head(
            cv, y, "仍需留意或接续", f"{len(items)} 项", lavender, pal
        )
        content_f = font(25, 600)
        if not items:
            cv.rrect(
                (58, y, 1022, y + 90),
                22,
                fill=(*pal["tint"], 60),
                outline=pal["line"],
                width=1.5,
            )
            cv.text(82, y + 29, "目前没有待关注事项", font(24, 450), sub)
            return y + 90 + 24
        for item in items:
            lines, card_h = self._attention_metrics(cv, item)
            self._card(cv, (58, y, 1022, y + card_h), pal, radius=22)
            cv.bookmark(100, y + 44, 24, 34, c(lavender))
            label = f"{self._kind_label(item.kind)} · {self._status_label(item.status)}"
            cv.text(136, y + 22, label, font(16, 450), sub)
            ly = y + 50
            for line in lines:
                cv.text(136, ly, line, content_f, ink)
                ly += 36
            timing = item.time_hint or item.due_at
            if timing:
                cv.text(136, ly + 8, f"时间提示：{timing}", font(16, 450), sub)
            y += card_h + 20
        return y

    # -- entry -------------------------------------------------------------------
    def _render(
        self,
        ledger: StateLedger,
        events: Sequence[InnerEvent],
        attention_items: Sequence[AttentionItem],
        body_stage: str,
        now: datetime,
        theme: str,
    ) -> bytes:
        pal = self._palette(theme)
        sub = pal["sub"]

        probe = Canvas(height=8)
        event_hs = []
        for event in events:
            _, _, card_h = self._event_metrics(probe, event)
            event_hs.append(card_h + 24)
        event_section_h = 56 + sum(event_hs)
        if not events:
            event_section_h = 56 + 114
        att_hs = []
        for item in attention_items:
            _, card_h = self._attention_metrics(probe, item)
            att_hs.append(card_h + 20)
        att_section_h = 56 + sum(att_hs)
        if not attention_items:
            att_section_h = 56 + 114

        mood_top = 340
        body_top = mood_top + 300 + 32
        events_top = body_top + 104 + 50
        att_top = events_top + event_section_h + 16
        yf = att_top + att_section_h + 32

        cv = Canvas(
            height=int(yf + 52) + 40, bg="#FDEEF3" if theme == "day" else "#1E1420"
        )
        cv.bg_gradient(pal["stops"])
        for col, gx, gy, gr, ga in pal["glows"]:
            cv.glow(gx, gy, gr, c(col), ga)

        self._draw_header(cv, pal, now)
        self._draw_mood(cv, mood_top, ledger, pal)
        self._draw_body(cv, body_top, ledger, body_stage, pal)
        y = self._draw_events(cv, events_top, events, pal, now)
        y = self._draw_attention(cv, y + 16, attention_items, pal)

        ft = font(16, 500)
        total = cv.spaced(540, yf, "LINGXI · INNER WORLD", ft, sub, 5, anchor="ma")
        cv.star4(540 - total / 2 - 26, yf + 12, 7, c(pal["rose"], 200))
        cv.star4(540 + total / 2 + 26, yf + 12, 7, c(pal["lavender"], 200))
        return cv.finish(int(yf + 52))
