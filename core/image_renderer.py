"""Local Pillow renderer for the emotion view command."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .models import AttentionItem, InnerEvent, StateLedger, parse_time


@dataclass(frozen=True)
class _EventLayout:
    event: InnerEvent
    fact_lines: list[str]
    meaning_lines: list[str]
    height: int


@dataclass(frozen=True)
class _AttentionLayout:
    item: AttentionItem
    content_lines: list[str]
    height: int


class EmotionStateImageRenderer:
    """Render one long-form mood sheet without browser dependencies."""

    width = 1080
    day_start_hour = 7
    night_start_hour = 19

    _DAY = {
        "background": "#FFF9FA",
        "header": "#F3DCE4",
        "header_border": "#D9B8C4",
        "text": "#4B3941",
        "muted": "#786B72",
        "title": "#5C445F",
        "border": "#EADDE1",
        "surface": "#FFFEFE",
        "shadow": (79, 54, 65, 22),
        "rose": "#D6869E",
        "blue": "#91A9DD",
        "gold": "#D6B85F",
        "track": "#EEE2E6",
        "body_surface": "#F8E8ED",
        "body_border": "#E8CBD5",
        "body_icon": "#B84F75",
        "body_icon_surface": "#FFF7FA",
        "chip": "#E5ECF8",
        "chip_text": "#576984",
        "index": "#B66683",
        "index_surface": "#F6DFE6",
        "meaning_surface": "#F7F0F8",
        "meaning_border": "#9C82BD",
        "meaning_label": "#765C91",
        "attention_surface": "#FFF3CF",
        "attention_border": "#EAD391",
        "attention_text": "#55482E",
        "attention_icon": "#F2D989",
        "footer": "#6C5B88",
        "footer_text": "#FFF9FF",
    }
    _NIGHT = {
        "background": "#101113",
        "header": "#17181B",
        "header_border": "#2B2C30",
        "text": "#E8E5E7",
        "muted": "#AAA6AA",
        "title": "#DED9DC",
        "border": "#292A2E",
        "surface": "#161719",
        "shadow": (0, 0, 0, 62),
        "rose": "#78676D",
        "blue": "#5C6370",
        "gold": "#656154",
        "track": "#292A2D",
        "body_surface": "#191A1D",
        "body_border": "#2E2F33",
        "body_icon": "#A98691",
        "body_icon_surface": "#27272B",
        "chip": "#25272B",
        "chip_text": "#C5C5C8",
        "index": "#A17F89",
        "index_surface": "#29252A",
        "meaning_surface": "#1D1D20",
        "meaning_border": "#625B65",
        "meaning_label": "#AAA3AD",
        "attention_surface": "#1B1A17",
        "attention_border": "#343129",
        "attention_text": "#D8D3C7",
        "attention_icon": "#292720",
        "footer": "#0A0B0D",
        "footer_text": "#BCB9BB",
    }

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
        theme = self.resolve_theme(mode, now)
        palette = self._DAY if theme == "day" else self._NIGHT
        fonts = self._fonts()
        probe = Image.new("RGB", (self.width, 100), palette["background"])
        probe_draw = ImageDraw.Draw(probe)
        event_layouts = [
            self._event_layout(probe_draw, item, fonts) for item in events
        ]
        attention_layouts = [
            self._attention_layout(probe_draw, item, fonts)
            for item in attention_items
        ]

        height = 220 + 320 + 168 + 92
        height += sum(item.height + 22 for item in event_layouts)
        if not event_layouts:
            height += 112
        height += 86
        if attention_layouts:
            height += sum(item.height + 16 for item in attention_layouts)
        else:
            height += 112
        height += 92

        image = Image.new("RGBA", (self.width, height), palette["background"])
        draw = ImageDraw.Draw(image)
        self._draw_header(image, draw, now, theme, palette, fonts)
        self._draw_mood(draw, ledger, 250, theme, palette, fonts)
        self._draw_body(draw, ledger, body_stage, 570, palette, fonts)

        y = 738
        self._draw_section_heading(
            draw,
            y,
            "仍在心里回响的事",
            f"{len(event_layouts)} 件",
            palette,
            fonts,
        )
        y += 82
        if event_layouts:
            for index, layout in enumerate(event_layouts, 1):
                self._draw_event(
                    image,
                    draw,
                    y,
                    index,
                    layout,
                    now,
                    theme,
                    palette,
                    fonts,
                )
                y += layout.height + 22
        else:
            self._draw_empty(draw, y, "现在没有持续影响心境的事情", palette, fonts)
            y += 112

        self._draw_section_heading(
            draw,
            y,
            "仍需留意或接续",
            f"{len(attention_layouts)} 项",
            palette,
            fonts,
        )
        y += 76
        if attention_layouts:
            for layout in attention_layouts:
                self._draw_attention(draw, y, layout, palette, fonts)
                y += layout.height + 16
        else:
            self._draw_empty(draw, y, "目前没有待关注事项", palette, fonts)
            y += 112

        self._draw_footer(draw, height - 76, palette, fonts)
        return self._encode_png(image)

    def _font(self, size: int, bold: bool = False) -> Any:
        candidates: list[Path] = []
        if bold:
            candidates.extend(
                [
                    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
                    Path(r"C:\Windows\Fonts\Dengb.ttf"),
                    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                    Path("/System/Library/Fonts/PingFang.ttc"),
                ]
            )
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\msyh.ttc"),
                Path(r"C:\Windows\Fonts\simhei.ttf"),
                Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
                Path("/System/Library/Fonts/PingFang.ttc"),
            ]
        )
        for path in candidates:
            if path.is_file():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default(size=size)

    def _fonts(self) -> dict[str, Any]:
        return {
            "brand": self._font(54, True),
            "hero": self._font(46, True),
            "score": self._font(48, True),
            "section": self._font(34, True),
            "event": self._font(30, True),
            "body": self._font(27),
            "body_bold": self._font(27, True),
            "small": self._font(22),
            "small_bold": self._font(22, True),
            "tiny": self._font(19),
        }

    @staticmethod
    def _text_width(draw: ImageDraw.ImageDraw, text: str, font: Any) -> int:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0]

    @staticmethod
    def _line_height(font: Any, spacing: int) -> int:
        box = font.getbbox("国Ag")
        return box[3] - box[1] + spacing

    def _wrap(
        self, draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int
    ) -> list[str]:
        lines: list[str] = []
        for paragraph in str(text or "").splitlines() or [""]:
            if not paragraph:
                lines.append("")
                continue
            line = ""
            for char in paragraph:
                candidate = line + char
                if line and self._text_width(draw, candidate, font) > max_width:
                    lines.append(line)
                    line = char
                else:
                    line = candidate
            if line:
                lines.append(line)
        return lines or [""]

    def _event_layout(
        self, draw: ImageDraw.ImageDraw, event: InnerEvent, fonts: dict[str, Any]
    ) -> _EventLayout:
        fact_lines = self._wrap(draw, event.fact, fonts["body"], 830)
        meaning_lines = (
            self._wrap(draw, event.emotional_meaning, fonts["body"], 790)
            if event.emotional_meaning.strip()
            else []
        )
        fact_height = len(fact_lines) * self._line_height(fonts["body"], 12)
        meaning_height = len(meaning_lines) * self._line_height(fonts["body"], 11)
        height = 144 + fact_height
        if meaning_lines:
            height += 84 + meaning_height
        height = max(270, height)
        return _EventLayout(event, fact_lines, meaning_lines, height)

    def _attention_layout(
        self, draw: ImageDraw.ImageDraw, item: AttentionItem, fonts: dict[str, Any]
    ) -> _AttentionLayout:
        content_lines = self._wrap(draw, item.content, fonts["body_bold"], 810)
        content_height = len(content_lines) * self._line_height(
            fonts["body_bold"], 11
        )
        return _AttentionLayout(item, content_lines, max(132, 108 + content_height))

    def _draw_header(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        now: datetime,
        theme: str,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        draw.rectangle((0, 0, self.width, 220), fill=palette["header"])
        draw.rectangle((0, 216, self.width, 220), fill=palette["header_border"])
        avatar = self._load_avatar(
            132,
            "#FFF7FA" if theme == "day" else "#303136",
        )
        if avatar:
            image.alpha_composite(avatar, (48, 42))
        text_x = 216 if avatar else 54
        draw.text(
            (text_x, 45),
            "LINGXI · INNER WORLD",
            font=fonts["small"],
            fill=palette["muted"],
        )
        draw.text(
            (text_x, 78), "内心世界", font=fonts["brand"], fill=palette["title"]
        )
        draw.text(
            (text_x, 151),
            f"{now:%Y 年 %m 月 %d 日  ·  %H:%M}",
            font=fonts["small"],
            fill=palette["muted"],
        )
        self._draw_sparkle(draw, 954, 68, palette["rose"])
        self._draw_sparkle(draw, 900, 138, palette["blue"], size=13)

    def _draw_mood(
        self,
        draw: ImageDraw.ImageDraw,
        ledger: StateLedger,
        y: int,
        theme: str,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        center = (150, y + 122)
        draw.ellipse(
            (center[0] - 86, center[1] - 86, center[0] + 86, center[1] + 86),
            fill="#FFFEFE" if theme == "day" else "#17181A",
            outline=palette["track"],
            width=9,
        )
        valence = max(-1.0, min(1.0, ledger.mood.valence))
        arc_extent = max(12, int(abs(valence) * 300))
        arc_color = palette["rose"] if valence >= 0 else palette["blue"]
        draw.arc(
            (center[0] - 86, center[1] - 86, center[0] + 86, center[1] + 86),
            -90,
            -90 + arc_extent,
            fill=arc_color,
            width=9,
        )
        score = int(round(valence * 100))
        score_text = f"{score:+d}"
        score_width = self._text_width(draw, score_text, fonts["score"])
        draw.text(
            (center[0] - score_width / 2, center[1] - 45),
            score_text,
            font=fonts["score"],
            fill=palette["title"],
        )
        label_width = self._text_width(draw, "情绪数值", fonts["tiny"])
        draw.text(
            (center[0] - label_width / 2, center[1] + 23),
            "情绪数值",
            font=fonts["tiny"],
            fill=palette["muted"],
        )

        draw.text((280, y + 22), "此刻的心境", font=fonts["small"], fill=palette["muted"])
        draw.text(
            (280, y + 56),
            ledger.mood.label or "平静",
            font=fonts["hero"],
            fill=palette["title"],
        )
        values = [
            ("情绪偏向", abs(valence), f"{score:+d}", palette["rose"]),
            ("能量", ledger.mood.energy, f"{round(ledger.mood.energy * 100):.0f}", palette["blue"]),
            ("紧张", ledger.mood.tension, f"{round(ledger.mood.tension * 100):.0f}", palette["gold"]),
        ]
        bar_y = y + 136
        for label, value, display, color in values:
            draw.text((280, bar_y), label, font=fonts["small"], fill=palette["muted"])
            draw.rounded_rectangle(
                (405, bar_y + 8, 880, bar_y + 18), radius=5, fill=palette["track"]
            )
            fill_width = max(8, int(475 * max(0.0, min(1.0, value))))
            draw.rounded_rectangle(
                (405, bar_y + 8, 405 + fill_width, bar_y + 18),
                radius=5,
                fill=color,
            )
            draw.text((916, bar_y), display, font=fonts["small_bold"], fill=palette["text"])
            bar_y += 50

    def _draw_body(
        self,
        draw: ImageDraw.ImageDraw,
        ledger: StateLedger,
        body_stage: str,
        y: int,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        box = (48, y, 1032, y + 138)
        draw.rounded_rectangle(
            box,
            radius=24,
            fill=palette["body_surface"],
            outline=palette["body_border"],
            width=2,
        )
        draw.ellipse((72, y + 34, 138, y + 100), fill=palette["body_icon_surface"])
        self._draw_heart(draw, 92, y + 52, palette["body_icon"])
        draw.text((166, y + 26), "身体反应", font=fonts["small"], fill=palette["muted"])
        draw.text(
            (166, y + 62),
            self._body_tier_label(body_stage),
            font=fonts["body_bold"],
            fill=palette["text"],
        )
        sensitivity = round(ledger.intimacy.body_sensitivity * 100)
        arousal = round(ledger.intimacy.sexual_arousal * 100)
        self._draw_chip(draw, 778, y + 46, f"敏感度 {sensitivity}", palette, fonts)
        self._draw_chip(draw, 908, y + 46, f"唤起 {arousal}", palette, fonts)

    def _draw_section_heading(
        self,
        draw: ImageDraw.ImageDraw,
        y: int,
        title: str,
        count: str,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        draw.ellipse((50, y + 8, 66, y + 24), fill=palette["rose"])
        draw.ellipse((72, y + 8, 88, y + 24), fill=palette["blue"])
        draw.text((108, y), title, font=fonts["section"], fill=palette["title"])
        count_width = self._text_width(draw, count, fonts["small_bold"])
        draw.rounded_rectangle(
            (930 - count_width, y + 2, 1032, y + 38),
            radius=8,
            fill=palette["chip"],
        )
        draw.text(
            (950 - count_width, y + 5),
            count,
            font=fonts["small_bold"],
            fill=palette["chip_text"],
        )

    def _draw_event(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        y: int,
        index: int,
        layout: _EventLayout,
        now: datetime,
        theme: str,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        box = (48, y, 1032, y + layout.height)
        self._shadow(image, box, palette["shadow"])
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            box,
            radius=24,
            fill=palette["surface"],
            outline=palette["border"],
            width=2,
        )
        draw.ellipse((72, y + 28, 128, y + 84), fill=palette["index_surface"])
        index_text = f"{index:02d}"
        index_width = self._text_width(draw, index_text, fonts["small_bold"])
        draw.text(
            (100 - index_width / 2, y + 42),
            index_text,
            font=fonts["small_bold"],
            fill=palette["index"],
        )

        event = layout.event
        time_text = self._event_time(event, now)
        draw.text((150, y + 28), time_text, font=fonts["small"], fill=palette["muted"])
        target = self._target_label(event.target)
        if target:
            chip_width = min(260, self._text_width(draw, target, fonts["tiny"]) + 28)
            chip_x = 1000 - chip_width
            draw.rounded_rectangle(
                (chip_x, y + 24, 1000, y + 58), radius=7, fill=palette["chip"]
            )
            draw.text(
                (chip_x + 14, y + 29),
                target,
                font=fonts["tiny"],
                fill=palette["chip_text"],
            )
        draw.text(
            (150, y + 70),
            f"当前有影响的事情 · {index:02d}",
            font=fonts["event"],
            fill=palette["title"],
        )
        fact_y = y + 116
        fact_bottom = self._draw_lines(
            draw,
            (150, fact_y),
            layout.fact_lines,
            fonts["body"],
            palette["text"],
            12,
        )
        if layout.meaning_lines:
            meaning_y = fact_bottom + 20
            meaning_height = 58 + len(layout.meaning_lines) * self._line_height(
                fonts["body"], 11
            )
            draw.rounded_rectangle(
                (132, meaning_y, 1000, meaning_y + meaning_height),
                radius=14,
                fill=palette["meaning_surface"],
            )
            draw.rectangle(
                (132, meaning_y, 139, meaning_y + meaning_height),
                fill=palette["meaning_border"],
            )
            draw.text(
                (164, meaning_y + 16),
                "留在心里的感觉",
                font=fonts["small_bold"],
                fill=palette["meaning_label"],
            )
            self._draw_lines(
                draw,
                (164, meaning_y + 52),
                layout.meaning_lines,
                fonts["body"],
                palette["text"],
                11,
            )

    def _draw_attention(
        self,
        draw: ImageDraw.ImageDraw,
        y: int,
        layout: _AttentionLayout,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        box = (48, y, 1032, y + layout.height)
        draw.rounded_rectangle(
            box,
            radius=22,
            fill=palette["attention_surface"],
            outline=palette["attention_border"],
            width=2,
        )
        draw.ellipse((72, y + 30, 128, y + 86), fill=palette["attention_icon"])
        self._draw_bookmark(draw, 91, y + 43, palette["attention_text"])
        item = layout.item
        label = f"{self._kind_label(item.kind)} · {self._status_label(item.status)}"
        draw.text((152, y + 24), label, font=fonts["small"], fill=palette["muted"])
        self._draw_lines(
            draw,
            (152, y + 58),
            layout.content_lines,
            fonts["body_bold"],
            palette["attention_text"],
            11,
        )
        timing = item.time_hint or item.due_at
        if timing:
            draw.text(
                (152, y + layout.height - 38),
                f"时间提示：{timing}",
                font=fonts["tiny"],
                fill=palette["muted"],
            )

    def _draw_empty(
        self,
        draw: ImageDraw.ImageDraw,
        y: int,
        text: str,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        draw.rounded_rectangle(
            (48, y, 1032, y + 90),
            radius=20,
            fill=palette["surface"],
            outline=palette["border"],
            width=2,
        )
        draw.text((82, y + 29), text, font=fonts["body"], fill=palette["muted"])

    @staticmethod
    def _draw_footer(
        draw: ImageDraw.ImageDraw,
        y: int,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        draw.rectangle((0, y, 1080, y + 76), fill=palette["footer"])
        draw.text(
            (48, y + 25),
            "灵犀 · 内心世界",
            font=fonts["small"],
            fill=palette["footer_text"],
        )
        right = "只属于这一刻的心绪"
        width = draw.textbbox((0, 0), right, font=fonts["small"])[2]
        draw.text(
            (1032 - width, y + 25),
            right,
            font=fonts["small"],
            fill=palette["footer_text"],
        )

    def _load_avatar(self, size: int, border: str) -> Image.Image | None:
        try:
            source = Image.open(self.plugin_dir / "logo.png").convert("RGB")
        except Exception:
            return None
        source = ImageOps.fit(
            source,
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
        result = Image.new("RGBA", (size + 12, size + 12), (0, 0, 0, 0))
        ImageDraw.Draw(result).ellipse((0, 0, size + 11, size + 11), fill=border)
        result.paste(source, (6, 6), mask)
        return result

    @staticmethod
    def _shadow(
        image: Image.Image,
        box: tuple[int, int, int, int],
        fill: tuple[int, int, int, int],
    ) -> None:
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(layer)
        x1, y1, x2, y2 = box
        shadow_draw.rounded_rectangle(
            (x1, y1 + 8, x2, y2 + 8), radius=24, fill=fill
        )
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(14)))

    @staticmethod
    def _draw_lines(
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        lines: Sequence[str],
        font: Any,
        fill: Any,
        spacing: int,
    ) -> int:
        x, y = xy
        line_height = EmotionStateImageRenderer._line_height(font, spacing)
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        return y

    def _draw_chip(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
        palette: dict[str, Any],
        fonts: dict[str, Any],
    ) -> None:
        width = self._text_width(draw, text, fonts["tiny"]) + 24
        draw.rounded_rectangle(
            (x - width, y, x, y + 38), radius=8, fill=palette["body_icon_surface"]
        )
        draw.text(
            (x - width + 12, y + 7),
            text,
            font=fonts["tiny"],
            fill=palette["text"],
        )

    @staticmethod
    def _draw_sparkle(
        draw: ImageDraw.ImageDraw, x: int, y: int, color: Any, size: int = 18
    ) -> None:
        draw.line((x - size, y, x + size, y), fill=color, width=3)
        draw.line((x, y - size, x, y + size), fill=color, width=3)
        half = size // 2
        draw.line((x - half, y - half, x + half, y + half), fill=color, width=2)
        draw.line((x - half, y + half, x + half, y - half), fill=color, width=2)

    @staticmethod
    def _draw_heart(draw: ImageDraw.ImageDraw, x: int, y: int, color: Any) -> None:
        draw.ellipse((x, y, x + 17, y + 17), fill=color)
        draw.ellipse((x + 15, y, x + 32, y + 17), fill=color)
        draw.polygon([(x, y + 9), (x + 32, y + 9), (x + 16, y + 31)], fill=color)

    @staticmethod
    def _draw_bookmark(draw: ImageDraw.ImageDraw, x: int, y: int, color: Any) -> None:
        draw.polygon(
            [(x, y), (x + 19, y), (x + 19, y + 30), (x + 10, y + 23), (x, y + 30)],
            fill=color,
        )

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
    def _status_label(status: str) -> str:
        return {
            "proposed": "待确认",
            "open": "仍待关注",
        }.get(str(status), "仍待关注")

    @staticmethod
    def _event_time(event: InnerEvent, now: datetime) -> str:
        try:
            value = parse_time(event.last_stimulated_at or event.updated_at)
            if now.tzinfo is not None:
                value = value.astimezone(now.tzinfo)
            return f"{value:%m 月 %d 日 · %H:%M}"
        except (TypeError, ValueError, OSError):
            return "最近更新"

    @staticmethod
    def _encode_png(image: Image.Image) -> bytes:
        output = BytesIO()
        image.convert("RGB").save(output, format="PNG", optimize=True)
        return output.getvalue()
