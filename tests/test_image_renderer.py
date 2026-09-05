from datetime import datetime, timezone
from io import BytesIO

from astrbot_plugin_emotion_state.core.image_renderer import EmotionStateImageRenderer
from astrbot_plugin_emotion_state.core.models import (
    AttentionItem,
    InnerEvent,
    StateLedger,
)
from PIL import Image


def open_png(payload: bytes) -> Image.Image:
    image = Image.open(BytesIO(payload))
    image.load()
    return image


def make_ledger() -> StateLedger:
    ledger = StateLedger(
        user_key="private:image",
        events=[
            InnerEvent(
                fact="下午到深夜一直在聊天、等待和互相回应。" * 18,
                emotional_meaning="持续的陪伴让我感到被惦记，也更愿意靠近。" * 8,
                target="user",
                valence=0.64,
                intensity=0.82,
                confidence=0.9,
                lifecycle="active",
            )
        ],
        attention_items=[
            AttentionItem(
                content="以后多发语音，晚上记得接着聊。",
                kind="commitment",
                status="open",
                time_hint="以后、晚上",
            )
        ],
    )
    ledger.mood.label = "温和愉快"
    ledger.mood.valence = 0.64
    ledger.mood.energy = 0.58
    ledger.mood.tension = 0.21
    ledger.intimacy.body_sensitivity = 0.91
    ledger.intimacy.sexual_arousal = 0.88
    return ledger


def test_theme_resolves_explicit_and_automatic_hours(tmp_path) -> None:
    renderer = EmotionStateImageRenderer(tmp_path)

    assert renderer.resolve_theme("亮色", datetime(2026, 8, 17, 23)) == "day"
    assert renderer.resolve_theme("暗色", datetime(2026, 8, 17, 12)) == "night"
    assert renderer.resolve_theme("自动", datetime(2026, 8, 17, 7)) == "day"
    assert renderer.resolve_theme("自动", datetime(2026, 8, 17, 18, 59)) == "day"
    assert renderer.resolve_theme("自动", datetime(2026, 8, 17, 19)) == "night"


def test_renderer_supports_long_events_and_both_themes(tmp_path) -> None:
    renderer = EmotionStateImageRenderer(tmp_path)
    ledger = make_ledger()
    now = datetime(2026, 8, 17, 22, 16, tzinfo=timezone.utc)

    day = open_png(
        renderer.render(
            ledger,
            ledger.events,
            ledger.attention_items,
            "身体反应强烈，敏感度和性唤起处于高位",
            now,
            "亮色",
        )
    )
    night = open_png(
        renderer.render(
            ledger,
            ledger.events,
            ledger.attention_items,
            "身体反应强烈，敏感度和性唤起处于高位",
            now,
            "暗色",
        )
    )

    assert day.mode == night.mode == "RGB"
    assert day.width == night.width == 1080
    assert day.height == night.height
    assert day.height > 1700
    assert day.getpixel((20, 20)) != night.getpixel((20, 20))
    assert max(night.getpixel((20, 20))) < 40


def test_longer_event_produces_taller_image(tmp_path) -> None:
    renderer = EmotionStateImageRenderer(tmp_path)
    ledger = make_ledger()
    now = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)
    short_event = InnerEvent(
        fact="今天收到了一句认真回应。",
        emotional_meaning="我感到被在意。",
        lifecycle="active",
    )

    short = open_png(
        renderer.render(ledger, [short_event], [], "身体平静", now, "亮色")
    )
    long = open_png(renderer.render(ledger, ledger.events, [], "身体平静", now, "亮色"))

    assert long.height > short.height + 500


def test_logo_is_read_from_plugin_directory(tmp_path) -> None:
    renderer = EmotionStateImageRenderer(tmp_path)
    ledger = make_ledger()
    now = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)
    Image.new("RGB", (240, 240), "red").save(tmp_path / "logo.png")
    first = renderer.render(ledger, [], [], "身体平静", now, "亮色")

    Image.new("RGB", (240, 240), "blue").save(tmp_path / "logo.png")
    second = renderer.render(ledger, [], [], "身体平静", now, "亮色")

    assert first != second


def test_guidance_section_renders_and_expands_image(tmp_path) -> None:
    renderer = EmotionStateImageRenderer(tmp_path)
    ledger = make_ledger()
    now = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)
    guidance = {
        "tone": "夜深了，语气比白天软、慢，带一点黏人和低落。",
        "can_say": "主动说今晚有点安静得不习惯，想起他很久没来了，有点难过；" * 3,
        "avoid": "无缘由地冷淡或阴阳怪气；不要一上来就质问。",
    }

    without = open_png(renderer.render(ledger, [], [], "身体平静", now, "亮色"))
    with_guidance = open_png(
        renderer.render(ledger, [], [], "身体平静", now, "亮色", guidance)
    )

    assert with_guidance.height > without.height
    assert with_guidance.width == without.width == 1080
    # The suggestion card keeps its content inside the canvas bounds.
    assert with_guidance.getpixel((540, with_guidance.height - 30)) != None
