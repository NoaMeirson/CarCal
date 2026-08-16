"""Draws OpenAI-predicted car-part polygons and labels on top of the source
image, for visual QA of the benchmark's output before/instead of running a
full quantitative evaluation."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# One distinct color per PART_CLASSES entry (23 classes, in schema.py's
# order) so each part type is visually distinguishable in the overlay.
# Generated via evenly-spaced hues (colorsys) with alternating
# saturation/value to keep adjacent hues distinguishable.
_COLORS = [
    (191, 66, 66),    # back_bumper
    (229, 85, 34),    # back_door
    (229, 158, 80),   # back_glass
    (191, 155, 28),   # back_left_door
    (223, 229, 80),   # back_left_light
    (170, 229, 34),   # back_light
    (120, 191, 66),   # back_right_door
    (68, 229, 34),    # back_right_light
    (80, 229, 93),    # front_bumper
    (28, 191, 85),    # front_door
    (80, 229, 171),   # front_glass
    (34, 229, 204),   # front_left_door
    (66, 175, 191),   # front_left_light
    (34, 153, 229),   # front_light
    (80, 132, 229),   # front_right_door
    (28, 42, 191),    # front_right_light
    (106, 80, 229),   # hood
    (119, 34, 229),   # left_mirror
    (153, 66, 191),   # object
    (221, 34, 229),   # right_mirror
    (229, 80, 197),   # tailgate
    (191, 28, 113),   # trunk
    (229, 80, 119),   # wheel
]


def _color_for_label(label: str, labels_order: list[str]) -> tuple[int, int, int]:
    try:
        idx = labels_order.index(label)
    except ValueError:
        idx = 0
    return _COLORS[idx % len(_COLORS)]


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(
    image_path: Path,
    detections: list[dict],
    output_path: Path,
    labels_order: list[str],
) -> None:
    """detections: list of Detection.model_dump() dicts, i.e. each has
    {id, type, label, confidence, polygon: {points: [{x, y}, ...]}, matches}.
    Always writes a PNG (regardless of the source format) so the
    semi-transparent overlay doesn't run into JPEG-has-no-alpha issues."""
    with Image.open(image_path) as src:
        base = src.convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(max(14, base.width // 60))

    for det in detections:
        points = [(p["x"], p["y"]) for p in det["polygon"]["points"]]
        if len(points) < 2:
            continue

        color = _color_for_label(det["label"], labels_order)
        draw.polygon(points, outline=color + (255,), fill=color + (60,), width=3)

        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        text = f"{det['label']} {det['confidence']:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        draw.rectangle(
            [cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad],
            fill=color + (230,),
        )
        draw.text((cx - tw / 2, cy - th / 2), text, fill=(255, 255, 255, 255), font=font)

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, format="PNG")
