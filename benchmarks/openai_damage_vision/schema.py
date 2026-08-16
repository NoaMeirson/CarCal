"""Damage classes, prompt, and Structured Outputs JSON schema for the
OpenAI vision benchmark (see benchmark.py).

The six classes below mirror the ground-truth COCO categories in
instances_test2017.json:
    1 dent, 2 scratch, 3 crack, 4 glass shatter, 5 lamp broken, 6 tire flat
"""

DAMAGE_CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

PROMPT_VERSION = "v2"

PROMPT_TEXT = f"""You are inspecting a photograph of a vehicle for visible physical damage.

Identify every visible instance of damage to the vehicle's body, glass, or lights. \
Classify each instance as exactly one of these six damage types: {", ".join(DAMAGE_CLASSES)}.

For each instance, trace a segmentation POLYGON that follows the actual visible \
boundary of the damaged pixels as closely as possible -- not the surrounding \
panel, part, or the whole vehicle, and not a bounding box.

Polygon detail requirements:
- Trace the visible boundary of the damaged pixels as closely as possible.
- Do not approximate the damage with a rectangle or bounding-box-like polygon.
- Use enough polygon points to follow curves, thin scratches, irregular edges, \
and changes in direction in the damage's outline.
- Prefer 12-30 points per polygon when the damage shape is complex.
- A four-point polygon is allowed only when the visible damaged region is \
genuinely rectangular.
- For thin scratches, trace a narrow polygon that hugs the scratch itself, \
not a large region surrounding it.
- If a single damage instance has multiple disconnected visible parts, return \
only its single largest connected visible region (consistent with how CarCal's \
own mask-to-polygon conversion keeps only the largest contour).

Coordinates must be normalized to the [0, 1] range: (0, 0) is the top-left \
corner of the image, (1, 1) is the bottom-right corner.

Do NOT report any of the following as damage:
- Reflections, glare, or specular highlights on paint or glass
- Shadows cast by the vehicle itself or by other objects
- Dirt, mud, water spots, dust, or other surface debris
- Normal panel seams, body lines, gaps, trim, badges, or manufacturer design lines
- Lighting or exposure differences that are not physical damage
- General wear such as faded paint, unless it is an actual crack, dent, or scratch

If the vehicle has no visible damage, return an empty detections list.

For each detection, give a confidence score from 0 to 1 reflecting how certain \
you are that this is genuine physical damage of the stated type."""

# minItems/maxItems on the polygon array are supported by OpenAI's strict
# Structured Outputs mode (unlike e.g. minimum/maximum on numbers or pattern
# on strings, which are NOT reliably supported there, so we still avoid
# those). minItems/maxItems only bound what the API is asked to return --
# convert.py still independently validates and clamps every point on the
# client side (coordinate range, point count, dedup) rather than trusting
# the API to have enforced them, mirroring how
# Engine/utils/segmentation_utils.mask_to_polygon rejects polygons with
# fewer than 3 points.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": DAMAGE_CLASSES},
                    "confidence": {"type": "number"},
                    "polygon": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 40,
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["label", "confidence", "polygon"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["detections"],
    "additionalProperties": False,
}
