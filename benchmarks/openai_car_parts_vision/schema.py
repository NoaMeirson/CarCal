"""Part classes, prompt, and Structured Outputs JSON schema for the OpenAI
vision benchmark of CarCal's car-parts (Mask2Former) model.

PART_CLASSES is read verbatim from the production model's own label
vocabulary (Engine/artifacts/car_parts/car_parts_M2F_model/config.json's
id2label), not invented, so results are as directly comparable to the
Mask2Former model's output as possible:
    0 back_bumper, 1 back_door, 2 back_glass, 3 back_left_door,
    4 back_left_light, 5 back_light, 6 back_right_door, 7 back_right_light,
    8 front_bumper, 9 front_door, 10 front_glass, 11 front_left_door,
    12 front_left_light, 13 front_light, 14 front_right_door,
    15 front_right_light, 16 hood, 17 left_mirror, 18 object,
    19 right_mirror, 20 tailgate, 21 trunk, 22 wheel

Note this vocabulary mixes generic and side-specific labels for the same
physical part (e.g. front_door vs. front_left_door/front_right_door,
front_light vs. front_left_light/front_right_light) -- this was updated to
match a newer Mask2Former checkpoint than the previous 14-class one. If the
production model is swapped again, re-read its config.json and update this
list to match.

If your ground-truth annotations use different names/casing for the same
parts, align them (or the evaluation code) with this exact vocabulary.
"""

PART_CLASSES = [
    "back_bumper",
    "back_door",
    "back_glass",
    "back_left_door",
    "back_left_light",
    "back_light",
    "back_right_door",
    "back_right_light",
    "front_bumper",
    "front_door",
    "front_glass",
    "front_left_door",
    "front_left_light",
    "front_light",
    "front_right_door",
    "front_right_light",
    "hood",
    "left_mirror",
    "object",
    "right_mirror",
    "tailgate",
    "trunk",
    "wheel",
]

PROMPT_VERSION = "v3"

PROMPT_TEXT = f"""You are inspecting a photograph of a vehicle to identify and segment its \
visible body parts, exactly as an instance segmentation model would.

Identify every visible instance of the following vehicle parts, and only these: \
{", ".join(PART_CLASSES)}. Do not report any other part of the vehicle (e.g. roof, \
pillars, exhaust, license plate, badges) -- this vocabulary is closed to the \
classes above.

Some parts have both a generic label and side-specific labels for the same \
physical part (e.g. front_door vs. front_left_door / front_right_door; \
front_light vs. front_left_light / front_right_light; and the back_ \
equivalents of both). Use the side-specific label whenever you can clearly \
tell from the vehicle's visible orientation whether it's the left or right \
side. Only fall back to the generic label (front_door, back_door, \
front_light, back_light) when the side genuinely cannot be determined from \
the photo. "object" is a catch-all for a vehicle body part that is clearly \
part of the vehicle but does not match any other class -- use it rarely, \
only when no other label fits.

For each visible instance, provide a segmentation POLYGON that follows its \
actual visible outline as closely as possible -- NOT a bounding box.

Polygon detail requirements:
- Trace the visible boundary of the part as closely as possible, including \
its true curves and contours.
- Do not approximate the part with a bounding-box-like rectangle. Most body \
parts (bumpers, doors, the hood, mirrors) are curved or irregular in a real \
photo, even if a simplified rectangle might loosely cover them -- follow the \
actual edge, not a box around it.
- Use enough points to follow curves and any occlusion edges where another \
part or object overlaps this one. A small, simply-shaped part (e.g. a \
mirror or a light seen head-on) may only need 5-10 points; a large or highly \
curved part (e.g. the hood, a bumper, or a door) should typically use 15-40 \
points to actually follow its outline.
- A four-point polygon is only acceptable when the visible region is \
genuinely a simple quadrilateral (e.g. a flat window panel seen straight-on) \
-- not as a default shortcut for parts that are actually curved.
- Trace only the VISIBLE (unoccluded) extent of the part. If another part or \
object partially covers it, follow the visible boundary where it is covered \
-- do not guess at or complete the hidden shape.
- If the same part type appears more than once (e.g. both mirrors, both \
front lights, more than one wheel), report each physical instance as its \
own separate detection -- never merge multiple instances into one polygon.
- If a part is not visible at all in the photo (e.g. the tailgate when only \
the front of the car is shown), do not report it. Only report what is \
actually visible.
- Use the vehicle's visible orientation in the photo to correctly \
distinguish front vs. back parts (front_bumper vs. back_bumper, front_door \
vs. back_door, front_glass vs. back_glass, etc.).

Coordinates must be normalized to the [0, 1] range: (0, 0) is the top-left \
corner of the image, (1, 1) is the bottom-right corner.

If no vehicle or none of these parts are visible, return an empty detections list.

For each detection, give a confidence score from 0 to 1 reflecting how certain \
you are of both the part's identity and its traced boundary."""

# minItems/maxItems on the polygon array are supported by OpenAI's strict
# Structured Outputs mode (unlike e.g. minimum/maximum on numbers or pattern
# on strings, which are NOT reliably supported there, so those are still
# avoided). maxItems is higher than the damage benchmark's (40) since whole
# body-panel outlines are typically larger/more complex shapes than a single
# damage instance. convert.py still independently validates and clamps
# every point client-side (coordinate range, point count, dedup) rather than
# trusting the API to have enforced these bounds, mirroring how
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
                    "label": {"type": "string", "enum": PART_CLASSES},
                    "confidence": {"type": "number"},
                    "polygon": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 50,
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
