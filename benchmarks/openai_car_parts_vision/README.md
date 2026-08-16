# OpenAI vision benchmark — car parts segmentation

Sends each image in a YOLO-segmentation-labeled test set to the OpenAI
Responses API, asks it to segment visible vehicle parts into the same 23
classes the production Mask2Former model was trained on, and writes results
in the exact `EngineAnalyzeResponse` / `Detection` schema that
`Engine/Engine_methods.py`'s `process_parts_only()` returns in production
(see root `models.py`). This lets the OpenAI output be scored with the same
evaluation code as the local Mask2Former car-parts model, without touching
the production inference pipeline at all.

Generating predictions never needs the ground truth, so the `.txt` label
files' *contents* are never read by this script — only checked for
existence, to know which images are part of the labeled test set.

This is the sibling of
[`../openai_damage_vision/`](../openai_damage_vision/) (same design, same
CLI shape) for the car-parts model instead of the damage model. See that
folder's script if you want to diff the two.

## Part vocabulary

`schema.py`'s `PART_CLASSES` is read verbatim from the production model's
own label set
(`Engine/artifacts/car_parts/car_parts_M2F_model/config.json`'s
`id2label`), not invented, so OpenAI's output uses the same 23 class names
Mask2Former does:
```
back_bumper, back_door, back_glass, back_left_door, back_left_light,
back_light, back_right_door, back_right_light, front_bumper, front_door,
front_glass, front_left_door, front_left_light, front_light,
front_right_door, front_right_light, hood, left_mirror, object,
right_mirror, tailgate, trunk, wheel
```
This class order (0=back_bumper ... 22=wheel) matches the numeric class IDs
used in the ground-truth `.txt` files below — keep that mapping in mind
when you write the evaluation code that compares predictions to ground
truth.

Note this vocabulary mixes generic and side-specific labels for the same
physical part (`front_door` vs. `front_left_door`/`front_right_door`, and
similarly for lights) — the prompt tells OpenAI to prefer the side-specific
label when the side is visually determinable and fall back to the generic
one otherwise. If the production model is swapped again, re-run the
inspection command below and update `PART_CLASSES` to match:
```powershell
python -c "import json; d=json.load(open('Engine/artifacts/car_parts/car_parts_M2F_model/config.json')); print([d['id2label'][str(i)] for i in range(len(d['id2label']))])"
```

## 1. Place the data

This script does not ship a test set. Ground truth here is YOLO-segmentation
format: one `.txt` per image, same filename stem, each line
`class_id x1 y1 x2 y2 ... xn yn` with coordinates normalized to `[0, 1]`
(this is *not* COCO JSON — see `../openai_damage_vision/` for that format,
used for the damage model's test set instead). For example:
```
1 0.5300853515624999 0.450048846875 0.5462093296874999 0.7030615484375 ...
```
(class `1` = `back_door` per the vocabulary above).

Put the images and labels anywhere on disk — same folder or split into
`images/`/`labels/` subfolders, both work:
```
<TEST_IMAGES_DIR>/       # e.g. images/0012.jpg
<TEST_LABELS_DIR>/       # e.g. labels/0012.txt  (defaults to TEST_IMAGES_DIR if omitted)
```
The script only checks that each image has a matching `.txt` in
`--labels-dir` (same stem) to confirm it's part of the labeled test set —
it never opens or parses that file's contents. Images in `--images-dir`
without a matching `.txt` are skipped with a warning (not an error), so
it's safe to point `--images-dir` at a folder that has extra, unlabeled
images mixed in.

## 2. Install dependencies

From this folder:
```powershell
pip install -r requirements.txt
```
(`pydantic`, `pillow`, and `tqdm` are already in the project's main `.venv`
if you'd rather reuse that; only `openai` needs installing there.)

## 3. Set your API key

```powershell
$env:OPENAI_API_KEY = "sk-..."
```
Read from the environment only — never hardcoded, never sent to or read by
`React_Client` or any of the Python services.

## 4. Run the 3-image test

```powershell
python benchmark.py --images-dir "<TEST_IMAGES_DIR>" --labels-dir "<TEST_LABELS_DIR>" --output results_test.jsonl --test
```
(omit `--labels-dir` if the `.txt` files sit right next to the images in
the same folder)

## 5. Run the full benchmark

```powershell
python benchmark.py --images-dir "<TEST_IMAGES_DIR>" --labels-dir "<TEST_LABELS_DIR>" --output results.jsonl
```

If it's interrupted, just re-run the same command — already-processed
images (by filename) are skipped. Add `--retry-errors` to reprocess only
the images that previously failed instead of skipping them too. A log file
(`results.jsonl` → `results.log`) captures every retry, dropped detection,
skipped/unlabeled image, and failure with timestamps.

Useful flags: `--limit N` (process only the first N images instead of
all/3), `--model` (default `gpt-4.1-mini`, cheaper/faster for initial
testing; pass `--model gpt-4.1` for the larger model), `--image-detail`
(default `high` — polygon tracing needs fine detail, but costs more tokens
than `low`/`auto`), `--request-timeout` (default 120s — aborts and retries
a single image if its request stalls, so one bad connection can't hang the
whole run), `--max-retries`, `--retry-backoff`, `--max-output-tokens`,
`--visualize` (see below).

Requests are sent with `store=False`, which excludes them from the
Responses API's conversation-state storage. That is not a statement about
OpenAI's platform-level data retention or abuse-monitoring policies, which
are independent of this parameter — see OpenAI's data usage policy if that
matters for your images.

## Visualizing predictions

Add `--visualize` to save a PNG per processed image with the returned
polygons and labels drawn on top (each of the 23 part classes gets its own
color), for a quick visual sanity check before (or instead of) running full
quantitative scoring:
```powershell
python benchmark.py --images-dir "<TEST_IMAGES_DIR>" --labels-dir "<TEST_LABELS_DIR>" --output results_test.jsonl --test --visualize
```
Images are written to `<output-dir>/<output-stem>_viz/<image-stem>.png` by
default, or to `--viz-dir <path>` if given. Only images that were
successfully processed get a visualization; failed images are skipped
(with a warning logged) rather than aborting the run.

## Output format

One JSON object per line in `results.jsonl`:
```json
{
  "imageId": "0012",
  "fileName": "0012.jpg",
  "response": {
    "requestId": "...",
    "FileName": "0012.jpg",
    "status": "ok",
    "mode": "car_parts_only",
    "image": { "width": 1024, "height": 768 },
    "detections": [
      { "id": "0", "type": "part", "label": "front_bumper", "confidence": 0.91,
        "polygon": { "points": [{ "x": 120.5, "y": 340.0 }, ...] }, "matches": null }
    ],
    "message": null
  },
  "meta": {
    "model": "gpt-4.1-mini",
    "promptVersion": "v3",
    "usage": { "input_tokens": 1234, "output_tokens": 310, "total_tokens": 1544 },
    "rawResponse": { "...": "full OpenAI response, for reproducibility" },
    "attempts": 1,
    "elapsedSeconds": 4.1,
    "error": null
  }
}
```
`response` is exactly what production's `car_parts_only` mode returns
(reuses the real `Detection`/`EngineAnalyzeResponse` Pydantic models from
root `models.py`) — that's the half meant for the evaluation code. `meta`
is benchmark-only bookkeeping and isn't part of the production schema.

On failure for a given image, `response.status` is `"error"` with a
`message`, mirroring how the real Engine reports failures.
