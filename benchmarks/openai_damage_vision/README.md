# OpenAI vision benchmark — damage segmentation

Sends each image in a COCO-format test set to the OpenAI Responses API,
asks it to segment visible vehicle damage into the same 6 classes as the
ground-truth annotations, and writes results in the exact
`EngineAnalyzeResponse` / `Detection` schema that
`Engine/Engine_methods.py`'s `process_damage_only()` returns in production
(see root `models.py`). This lets the OpenAI output be scored with the same
evaluation code as the local YOLO damage model, without touching the
production inference pipeline at all.

## 1. Place the data

This script does not ship the test set. Put it anywhere on disk:

```
<TEST_IMAGES_DIR>/           # the 374 test images (.jpg/.png/...)
<GROUND_TRUTH_JSON>          # instances_test2017.json (COCO instance-segmentation format)
```

The script only reads the `images` array from the COCO JSON (`id`,
`file_name`, `width`, `height`) to know which files to process — it does
not need or use the `annotations`/ground-truth boxes for generating
predictions.

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
python benchmark.py --images-dir "<TEST_IMAGES_DIR>" --coco-json "<GROUND_TRUTH_JSON>" --output results_test.jsonl --test
```

## 5. Run the full 374-image benchmark

```powershell
python benchmark.py --images-dir "<TEST_IMAGES_DIR>" --coco-json "<GROUND_TRUTH_JSON>" --output results.jsonl
```

If it's interrupted, just re-run the same command — already-processed
images (by COCO image `id`) are skipped. Add `--retry-errors` to
reprocess only the images that previously failed instead of skipping them
too. A log file (`results.jsonl` → `results.log`) captures every retry,
dropped detection, and failure with timestamps.

Useful flags: `--limit N` (process only the first N images instead of all/3),
`--model` (default `gpt-4.1-mini`, cheaper/faster for initial testing; pass
`--model gpt-4.1` for the larger model), `--image-detail` (default `high`
-- polygon tracing needs fine detail, but costs more tokens than
`low`/`auto`), `--request-timeout` (default 120s -- aborts and retries a
single image if its request stalls, so one bad connection can't hang the
whole run), `--max-retries`, `--retry-backoff`, `--max-output-tokens`,
`--visualize` (see below).

Requests are sent with `store=False`, which excludes them from the
Responses API's conversation-state storage. That is not a statement about
OpenAI's platform-level data retention or abuse-monitoring policies, which
are independent of this parameter — see OpenAI's data usage policy if that
matters for your images.

## Visualizing predictions

Add `--visualize` to save a PNG per processed image with the returned
polygons and labels drawn on top, for a quick visual sanity check before
(or instead of) running full quantitative scoring:
```powershell
python benchmark.py --images-dir "<TEST_IMAGES_DIR>" --coco-json "<GROUND_TRUTH_JSON>" --output results_test.jsonl --test --visualize
```
Images are written to `<output-dir>/<output-stem>_viz/<image-stem>.png` by
default (e.g. `results_test_viz/img1.png`), or to `--viz-dir <path>` if
given. Only images that were successfully processed get a visualization;
failed images are skipped (with a warning logged) rather than aborting the
run.

## Output format

One JSON object per line in `results.jsonl`:
```json
{
  "imageId": 12,
  "fileName": "0012.jpg",
  "response": {
    "requestId": "...",
    "FileName": "0012.jpg",
    "status": "ok",
    "mode": "damage_only",
    "image": { "width": 1024, "height": 768 },
    "detections": [
      { "id": "0", "type": "damage", "label": "scratch", "confidence": 0.82,
        "polygon": { "points": [{ "x": 120.5, "y": 340.0 }, ...] }, "matches": null }
    ],
    "message": null
  },
  "meta": {
    "model": "gpt-4.1",
    "promptVersion": "v2",
    "usage": { "input_tokens": 1234, "output_tokens": 210, "total_tokens": 1444 },
    "rawResponse": { "...": "full OpenAI response, for reproducibility" },
    "attempts": 1,
    "elapsedSeconds": 3.412,
    "error": null
  }
}
```
`response` is exactly what production's `damage_only` mode returns
(reuses the real `Detection`/`EngineAnalyzeResponse` Pydantic models from
root `models.py`) — that's the half meant for the evaluation code. `meta`
is benchmark-only bookkeeping and isn't part of the production schema.

On failure for a given image, `response.status` is `"error"` with a
`message`, mirroring how the real Engine reports failures.
