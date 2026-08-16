"""Standalone OpenAI vision benchmark for the CarCal car-parts segmentation
task.

Sends each test image to the OpenAI Responses API and asks it to segment
visible vehicle parts into the same 14 classes the production Mask2Former
model was trained on (Engine/artifacts/car_parts/car_parts_M2F_model/config.json's
id2label -- see schema.py). Predictions are converted into the exact
EngineAnalyzeResponse / Detection schema that Engine/Engine_methods.py's
process_parts_only() returns in production (see root models.py), so the two
can later be scored with the same evaluation code.

Fully standalone: does not import or modify anything in Client/, API/,
Engine/, or React_Client/ -- only the shared, dependency-free contract in
root models.py. The production inference pipeline is untouched.

Ground truth for this test set is YOLO-segmentation .txt files (one per
image, "class_id x1 y1 x2 y2 ... xn yn" normalized), not COCO JSON. Since
predictions never need the ground truth, this script only checks that a
matching .txt exists per image (to know it's part of the labeled test set)
-- it never reads the .txt's contents. See README.md for the expected
folder layout.

Usage:
    python benchmark.py --images-dir <dir> --labels-dir <dir> --output results.jsonl --test
    python benchmark.py --images-dir <dir> --labels-dir <dir> --output results.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from convert import build_error_response, build_response
from schema import PART_CLASSES, PROMPT_TEXT, PROMPT_VERSION, RESPONSE_SCHEMA
from visualize import draw_detections

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing the test images")
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Directory containing the YOLO-format .txt ground-truth files, one per image, "
        "same stem as the image (default: same as --images-dir). Only checked for "
        "existence per image, never read -- ground truth isn't needed to generate predictions.",
    )
    parser.add_argument("--output", default="results.jsonl", help="Output JSONL path (default: results.jsonl)")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI vision model (default: gpt-4.1-mini; pass --model gpt-4.1 for the larger model)",
    )
    parser.add_argument("--test", action="store_true", help="Only process the first 3 images")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per image on API failure (default: 3)")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Seconds to wait before a retry, multiplied by the attempt number (default: 2.0)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="On resume, reprocess images that previously failed instead of skipping them",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=4000, help="Max output tokens per request (default: 4000)"
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Seconds before a single OpenAI request is aborted and treated as a failure "
        "(default: 120). Without this, a stalled network request can hang indefinitely "
        "instead of triggering the retry logic.",
    )
    parser.add_argument(
        "--image-detail",
        default="high",
        choices=["low", "high", "auto"],
        help="OpenAI image input detail level (default: high -- polygon tracing needs fine detail; costs more tokens than low/auto)",
    )
    parser.add_argument("--log-file", default=None, help="Log file path (default: <output>.log)")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a PNG per processed image with the returned polygons and labels drawn on it, for visual QA",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Directory for --visualize output (default: <output-dir>/<output-stem>_viz/)",
    )
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("openai_car_parts_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    return logger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_test_images(images_dir: str, labels_dir: str, logger: logging.Logger) -> list[dict]:
    """Scans images_dir for image files that have a matching YOLO-format
    label .txt (same stem) in labels_dir. That .txt's presence marks an
    image as part of the labeled test set -- its contents (the ground
    truth polygons) are never read here, only used later for evaluation.
    Each image's "id" is its filename stem (YOLO ground truth has no
    numeric image id like COCO does).
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    image_files = sorted(
        p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    items = []
    skipped = []
    for img_path in image_files:
        if (labels_path / f"{img_path.stem}.txt").exists():
            items.append({"id": img_path.stem, "file_name": img_path.name})
        else:
            skipped.append(img_path.name)

    if not items:
        raise ValueError(
            f"None of the {len(image_files)} image(s) in {images_dir} have a matching "
            f".txt in {labels_dir}"
        )
    if skipped:
        preview = ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else "")
        logger.warning(
            f"{len(skipped)} image(s) in {images_dir} have no matching .txt in "
            f"{labels_dir} and were skipped: {preview}"
        )

    return items


def load_existing_results(output_path: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not output_path.exists():
        return existing
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            existing[record["imageId"]] = record
    return existing


def write_results(output_path: Path, results: dict[str, dict]) -> None:
    """Rewrites the whole file atomically on every image, keyed by imageId.
    A few hundred images is small enough that this is simpler and safer
    than streaming appends (no duplicate lines on --retry-errors, and the
    file is always valid JSONL even if the process is killed mid-run)."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for image_id in sorted(results):
            f.write(json.dumps(results[image_id]) + "\n")
    tmp_path.replace(output_path)


def call_openai(
    client, model: str, image_path: Path, max_output_tokens: int, image_detail: str, request_timeout: float
):
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    return client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT_TEXT},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": image_detail},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "car_parts_detections",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
        # Without an explicit timeout, a stalled connection can hang this
        # call indefinitely instead of raising -- which would never reach
        # the retry loop below and would block the whole run on one image.
        timeout=request_timeout,
        # temperature=0 for reproducibility. Only meaningful for standard
        # chat-style models (e.g. the gpt-4.x family); reasoning-family
        # models (o-series) reject this parameter, so remove it if --model
        # is ever pointed at one of those.
        temperature=0,
        # store=False: this request is excluded from the Responses API's
        # conversation-state storage. This is NOT a claim about OpenAI's
        # platform-level data retention/abuse-monitoring policies, which
        # are independent of this parameter -- see OpenAI's data usage
        # policy for what those entail.
        store=False,
    )


def _safe_dump(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def process_image(
    client,
    img_meta: dict,
    args: argparse.Namespace,
    logger: logging.Logger,
    viz_dir: Path | None = None,
) -> dict:
    image_id = img_meta["id"]
    file_name = img_meta["file_name"]
    image_path = Path(args.images_dir) / file_name
    request_id = str(uuid.uuid4())
    started = time.time()

    meta = {
        "model": args.model,
        "promptVersion": PROMPT_VERSION,
        "usage": None,
        "rawResponse": None,
        "attempts": 0,
        "elapsedSeconds": None,
        "error": None,
    }

    def finish(response_obj, error: str | None = None) -> dict:
        meta["elapsedSeconds"] = round(time.time() - started, 3)
        meta["error"] = error
        return {
            "imageId": image_id,
            "fileName": file_name,
            "response": response_obj.model_dump(),
            "meta": meta,
        }

    if not image_path.exists():
        msg = f"Image file not found: {image_path}"
        logger.error(f"[{file_name}] {msg}")
        return finish(build_error_response(request_id, file_name, msg), error=msg)

    try:
        with Image.open(image_path) as im:
            width, height = im.size
    except Exception as exc:
        msg = f"Failed to open image: {exc}"
        logger.error(f"[{file_name}] {msg}")
        return finish(build_error_response(request_id, file_name, msg), error=msg)

    last_error: Exception | None = None
    response = None
    for attempt in range(1, args.max_retries + 1):
        meta["attempts"] = attempt
        try:
            response = call_openai(
                client, args.model, image_path, args.max_output_tokens, args.image_detail, args.request_timeout
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            logger.warning(f"[{file_name}] attempt {attempt}/{args.max_retries} failed: {exc}")
            if attempt < args.max_retries:
                time.sleep(args.retry_backoff * attempt)

    if response is None:
        msg = f"OpenAI request failed after {args.max_retries} attempts: {last_error}"
        logger.error(f"[{file_name}] {msg}")
        return finish(build_error_response(request_id, file_name, msg), error=msg)

    meta["usage"] = _safe_dump(getattr(response, "usage", None))
    meta["rawResponse"] = _safe_dump(response)

    try:
        parsed = json.loads(response.output_text)
    except Exception as exc:
        msg = f"Could not parse structured output as JSON: {exc}"
        logger.error(f"[{file_name}] {msg}")
        return finish(build_error_response(request_id, file_name, msg), error=msg)

    def warn(text: str) -> None:
        logger.warning(f"[{file_name}] {text}")

    result = build_response(request_id, file_name, width, height, parsed.get("detections"), warn)
    logger.info(f"[{file_name}] ok: {len(result.detections)} detection(s)")

    if viz_dir is not None:
        try:
            viz_path = viz_dir / f"{Path(file_name).stem}.png"
            draw_detections(
                image_path,
                [d.model_dump() for d in result.detections],
                viz_path,
                PART_CLASSES,
            )
        except Exception as exc:
            logger.warning(f"[{file_name}] failed to save visualization: {exc}")

    return finish(result)


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else output_path.with_suffix(".log")
    logger = setup_logger(log_path)

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("The 'openai' package is not installed. Run: pip install -r requirements.txt")
        raise SystemExit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise SystemExit(1)

    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    labels_dir = args.labels_dir if args.labels_dir else args.images_dir
    images = load_test_images(args.images_dir, labels_dir, logger)
    if args.test:
        images = images[:3]
    elif args.limit is not None:
        images = images[: args.limit]

    results = load_existing_results(output_path)
    retryable_ids = (
        {img_id for img_id, rec in results.items() if rec["response"]["status"] == "error"}
        if args.retry_errors
        else set()
    )
    done_ids = set(results) - retryable_ids
    pending = [img for img in images if img["id"] not in done_ids]

    viz_dir = None
    if args.visualize:
        viz_dir = Path(args.viz_dir) if args.viz_dir else output_path.parent / f"{output_path.stem}_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"{len(images)} image(s) in scope, {len(images) - len(pending)} already done, "
        f"{len(pending)} to process (model={args.model}, prompt={PROMPT_VERSION})"
        + (f", visualizations -> {viz_dir}" if viz_dir else "")
    )

    ok_count = 0
    error_count = 0
    for img_meta in tqdm(pending, desc="benchmark"):
        record = process_image(client, img_meta, args, logger, viz_dir=viz_dir)
        results[record["imageId"]] = record
        write_results(output_path, results)
        if record["response"]["status"] == "ok":
            ok_count += 1
        else:
            error_count += 1

    logger.info(f"Done. {ok_count} ok, {error_count} failed this run. Results: {output_path}")


if __name__ == "__main__":
    main()
