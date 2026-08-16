#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI polygon-segmentation evaluation on a COCO-format test set.

Edit the four paths in USER CONFIGURATION and run:
    python evaluate_openai_segmentation_coco.py

Outputs:
  report.html                  minimal metrics report
  visual_comparisons.html     Original | GT | Prediction | Pixel errors
  metrics_overall.csv
  metrics_per_class.csv
  pixel_metrics_per_class.csv
  confusion_matrix.csv/png
  coco_predictions.json
  visuals/*.jpg

Required packages:
  pip install pycocotools opencv-python numpy pandas matplotlib
"""

from __future__ import annotations

import html
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# USER CONFIGURATION — EDIT THESE FOUR PATHS
# ============================================================

# Original test images.
# These images are NOT sent to OpenAI again.
# They are used only for dimensions and visual comparison reports.
TEST_IMAGES_DIR = r"C:\Users\cs513\Desktop\Evaluation\Damages\test_set"

# COCO ground-truth annotations.
COCO_ANNOTATIONS_JSON = (
    r"C:\Users\cs513\Desktop\Evaluation\Damages"
    r"\test_set\instances_test2017.json"
)

# JSONL predictions already produced by benchmark.py.
OPENAI_RESULTS_JSONL = (
    r"C:\Users\cs513\Desktop\Evaluation\Damages"
    r"\results.jsonl"
)

# Output directory for OpenAI evaluation results.
OUTPUT_DIR = (
    r"C:\Users\cs513\Desktop\Evaluation\Damages"
    r"\OpenAI_Evaluation_results"
)

# Evaluation settings — kept equal to the YOLO evaluator.
VISUAL_CONF = 0.25
MATCH_MASK_IOU = 0.50
MAX_DETECTIONS = 50
MAX_VISUALS = 60          # Set 0 to create comparisons for all images.
SEED = 42


@dataclass
class Instance:
    cls: int
    score: float
    mask: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def color_for_class(index: int) -> Tuple[int, int, int]:
    # Deterministic BGR colors; no external palette dependency.
    palette = [
        (230, 126, 34), (52, 152, 219), (155, 89, 182),
        (46, 204, 113), (241, 196, 15), (231, 76, 60),
        (26, 188, 156), (149, 165, 166), (52, 73, 94),
    ]
    return palette[index % len(palette)]


def to_python_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def bbox_from_mask(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / union) if union > 0 else 0.0


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    denom = a.sum(dtype=np.float64) + b.sum(dtype=np.float64)
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def coco_segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def encode_binary_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def load_coco_ground_truth(coco: COCO, image_id: int, cat_id_to_index: Dict[int, int]) -> List[Instance]:
    info = coco.loadImgs([image_id])[0]
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    items: List[Instance] = []
    for ann in anns:
        cat_id = int(ann["category_id"])
        if cat_id not in cat_id_to_index:
            continue
        mask = coco_segmentation_to_mask(ann["segmentation"], int(info["height"]), int(info["width"]))
        if not mask.any():
            continue
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        items.append(Instance(
            cls=cat_id_to_index[cat_id],
            score=1.0,
            mask=mask,
            bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
        ))
    return items



def greedy_match_all_classes(gt: Sequence[Instance], pred: Sequence[Instance], threshold: float):
    """Greedy mask-IoU matching regardless of class, used for multiclass confusion matrix."""
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = mask_iou(g.mask, p.mask)
            if iou >= threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, iou))
    return matches, sorted(set(range(len(gt))) - used_g), sorted(set(range(len(pred))) - used_p)


def greedy_match_same_class(gt: Sequence[Instance], pred: Sequence[Instance], cls_idx: int, threshold: float):
    g_indices = [i for i, x in enumerate(gt) if x.cls == cls_idx]
    p_indices = [i for i, x in enumerate(pred) if x.cls == cls_idx]
    candidates = []
    for gi in g_indices:
        for pi in p_indices:
            iou = mask_iou(gt[gi].mask, pred[pi].mask)
            if iou >= threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_g, used_p, matches = set(), set(), []
    for iou, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, iou))
    return matches, [i for i in g_indices if i not in used_g], [i for i in p_indices if i not in used_p]


def overlay_instances(image: np.ndarray, instances: Sequence[Instance], class_names: Sequence[str], alpha: float = 0.42) -> np.ndarray:
    out = image.copy()
    layer = image.copy()
    for inst in instances:
        color = color_for_class(inst.cls)
        layer[inst.mask] = color
    out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    for inst in instances:
        color = color_for_class(inst.cls)
        contours, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, _, _ = cv2.boundingRect(largest)
            label = class_names[inst.cls] if 0 <= inst.cls < len(class_names) else str(inst.cls)
            if inst.score < 0.999:
                label += f" {inst.score:.2f}"
            cv2.putText(out, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return out


def pixel_error_image(gt: Sequence[Instance], pred: Sequence[Instance], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    gt_union = np.zeros((h, w), dtype=bool)
    pred_union = np.zeros((h, w), dtype=bool)
    for x in gt:
        gt_union |= x.mask
    for x in pred:
        pred_union |= x.mask

    correct = gt_union & pred_union
    missed = gt_union & ~pred_union
    false_pred = pred_union & ~gt_union

    # BGR: green correct, red missed, orange false prediction, dark background.
    out = np.full((h, w, 3), 28, dtype=np.uint8)
    out[correct] = (60, 170, 60)
    out[missed] = (45, 45, 230)
    out[false_pred] = (0, 150, 255)
    return out


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    h, w = image.shape[:2]
    bar = np.full((44, w, 3), 250, dtype=np.uint8)
    cv2.putText(bar, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (35, 35, 35), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def resize_for_panel(image: np.ndarray, panel_h: int = 420) -> np.ndarray:
    h, w = image.shape[:2]
    scale = panel_h / max(h, 1)
    return cv2.resize(image, (max(1, int(w * scale)), panel_h), interpolation=cv2.INTER_AREA)


def make_comparison(image: np.ndarray, gt: Sequence[Instance], pred: Sequence[Instance], class_names: Sequence[str]) -> np.ndarray:
    panels = [
        add_title(resize_for_panel(image), "Original"),
        add_title(resize_for_panel(overlay_instances(image, gt, class_names)), "Ground Truth overlay"),
        add_title(resize_for_panel(overlay_instances(image, pred, class_names)), "Prediction overlay"),
        add_title(resize_for_panel(pixel_error_image(gt, pred, image.shape[:2])), "Pixel errors: Green=correct, Red=missed, Orange=false"),
    ]
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            p = cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(p)
    return cv2.hconcat(padded)


def save_confusion_matrix(matrix: np.ndarray, labels: Sequence[str], out_path: Path, normalize: bool = False) -> None:
    data = matrix.astype(float)
    if normalize:
        denom = data.sum(axis=1, keepdims=True)
        data = np.divide(data, denom, out=np.zeros_like(data), where=denom != 0)
    fig_w = max(8, len(labels) * 1.15)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.88))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    threshold = data.max() / 2 if data.size and data.max() > 0 else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.2f}" if normalize else str(int(data[i, j]))
            ax.text(j, i, text, ha="center", va="center", color="white" if data[i, j] > threshold else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_per_class_chart(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    vals = df[metric].fillna(0).astype(float).to_numpy()
    labels = df["class_name"].tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.25), 4.8))
    bars = ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(f"Per-class {metric}")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def extract_coco_metrics(coco_eval: COCOeval) -> Dict[str, float]:
    keys = [
        "segm_AP50_95", "segm_AP50", "segm_AP75", "segm_AP_small", "segm_AP_medium", "segm_AP_large",
        "segm_AR_1", "segm_AR_10", "segm_AR_100", "segm_AR_small", "segm_AR_medium", "segm_AR_large",
    ]
    return {k: float(v) for k, v in zip(keys, coco_eval.stats.tolist())}


def coco_per_class_ap(coco_eval: COCOeval, category_ids: Sequence[int], class_names: Sequence[str]) -> pd.DataFrame:
    precision = coco_eval.eval["precision"]  # T x R x K x A x M
    recall = coco_eval.eval["recall"]        # T x K x A x M
    iou_thrs = coco_eval.params.iouThrs
    rows = []
    i50 = int(np.argmin(np.abs(iou_thrs - 0.50)))
    i75 = int(np.argmin(np.abs(iou_thrs - 0.75)))
    for k, (cat_id, name) in enumerate(zip(category_ids, class_names)):
        p_all = precision[:, :, k, 0, -1]
        p50 = precision[i50, :, k, 0, -1]
        p75 = precision[i75, :, k, 0, -1]
        r_all = recall[:, k, 0, -1]
        valid_all = p_all[p_all > -1]
        valid50 = p50[p50 > -1]
        valid75 = p75[p75 > -1]
        valid_r = r_all[r_all > -1]
        rows.append({
            "class_id": int(cat_id),
            "class_name": name,
            "AP50_95": float(valid_all.mean()) if valid_all.size else np.nan,
            "AP50": float(valid50.mean()) if valid50.size else np.nan,
            "AP75": float(valid75.mean()) if valid75.size else np.nan,
            "AR50_95": float(valid_r.mean()) if valid_r.size else np.nan,
        })
    return pd.DataFrame(rows)


def metric_table_html(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "<p>No data</p>"
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "—" if pd.isna(x) else f"{x:.{digits}f}")
    return d.to_html(index=False, classes="data", border=0, escape=True)


def image_rel(path: Path, report_dir: Path) -> str:
    return Path(os.path.relpath(path, report_dir)).as_posix()


def generate_reports(
    out_dir: Path,
    predictions_path: Path,
    images_dir: Path,
    annotations_path: Path,
    class_names: Sequence[str],
    overall_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    pixel_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    plot_paths: Sequence[Path],
    visual_rows: Sequence[Tuple[str, Path, float]],
    elapsed_seconds: float,
) -> Tuple[Path, Path]:
    report = out_dir / "report.html"
    visual_report = out_dir / "visual_comparisons.html"

    cards = []
    card_metrics = ["segm_AP50_95", "segm_AP50", "segm_AP75", "instance_precision", "instance_recall", "instance_F1", "pixel_mIoU", "pixel_mDice"]
    row = overall_df.iloc[0].to_dict()
    for key in card_metrics:
        value = row.get(key, np.nan)
        text = "—" if pd.isna(value) else f"{float(value):.3f}"
        cards.append(f"<div class='card'><span>{html.escape(key)}</span><strong>{text}</strong></div>")

    plots_html = "".join(
        f"<figure><img src='{image_rel(p, out_dir)}'><figcaption>{html.escape(p.stem)}</figcaption></figure>"
        for p in plot_paths if p.exists()
    )

    common_css = """
    *{box-sizing:border-box} body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#18212b}
    header{background:#17212b;color:white;padding:24px 34px} header h1{margin:0 0 7px;font-size:24px} header p{margin:3px 0;color:#c8d1da;font-size:12px}
    main{padding:24px 32px;max-width:1500px;margin:auto} section{background:white;border:1px solid #e1e6eb;border-radius:12px;padding:20px;margin-bottom:18px}
    h2{font-size:17px;margin:0 0 14px;border-bottom:1px solid #e5e7eb;padding-bottom:9px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}
    .card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:13px}.card span{display:block;font-size:11px;color:#667085}.card strong{font-size:23px}
    table.data{border-collapse:collapse;width:100%;font-size:12px}table.data th{background:#eef2f6;text-align:left;padding:8px;border:1px solid #dce2e8}table.data td{padding:8px;border:1px solid #e2e7ec}
    .plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.plots figure{margin:0;border:1px solid #e1e6eb;border-radius:9px;padding:8px}.plots img{width:100%;display:block}.plots figcaption{text-align:center;font-size:11px;color:#667085;margin-top:5px}
    a.button{display:inline-block;background:#2563eb;color:white;text-decoration:none;padding:10px 15px;border-radius:8px;font-weight:600}
    .meta{font-size:12px;color:#596574}.visual{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:12px;margin-bottom:16px}.visual img{width:100%;display:block}.visual h3{margin:0 0 8px;font-size:14px}.score{font-size:12px;color:#667085}
    """

    report.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>OpenAI Segmentation Evaluation</title><style>{common_css}</style></head><body>
<header><h1>OpenAI Segmentation Evaluation</h1><p>Predictions: {html.escape(predictions_path.name)}</p><p>Images: {html.escape(str(images_dir))}</p><p>COCO annotations: {html.escape(str(annotations_path))}</p></header>
<main>
<section><h2>Overall metrics</h2><div class='cards'>{''.join(cards)}</div></section>
<section><h2>Dataset</h2>{metric_table_html(dataset_df)}</section>
<section><h2>COCO + instance metrics by class</h2>{metric_table_html(per_class_df)}</section>
<section><h2>Pixel metrics by class</h2>{metric_table_html(pixel_df)}</section>
<section><h2>Charts</h2><div class='plots'>{plots_html}</div></section>
<section><h2>Visual comparisons</h2><a class='button' href='visual_comparisons.html'>Open Original / GT / Prediction / Errors</a><p class='meta'>{len(visual_rows)} examples · evaluation time {elapsed_seconds:.1f}s</p></section>
</main></body></html>""", encoding="utf-8")

    visual_items = []
    for file_name, path, quality in visual_rows:
        visual_items.append(f"<div class='visual'><h3>{html.escape(file_name)} <span class='score'>union IoU={quality:.3f}</span></h3><img loading='lazy' src='{image_rel(path, out_dir)}'></div>")
    visual_report.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Visual comparisons</title><style>{common_css}</style></head><body>
<header><h1>Visual comparisons</h1><p>Original · Ground Truth · Prediction · Pixel errors</p></header><main><p><a class='button' href='report.html'>Back to metrics</a></p>{''.join(visual_items)}</main></body></html>""", encoding="utf-8")
    return report, visual_report




def polygon_points_to_mask(points, height: int, width: int) -> Optional[np.ndarray]:
    """Rasterize one OpenAI polygon without simplifying or repairing it."""
    if not isinstance(points, list) or len(points) < 3:
        return None
    coords = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        x = min(max(x, 0.0), max(0.0, width - 1.0))
        y = min(max(y, 0.0), max(0.0, height - 1.0))
        coords.append((int(round(x)), int(round(y))))

    # Preserve the model's polygon; only remove consecutive duplicates caused by rounding.
    deduped = []
    for point in coords:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) >= 2 and deduped[0] == deduped[-1]:
        deduped.pop()
    if len(set(deduped)) < 3:
        return None

    polygon = np.asarray(deduped, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    binary = mask.astype(bool)
    return binary if binary.any() else None


def load_jsonl_predictions(path: Path):
    by_image_id = {}
    by_file_name = {}
    malformed_lines = 0
    duplicate_records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                print(f"  Warning: malformed JSONL line {line_number}; ignored.")
                continue
            image_id = record.get("imageId")
            file_name = record.get("fileName") or record.get("response", {}).get("FileName")
            if image_id is not None:
                try:
                    key = int(image_id)
                    if key in by_image_id:
                        duplicate_records += 1
                    by_image_id[key] = record
                except (TypeError, ValueError):
                    pass
            if file_name:
                key = str(file_name)
                if key in by_file_name:
                    duplicate_records += 1
                by_file_name[key] = record
    return by_image_id, by_file_name, malformed_lines, duplicate_records


def extract_openai_predictions(
    record: Optional[dict],
    expected_h: int,
    expected_w: int,
    name_to_coco_idx: Dict[str, int],
) -> Tuple[List[Instance], dict]:
    stats = {"record_error": 0, "invalid_detection": 0, "unknown_label": 0}
    if record is None:
        return [], stats

    response = record.get("response") or {}
    if response.get("status") != "ok":
        stats["record_error"] = 1
        return [], stats

    items: List[Instance] = []
    for detection in response.get("detections") or []:
        label_key = normalize_name(detection.get("label", ""))
        if label_key not in name_to_coco_idx:
            stats["unknown_label"] += 1
            continue
        try:
            score = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        score = min(max(score, 0.0), 1.0)

        polygon = detection.get("polygon") or {}
        mask = polygon_points_to_mask(polygon.get("points"), expected_h, expected_w)
        if mask is None:
            stats["invalid_detection"] += 1
            continue
        items.append(Instance(
            cls=name_to_coco_idx[label_key],
            score=score,
            mask=mask,
            bbox_xyxy=bbox_from_mask(mask),
        ))
    return items, stats



def main() -> int:
    np.random.seed(SEED)

    images_dir = Path(TEST_IMAGES_DIR).expanduser().resolve()
    annotations_path = Path(COCO_ANNOTATIONS_JSON).expanduser().resolve()
    predictions_input = Path(OPENAI_RESULTS_JSONL).expanduser().resolve()
    out_dir = ensure_dir(Path(OUTPUT_DIR).expanduser().resolve())
    visuals_dir = ensure_dir(out_dir / "visuals")
    plots_dir = ensure_dir(out_dir / "plots")

    for path, label in [
        (images_dir, "test images directory"),
        (annotations_path, "COCO annotations"),
        (predictions_input, "OpenAI predictions JSONL"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    print("[1/7] Loading COCO annotations...")
    coco = COCO(str(annotations_path))
    image_infos = sorted(coco.loadImgs(coco.getImgIds()), key=lambda x: x["file_name"])
    category_infos = sorted(coco.loadCats(coco.getCatIds()), key=lambda x: x["id"])
    category_ids = [int(x["id"]) for x in category_infos]
    class_names = [str(x["name"]) for x in category_infos]
    cat_id_to_index = {cat_id: i for i, cat_id in enumerate(category_ids)}
    name_to_coco_idx = {normalize_name(name): i for i, name in enumerate(class_names)}

    image_paths = []
    missing_images = []
    for info in image_infos:
        path = images_dir / info["file_name"]
        if path.exists():
            image_paths.append(path)
        else:
            missing_images.append(str(path))
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} COCO images are missing. First missing file: {missing_images[0]}"
        )

    print("[2/7] Loading OpenAI JSONL predictions...")
    by_image_id, by_file_name, malformed_lines, duplicate_records = load_jsonl_predictions(predictions_input)

    print(f"[3/7] Evaluating predictions on {len(image_paths)} images...")
    start = time.time()
    n_classes = len(class_names)
    background_idx = n_classes
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    inst_counts = [{"TP": 0, "FP": 0, "FN": 0, "ious": [], "dices": []} for _ in range(n_classes)]
    pixel_counts = [{"TP_px": 0, "FP_px": 0, "FN_px": 0, "TN_px": 0} for _ in range(n_classes)]
    coco_predictions = []
    visual_candidates = []
    processed = 0
    missing_prediction_records = 0
    failed_api_records = 0
    invalid_detections = 0
    unknown_labels = 0

    for info, image_path in zip(image_infos, image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = image.shape[:2]
        if h != int(info["height"]) or w != int(info["width"]):
            print(
                f"  Warning: size differs for {info['file_name']}: "
                f"file={w}x{h}, JSON={info['width']}x{info['height']}"
            )

        record = by_image_id.get(int(info["id"])) or by_file_name.get(str(info["file_name"]))
        if record is None:
            missing_prediction_records += 1
        pred_all, extraction_stats = extract_openai_predictions(record, h, w, name_to_coco_idx)
        failed_api_records += extraction_stats["record_error"]
        invalid_detections += extraction_stats["invalid_detection"]
        unknown_labels += extraction_stats["unknown_label"]

        gt = load_coco_ground_truth(coco, int(info["id"]), cat_id_to_index)

        # COCO AP uses every valid OpenAI prediction before VISUAL_CONF filtering.
        for p in pred_all[: MAX_DETECTIONS]:
            x1, y1, x2, y2 = p.bbox_xyxy
            coco_predictions.append({
                "image_id": int(info["id"]),
                "category_id": int(category_ids[p.cls]),
                "segmentation": encode_binary_mask(p.mask),
                "score": float(p.score),
                "bbox": [float(x1), float(y1), float(max(0, x2 - x1)), float(max(0, y2 - y1))],
            })

        pred = [p for p in pred_all[: MAX_DETECTIONS] if p.score >= VISUAL_CONF]

        matches_any, unmatched_gt_any, unmatched_pred_any = greedy_match_all_classes(gt, pred, MATCH_MASK_IOU)
        for gi, pi, _ in matches_any:
            confusion[gt[gi].cls, pred[pi].cls] += 1
        for gi in unmatched_gt_any:
            confusion[gt[gi].cls, background_idx] += 1
        for pi in unmatched_pred_any:
            confusion[background_idx, pred[pi].cls] += 1

        for cls_idx in range(n_classes):
            matches, unmatched_g, unmatched_p = greedy_match_same_class(gt, pred, cls_idx, MATCH_MASK_IOU)
            inst_counts[cls_idx]["TP"] += len(matches)
            inst_counts[cls_idx]["FN"] += len(unmatched_g)
            inst_counts[cls_idx]["FP"] += len(unmatched_p)
            for gi, pi, iou in matches:
                inst_counts[cls_idx]["ious"].append(iou)
                inst_counts[cls_idx]["dices"].append(mask_dice(gt[gi].mask, pred[pi].mask))

            gt_union = np.zeros((h, w), dtype=bool)
            pred_union = np.zeros((h, w), dtype=bool)
            for g in gt:
                if g.cls == cls_idx:
                    gt_union |= g.mask
            for p in pred:
                if p.cls == cls_idx:
                    pred_union |= p.mask
            pc = pixel_counts[cls_idx]
            pc["TP_px"] += int(np.logical_and(gt_union, pred_union).sum())
            pc["FP_px"] += int(np.logical_and(~gt_union, pred_union).sum())
            pc["FN_px"] += int(np.logical_and(gt_union, ~pred_union).sum())
            pc["TN_px"] += int(np.logical_and(~gt_union, ~pred_union).sum())

        gt_union = np.zeros((h, w), dtype=bool)
        pred_union = np.zeros((h, w), dtype=bool)
        for g in gt:
            gt_union |= g.mask
        for p in pred:
            pred_union |= p.mask
        quality = mask_iou(gt_union, pred_union)
        visual_candidates.append((quality, info["file_name"], image.copy(), gt, pred))

        processed += 1
        if processed % 25 == 0 or processed == len(image_paths):
            print(f"  {processed}/{len(image_paths)}")

        del image, pred_all, pred

    if processed != len(image_paths):
        raise RuntimeError(
            f"Only {processed} of {len(image_paths)} images were processed. "
            "The report was not created because it would be incomplete."
        )

    print(f"Successfully processed all {processed} images.")
    elapsed = time.time() - start

    print("[4/7] COCO segmentation evaluation...")
    predictions_path = out_dir / "coco_predictions.json"
    predictions_path.write_text(json.dumps(coco_predictions, ensure_ascii=False), encoding="utf-8")
    if not coco_predictions:
        raise RuntimeError("The OpenAI result file contains no valid segmentation predictions.")
    coco_dt = coco.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco, coco_dt, iouType="segm")
    coco_eval.params.imgIds = [int(x["id"]) for x in image_infos]
    coco_eval.params.catIds = category_ids
    coco_eval.params.maxDets = [1, 10, MAX_DETECTIONS]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = extract_coco_metrics(coco_eval)
    coco_class_df = coco_per_class_ap(coco_eval, category_ids, class_names)

    print("[5/7] Computing tables and plots...")
    instance_rows = []
    for idx, name in enumerate(class_names):
        counts = inst_counts[idx]
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        instance_rows.append({
            "class_id": category_ids[idx],
            "class_name": name,
            "GT_instances": tp + fn,
            "Pred_instances": tp + fp,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision@matchIoU": precision,
            "Recall@matchIoU": recall,
            "F1@matchIoU": f1,
            "Mean_matched_mask_IoU": float(np.mean(counts["ious"])) if counts["ious"] else 0.0,
            "Mean_matched_mask_Dice": float(np.mean(counts["dices"])) if counts["dices"] else 0.0,
        })
    instance_df = pd.DataFrame(instance_rows)
    per_class_df = coco_class_df.merge(instance_df, on=["class_id", "class_name"], how="outer")

    pixel_rows = []
    for idx, name in enumerate(class_names):
        counts = pixel_counts[idx]
        tp, fp, fn, tn = counts["TP_px"], counts["FP_px"], counts["FN_px"], counts["TN_px"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
        pixel_rows.append({
            "class_id": category_ids[idx],
            "class_name": name,
            "Pixel_Precision": precision,
            "Pixel_Recall": recall,
            "Pixel_IoU": iou,
            "Pixel_Dice": dice,
            "Pixel_Accuracy": accuracy,
            "TP_pixels": tp,
            "FP_pixels": fp,
            "FN_pixels": fn,
        })
    pixel_df = pd.DataFrame(pixel_rows)

    total_tp = int(instance_df["TP"].sum())
    total_fp = int(instance_df["FP"].sum())
    total_fn = int(instance_df["FN"].sum())
    inst_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    inst_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    inst_f1 = 2 * inst_precision * inst_recall / (inst_precision + inst_recall) if inst_precision + inst_recall else 0.0
    overall = dict(coco_metrics)
    overall.update({
        "instance_precision": inst_precision,
        "instance_recall": inst_recall,
        "instance_F1": inst_f1,
        "pixel_mIoU": float(pixel_df["Pixel_IoU"].mean()),
        "pixel_mDice": float(pixel_df["Pixel_Dice"].mean()),
        "visual_conf": float(VISUAL_CONF),
        "match_mask_iou": float(MATCH_MASK_IOU),
        "images": len(image_infos),
        "GT_instances": len(coco.dataset.get("annotations", [])),
        "predictions_COCO": len(coco_predictions),
        "runtime_seconds": elapsed,
        "seconds_per_image": elapsed / len(image_infos),
        "missing_prediction_records": missing_prediction_records,
        "failed_api_records": failed_api_records,
        "invalid_detections": invalid_detections,
        "unknown_labels": unknown_labels,
        "malformed_jsonl_lines": malformed_lines,
        "duplicate_prediction_records": duplicate_records,
    })
    overall_df = pd.DataFrame([overall])

    dataset_df = pd.DataFrame([{
        "images": len(image_infos),
        "annotations": len(coco.dataset.get("annotations", [])),
        "classes": n_classes,
        "class_names": ", ".join(class_names),
        "prediction_classes": ", ".join(class_names),
        "prediction_source": predictions_input.name,
    }])

    overall_df.to_csv(out_dir / "metrics_overall.csv", index=False)
    per_class_df.to_csv(out_dir / "metrics_per_class.csv", index=False)
    pixel_df.to_csv(out_dir / "pixel_metrics_per_class.csv", index=False)
    labels_bg = list(class_names) + ["background"]
    pd.DataFrame(confusion, index=labels_bg, columns=labels_bg).to_csv(out_dir / "confusion_matrix.csv")

    cm_path = plots_dir / "confusion_matrix.png"
    cmn_path = plots_dir / "confusion_matrix_normalized.png"
    save_confusion_matrix(confusion, labels_bg, cm_path, normalize=False)
    save_confusion_matrix(confusion, labels_bg, cmn_path, normalize=True)
    chart_paths = [cm_path, cmn_path]
    for metric in [
        "AP50_95", "AP50", "AP75", "Precision@matchIoU", "Recall@matchIoU",
        "F1@matchIoU", "Mean_matched_mask_IoU",
    ]:
        path = plots_dir / f"per_class_{metric.replace('@', '_').replace(':', '_')}.png"
        save_per_class_chart(per_class_df, metric, path)
        if path.exists():
            chart_paths.append(path)
    for metric in ["Pixel_IoU", "Pixel_Dice"]:
        path = plots_dir / f"per_class_{metric}.png"
        save_per_class_chart(pixel_df, metric, path)
        if path.exists():
            chart_paths.append(path)

    print("[6/7] Creating visual comparison page...")
    visual_candidates.sort(key=lambda x: x[0])
    if MAX_VISUALS == 0 or MAX_VISUALS >= len(visual_candidates):
        selected = visual_candidates
    else:
        n = MAX_VISUALS
        n_worst = max(1, n // 2)
        n_best = max(1, n // 4)
        n_mid = max(0, n - n_worst - n_best)
        worst = visual_candidates[:n_worst]
        best = visual_candidates[-n_best:]
        middle_pool = visual_candidates[n_worst:len(visual_candidates) - n_best]
        if n_mid and middle_pool:
            positions = np.linspace(0, len(middle_pool) - 1, n_mid, dtype=int)
            middle = [middle_pool[i] for i in positions]
        else:
            middle = []
        selected = worst + middle + best

    visual_rows = []
    for quality, file_name, image, gt, pred in selected:
        comparison = make_comparison(image, gt, pred, class_names)
        out_path = visuals_dir / f"{Path(file_name).stem}_comparison.jpg"
        cv2.imwrite(str(out_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])
        visual_rows.append((file_name, out_path, quality))

    print("[7/7] Writing HTML reports...")
    report_path, visual_report_path = generate_reports(
        out_dir,
        predictions_input,
        images_dir,
        annotations_path,
        class_names,
        overall_df,
        per_class_df,
        pixel_df,
        dataset_df,
        chart_paths,
        visual_rows,
        elapsed,
    )

    print("\nDone.")
    print(f"Main report:        {report_path}")
    print(f"Visual comparisons: {visual_report_path}")
    print(f"Output directory:   {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
