#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask2Former vehicle-parts segmentation evaluation.

This is a combined inference + evaluation script designed to mirror the output
layout of evaluate_yolo_segmentation_coco.py while keeping the Mask2Former
23-class mapping and the existing YOLO-segmentation TXT ground-truth format.

Edit the four paths in USER CONFIGURATION and run:
    python evaluate_mask2former_like_yolo.py

Or override them from CMD / PowerShell:
    python evaluate_mask2former_like_yolo.py --images "..." --labels "..." --model "..." --out "..."

Outputs (same layout/names as the YOLO evaluator):
  report.html
  visual_comparisons.html
  metrics_overall.csv
  metrics_per_class.csv
  pixel_metrics_per_class.csv
  confusion_matrix.csv
  coco_predictions.json          # COCO-style prediction records (for inspection/export)
  plots/confusion_matrix.png
  plots/confusion_matrix_normalized.png
  plots/per_class_*.png
  visuals/*_comparison.jpg

Notes:
- Ground truth is YOLO segmentation TXT: class_id x1 y1 x2 y2 ... (normalized).
- AP metrics are computed directly from scored masks at IoU 0.50:0.95 using
  101-point interpolated AP. They are COCO-style AP metrics, but are not produced
  by pycocotools COCOeval because the source GT is YOLO TXT rather than COCO JSON.

Required packages:
    pip install torch transformers pillow opencv-python numpy pandas matplotlib
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# ============================================================
# USER CONFIGURATION — EDIT THESE FOUR PATHS
# ============================================================
MODEL_DIR = r"C:\Users\cs513\Desktop\CarCal\Engine\artifacts\car_parts\car_parts_M2F_model_NOA2"
TEST_IMAGES_DIR = r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Test_set\images"
GT_REFERENCES_DIR = r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Test_set\labels"
OUTPUT_DIR = r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\NOA2"

# Evaluation settings
DEVICE = ""               # "" = automatic, or "cpu", "cuda", "cuda:0"
PREDICTION_CONF = 0.001    # low threshold used for AP-style evaluation/export
VISUAL_CONF = 0.25         # used for confusion matrix, instance/pixel metrics and visuals
MASK_THRESHOLD = 0.50
MATCH_MASK_IOU = 0.50
MAX_DETECTIONS = 50
MAX_VISUALS = 60           # 0 = create comparisons for all images
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

# Exact 23-class mapping from the Mask2Former scripts/training setup.
MODEL_LABEL2ID = {
    "back_bumper": 0,
    "back_door": 1,
    "back_glass": 2,
    "back_left_door": 3,
    "back_left_light": 4,
    "back_light": 5,
    "back_right_door": 6,
    "back_right_light": 7,
    "front_bumper": 8,
    "front_door": 9,
    "front_glass": 10,
    "front_left_door": 11,
    "front_left_light": 12,
    "front_light": 13,
    "front_right_door": 14,
    "front_right_light": 15,
    "hood": 16,
    "left_mirror": 17,
    "object": 18,
    "right_mirror": 19,
    "tailgate": 20,
    "trunk": 21,
    "wheel": 22,
}
MODEL_ID2LABEL = {v: k for k, v in MODEL_LABEL2ID.items()}
CLASS_NAMES = [MODEL_ID2LABEL[i] for i in range(len(MODEL_ID2LABEL))]


@dataclass
class Instance:
    cls: int
    score: float
    mask: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def color_for_class(index: int) -> Tuple[int, int, int]:
    palette = [
        (230, 126, 34), (52, 152, 219), (155, 89, 182),
        (46, 204, 113), (241, 196, 15), (231, 76, 60),
        (26, 188, 156), (149, 165, 166), (52, 73, 94),
    ]
    return palette[index % len(palette)]


def find_label_file(image_path: Path, labels_dir: Path) -> Optional[Path]:
    p = labels_dir / f"{image_path.stem}.txt"
    return p if p.exists() else None


def get_image_files(images_dir: Path) -> List[Path]:
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_yolo_gt_instances(txt_path: Optional[Path], width: int, height: int) -> List[Instance]:
    """Load YOLO segmentation TXT into separate GT instances."""
    if txt_path is None:
        return []

    items: List[Instance] = []
    for line_no, raw in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            print(f"  Warning: {txt_path.name}:{line_no} has too few polygon values; skipped")
            continue
        try:
            cls_idx = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            print(f"  Warning: {txt_path.name}:{line_no} has invalid numeric values; skipped")
            continue
        if cls_idx not in MODEL_ID2LABEL:
            print(f"  Warning: {txt_path.name}:{line_no} class {cls_idx} is outside 0..22; skipped")
            continue
        if len(coords) % 2 != 0:
            print(f"  Warning: {txt_path.name}:{line_no} has an odd coordinate count; skipped")
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * width))
            y = int(round(coords[i + 1] * height))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            pts.append((x, y))
        if len(pts) < 3:
            continue

        m = Image.new("L", (width, height), 0)
        ImageDraw.Draw(m).polygon(pts, fill=1)
        mask = np.array(m, dtype=np.uint8).astype(bool)
        if not mask.any():
            continue
        items.append(Instance(cls=cls_idx, score=1.0, mask=mask, bbox_xyxy=bbox_from_mask(mask)))
    return items


def run_mask2former(model, processor, image: Image.Image, device: str) -> List[Instance]:
    """Run Mask2Former once and return scored instance masks in original image size."""
    width, height = image.size
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    processed = processor.post_process_instance_segmentation(
        outputs,
        threshold=PREDICTION_CONF,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=[(height, width)],
    )[0]

    seg = processed["segmentation"]
    if hasattr(seg, "detach"):
        seg = seg.detach().cpu().numpy()
    else:
        seg = np.asarray(seg)

    items: List[Instance] = []
    for s in processed["segments_info"]:
        cls_idx = int(s["label_id"])
        if cls_idx not in MODEL_ID2LABEL:
            continue
        score = float(s.get("score", 1.0))
        mask = (seg == int(s["id"]))
        if not mask.any():
            continue
        items.append(Instance(cls=cls_idx, score=score, mask=mask, bbox_xyxy=bbox_from_mask(mask)))

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:MAX_DETECTIONS]


def greedy_match_all_classes(gt: Sequence[Instance], pred: Sequence[Instance], threshold: float):
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
        used_g.add(gi); used_p.add(pi); matches.append((gi, pi, iou))
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
        used_g.add(gi); used_p.add(pi); matches.append((gi, pi, iou))
    return matches, [i for i in g_indices if i not in used_g], [i for i in p_indices if i not in used_p]


def overlay_instances(image: np.ndarray, instances: Sequence[Instance], class_names: Sequence[str], alpha: float = 0.42) -> np.ndarray:
    out = image.copy()
    layer = image.copy()
    for inst in instances:
        layer[inst.mask] = color_for_class(inst.cls)
    out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
    for inst in instances:
        color = color_for_class(inst.cls)
        contours, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, _, _ = cv2.boundingRect(largest)
            label = class_names[inst.cls]
            if inst.score < 0.999:
                label += f" {inst.score:.2f}"
            cv2.putText(out, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return out


def pixel_error_image(gt: Sequence[Instance], pred: Sequence[Instance], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    gt_union = np.zeros((h, w), dtype=bool)
    pred_union = np.zeros((h, w), dtype=bool)
    for x in gt: gt_union |= x.mask
    for x in pred: pred_union |= x.mask
    correct = gt_union & pred_union
    missed = gt_union & ~pred_union
    false_pred = pred_union & ~gt_union
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
    padded = [cv2.copyMakeBorder(p, 0, max_h-p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)) if p.shape[0] < max_h else p for p in panels]
    return cv2.hconcat(padded)


def save_confusion_matrix(matrix: np.ndarray, labels: Sequence[str], out_path: Path, normalize: bool = False) -> None:
    data = matrix.astype(float)
    if normalize:
        denom = data.sum(axis=1, keepdims=True)
        data = np.divide(data, denom, out=np.zeros_like(data), where=denom != 0)
    fig_w = max(8, len(labels) * 1.15)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.88))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    threshold = data.max()/2 if data.size and data.max() > 0 else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i,j]:.2f}" if normalize else str(int(data[i,j]))
            ax.text(j, i, text, ha="center", va="center", color="white" if data[i,j] > threshold else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def save_per_class_chart(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    vals = df[metric].fillna(0).astype(float).to_numpy()
    labels = df["class_name"].tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.25), 4.8))
    bars = ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel(metric); ax.set_title(f"Per-class {metric}"); ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.02, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def binary_mask_to_uncompressed_rle(mask: np.ndarray) -> dict:
    """COCO-style uncompressed RLE (Fortran order) for export/inspection."""
    pixels = mask.astype(np.uint8).flatten(order="F")
    counts = []
    prev = 0
    run = 0
    for px in pixels:
        px = int(px)
        if px == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = px
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated AP."""
    if recalls.size == 0:
        return 0.0
    levels = np.linspace(0, 1, 101)
    vals = []
    for r in levels:
        valid = precisions[recalls >= r]
        vals.append(float(valid.max()) if valid.size else 0.0)
    return float(np.mean(vals))


def evaluate_ap_for_class(records, cls_idx: int, iou_thr: float, area_range=None, max_det: Optional[int] = None):
    """Dataset-level scored-mask AP/recall for one class and IoU threshold."""
    if max_det is None:
        max_det = MAX_DETECTIONS
    gt_by_image: Dict[str, List[Instance]] = {}
    predictions = []
    total_gt = 0

    amin, amax = (None, None) if area_range is None else area_range
    def in_area(inst):
        a = float(inst.mask.sum())
        return (amin is None or a >= amin) and (amax is None or a < amax)

    for rec in records:
        gts = [g for g in rec["gt"] if g.cls == cls_idx and in_area(g)]
        preds = [p for p in rec["pred_all"] if p.cls == cls_idx and in_area(p)][:max_det]
        gt_by_image[rec["image_id"]] = gts
        total_gt += len(gts)
        for p in preds:
            predictions.append((p.score, rec["image_id"], p))

    if total_gt == 0:
        return np.nan, np.nan

    predictions.sort(key=lambda x: x[0], reverse=True)
    matched = {img: set() for img in gt_by_image}
    tp = np.zeros(len(predictions), dtype=float)
    fp = np.zeros(len(predictions), dtype=float)

    for i, (_, image_id, pred) in enumerate(predictions):
        gts = gt_by_image[image_id]
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if j in matched[image_id]:
                continue
            iou = mask_iou(gt.mask, pred.mask)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1.0
            matched[image_id].add(best_j)
        else:
            fp[i] = 1.0

    if len(predictions) == 0:
        return 0.0, 0.0
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    recalls = tp_c / total_gt
    precisions = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    return interpolated_ap(recalls, precisions), float(tp_c[-1] / total_gt)


def compute_ap_tables(records, class_names: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    all_ap_by_thr = {float(t): [] for t in IOU_THRESHOLDS}
    all_ar_by_thr = {float(t): [] for t in IOU_THRESHOLDS}

    for cls_idx, name in enumerate(class_names):
        aps, ars = [], []
        for thr in IOU_THRESHOLDS:
            ap, ar = evaluate_ap_for_class(records, cls_idx, float(thr))
            aps.append(ap); ars.append(ar)
            if not np.isnan(ap): all_ap_by_thr[float(thr)].append(ap)
            if not np.isnan(ar): all_ar_by_thr[float(thr)].append(ar)
        rows.append({
            "class_id": cls_idx,
            "class_name": name,
            "AP50_95": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else np.nan,
            "AP50": aps[0],
            "AP75": aps[5],
            "AR50_95": float(np.nanmean(ars)) if not np.all(np.isnan(ars)) else np.nan,
        })

    def mean_thr(d):
        vals = [x for arr in d.values() for x in arr]
        return float(np.mean(vals)) if vals else 0.0

    overall = {
        "segm_AP50_95": mean_thr(all_ap_by_thr),
        "segm_AP50": float(np.mean(all_ap_by_thr[float(IOU_THRESHOLDS[0])])) if all_ap_by_thr[float(IOU_THRESHOLDS[0])] else 0.0,
        "segm_AP75": float(np.mean(all_ap_by_thr[float(IOU_THRESHOLDS[5])])) if all_ap_by_thr[float(IOU_THRESHOLDS[5])] else 0.0,
    }

    size_ranges = {
        "small": (0, 32**2),
        "medium": (32**2, 96**2),
        "large": (96**2, None),
    }
    for label, area in size_ranges.items():
        vals_ap, vals_ar = [], []
        for cls_idx in range(len(class_names)):
            for thr in IOU_THRESHOLDS:
                ap, ar = evaluate_ap_for_class(records, cls_idx, float(thr), area_range=area)
                if not np.isnan(ap): vals_ap.append(ap)
                if not np.isnan(ar): vals_ar.append(ar)
        overall[f"segm_AP_{label}"] = float(np.mean(vals_ap)) if vals_ap else np.nan
        overall[f"segm_AR_{label}"] = float(np.mean(vals_ar)) if vals_ar else np.nan

    # AR at different max detections, averaged over IoUs/classes.
    for md in [1, 10, 100]:
        vals = []
        for cls_idx in range(len(class_names)):
            for thr in IOU_THRESHOLDS:
                _, ar = evaluate_ap_for_class(records, cls_idx, float(thr), max_det=md)
                if not np.isnan(ar): vals.append(ar)
        overall[f"segm_AR_{md}"] = float(np.mean(vals)) if vals else 0.0

    return pd.DataFrame(rows), overall


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


def generate_reports(out_dir: Path, model_path: Path, images_dir: Path, labels_dir: Path,
                     class_names: Sequence[str], overall_df: pd.DataFrame, per_class_df: pd.DataFrame,
                     pixel_df: pd.DataFrame, dataset_df: pd.DataFrame, plot_paths: Sequence[Path],
                     visual_rows: Sequence[Tuple[str, Path, float]], elapsed_seconds: float) -> Tuple[Path, Path]:
    report = out_dir / "report.html"
    visual_report = out_dir / "visual_comparisons.html"
    row = overall_df.iloc[0].to_dict()
    card_metrics = ["segm_AP50_95", "segm_AP50", "segm_AP75", "instance_precision", "instance_recall", "instance_F1", "pixel_mIoU", "pixel_mDice"]
    cards = []
    for key in card_metrics:
        value = row.get(key, np.nan)
        text = "—" if pd.isna(value) else f"{float(value):.3f}"
        cards.append(f"<div class='card'><span>{html.escape(key)}</span><strong>{text}</strong></div>")

    plots_html = "".join(f"<figure><img src='{image_rel(p, out_dir)}'><figcaption>{html.escape(p.stem)}</figcaption></figure>" for p in plot_paths if p.exists())
    common_css = """
    *{box-sizing:border-box} body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#18212b}
    header{background:#17212b;color:white;padding:24px 34px} header h1{margin:0 0 7px;font-size:24px} header p{margin:3px 0;color:#c8d1da;font-size:12px}
    main{padding:24px 32px;max-width:1500px;margin:auto} section{background:white;border:1px solid #e1e6eb;border-radius:12px;padding:20px;margin-bottom:18px}
    h2{font-size:17px;margin:0 0 14px;border-bottom:1px solid #e5e7eb;padding-bottom:9px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}
    .card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:13px}.card span{display:block;font-size:11px;color:#667085}.card strong{font-size:23px}
    table.data{border-collapse:collapse;width:100%;font-size:12px}table.data th{background:#eef2f6;text-align:left;padding:8px;border:1px solid #dce2e8}table.data td{padding:8px;border:1px solid #e2e7ec}
    .plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.plots figure{margin:0;border:1px solid #e1e6eb;border-radius:9px;padding:8px}.plots img{width:100%;display:block}.plots figcaption{text-align:center;font-size:11px;color:#667085;margin-top:5px}
    a.button{display:inline-block;background:#2563eb;color:white;text-decoration:none;padding:10px 15px;border-radius:8px;font-weight:600}.meta{font-size:12px;color:#596574}
    .visual{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:12px;margin-bottom:16px}.visual img{width:100%;display:block}.visual h3{margin:0 0 8px;font-size:14px}.score{font-size:12px;color:#667085}
    """
    report.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Mask2Former Segmentation Evaluation</title><style>{common_css}</style></head><body>
<header><h1>Mask2Former Segmentation Evaluation</h1><p>Model: {html.escape(model_path.name)}</p><p>Images: {html.escape(str(images_dir))}</p><p>GT labels: {html.escape(str(labels_dir))}</p></header>
<main>
<section><h2>Overall metrics</h2><div class='cards'>{''.join(cards)}</div><p class='meta'>AP values are computed directly from scored masks with 101-point interpolation over IoU 0.50:0.95; GT source is YOLO segmentation TXT.</p></section>
<section><h2>Dataset</h2>{metric_table_html(dataset_df)}</section>
<section><h2>AP-style + instance metrics by class</h2>{metric_table_html(per_class_df)}</section>
<section><h2>Pixel metrics by class</h2>{metric_table_html(pixel_df)}</section>
<section><h2>Charts</h2><div class='plots'>{plots_html}</div></section>
<section><h2>Visual comparisons</h2><a class='button' href='visual_comparisons.html'>Open Original / GT / Prediction / Errors</a><p class='meta'>{len(visual_rows)} examples · evaluation runtime {elapsed_seconds:.1f}s</p></section>
</main></body></html>""", encoding="utf-8")

    visual_blocks = []
    for name, path, quality in visual_rows:
        visual_blocks.append(f"<div class='visual'><h3>{html.escape(name)}</h3><div class='score'>Union mask IoU: {quality:.3f}</div><img src='{image_rel(path, out_dir)}'></div>")
    visual_report.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Visual comparisons</title><style>{common_css}</style></head><body>
<header><h1>Visual comparisons</h1><p>Original · Ground Truth · Prediction · Pixel errors</p></header><main><p><a class='button' href='report.html'>Back to metrics</a></p>{''.join(visual_blocks)}</main></body></html>""", encoding="utf-8")
    return report, visual_report


def parse_args():
    p = argparse.ArgumentParser(description="Mask2Former 23-class segmentation evaluator with YOLO-like outputs")
    p.add_argument("--images", default=TEST_IMAGES_DIR, help="Test images directory")
    p.add_argument("--labels", default=GT_REFERENCES_DIR, help="YOLO segmentation TXT ground-truth directory")
    p.add_argument("--model", default=MODEL_DIR, help="Mask2Former model directory")
    p.add_argument("--out", default=OUTPUT_DIR, help="Output directory")
    p.add_argument("--device", default=DEVICE, help='""=auto, cpu, cuda, cuda:0 ...')
    p.add_argument("--pred-conf", type=float, default=PREDICTION_CONF)
    p.add_argument("--visual-conf", type=float, default=VISUAL_CONF)
    p.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD)
    p.add_argument("--match-iou", type=float, default=MATCH_MASK_IOU)
    p.add_argument("--max-det", type=int, default=MAX_DETECTIONS)
    p.add_argument("--max-visuals", type=int, default=MAX_VISUALS)
    return p.parse_args()


def main() -> int:
    global PREDICTION_CONF, VISUAL_CONF, MASK_THRESHOLD, MATCH_MASK_IOU, MAX_DETECTIONS
    args = parse_args()
    PREDICTION_CONF = args.pred_conf
    VISUAL_CONF = args.visual_conf
    MASK_THRESHOLD = args.mask_threshold
    MATCH_MASK_IOU = args.match_iou
    MAX_DETECTIONS = args.max_det
    np.random.seed(SEED)

    images_dir = Path(args.images).expanduser().resolve()
    labels_dir = Path(args.labels).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out).expanduser().resolve())
    visuals_dir = ensure_dir(out_dir / "visuals")
    plots_dir = ensure_dir(out_dir / "plots")

    for path, label in [(images_dir, "test images directory"), (labels_dir, "GT labels directory"), (model_path, "model directory")]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    image_paths = get_image_files(images_dir)
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {images_dir}")

    missing_labels = [p.name for p in image_paths if find_label_file(p, labels_dir) is None]
    if missing_labels:
        raise FileNotFoundError(f"Missing GT TXT for {len(missing_labels)} images. First: {missing_labels[0]}")

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1/7] Loading Mask2Former processor/model...")
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(model_path))
    model.to(device); model.eval()
    print(f"  Device: {device}")
    print(f"  Classes: {len(CLASS_NAMES)}")

    # Validate model config using id2label.
    model_id2label = getattr(model.config, "id2label", None)

    if model_id2label:
        model_class_names = [
            model_id2label.get(i, model_id2label.get(str(i)))
            for i in range(len(model_id2label))
        ]

        if len(model_class_names) != len(CLASS_NAMES):
            raise ValueError(
                f"Model id2label contains {len(model_class_names)} classes, "
                f"expected {len(CLASS_NAMES)}."
            )

        if model_class_names != CLASS_NAMES:
            raise ValueError(
                "Model id2label does not match the expected class mapping.\n"
                f"Model classes: {model_class_names}\n"
                f"Expected classes: {CLASS_NAMES}"
            )

    print(f"[2/7] Inference + evaluation on {len(image_paths)} images...")
    start = time.time()
    n_classes = len(CLASS_NAMES)
    background_idx = n_classes
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    inst_counts = [{"TP":0,"FP":0,"FN":0,"ious":[],"dices":[]} for _ in range(n_classes)]
    pixel_counts = [{"TP_px":0,"FP_px":0,"FN_px":0,"TN_px":0} for _ in range(n_classes)]
    records = []
    export_predictions = []
    visual_candidates = []
    total_gt_instances = 0

    for idx, image_path in enumerate(image_paths, start=1):
        pil = Image.open(image_path).convert("RGB")
        w, h = pil.size
        gt = load_yolo_gt_instances(find_label_file(image_path, labels_dir), w, h)
        pred_all = run_mask2former(model, processor, pil, device)
        pred = [p for p in pred_all if p.score >= VISUAL_CONF]
        total_gt_instances += len(gt)

        records.append({"image_id": image_path.name, "gt": gt, "pred_all": pred_all})
        for p in pred_all:
            x1,y1,x2,y2 = p.bbox_xyxy
            export_predictions.append({
                "image_id": image_path.name,
                "category_id": int(p.cls),
                "category_name": CLASS_NAMES[p.cls],
                "segmentation": binary_mask_to_uncompressed_rle(p.mask),
                "score": float(p.score),
                "bbox": [float(x1), float(y1), float(max(0,x2-x1)), float(max(0,y2-y1))],
            })

        matches_any, unmatched_gt_any, unmatched_pred_any = greedy_match_all_classes(gt, pred, MATCH_MASK_IOU)
        for gi, pi, _ in matches_any: confusion[gt[gi].cls, pred[pi].cls] += 1
        for gi in unmatched_gt_any: confusion[gt[gi].cls, background_idx] += 1
        for pi in unmatched_pred_any: confusion[background_idx, pred[pi].cls] += 1

        for cls_idx in range(n_classes):
            matches, ug, up = greedy_match_same_class(gt, pred, cls_idx, MATCH_MASK_IOU)
            c = inst_counts[cls_idx]
            c["TP"] += len(matches); c["FN"] += len(ug); c["FP"] += len(up)
            for gi, pi, iou in matches:
                c["ious"].append(iou); c["dices"].append(mask_dice(gt[gi].mask, pred[pi].mask))

            gt_union = np.zeros((h,w), dtype=bool); pred_union = np.zeros((h,w), dtype=bool)
            for g in gt:
                if g.cls == cls_idx: gt_union |= g.mask
            for p in pred:
                if p.cls == cls_idx: pred_union |= p.mask
            pc = pixel_counts[cls_idx]
            pc["TP_px"] += int(np.logical_and(gt_union,pred_union).sum())
            pc["FP_px"] += int(np.logical_and(~gt_union,pred_union).sum())
            pc["FN_px"] += int(np.logical_and(gt_union,~pred_union).sum())
            pc["TN_px"] += int(np.logical_and(~gt_union,~pred_union).sum())

        gt_union = np.zeros((h,w), dtype=bool); pred_union = np.zeros((h,w), dtype=bool)
        for g in gt: gt_union |= g.mask
        for p in pred: pred_union |= p.mask
        quality = mask_iou(gt_union, pred_union)
        image_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        visual_candidates.append((quality, image_path.name, image_bgr, gt, pred))

        if idx % 25 == 0 or idx == len(image_paths):
            print(f"  {idx}/{len(image_paths)}")

    elapsed = time.time() - start

    print("[3/7] Computing AP-style mask metrics...")
    ap_df, ap_overall = compute_ap_tables(records, CLASS_NAMES)
    (out_dir / "coco_predictions.json").write_text(json.dumps(export_predictions, ensure_ascii=False), encoding="utf-8")

    print("[4/7] Computing instance/pixel tables...")
    instance_rows = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        c = inst_counts[cls_idx]
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        instance_rows.append({
            "class_id": cls_idx, "class_name": name,
            "GT_instances": tp+fn, "Pred_instances": tp+fp,
            "TP": tp, "FP": fp, "FN": fn,
            "Precision@matchIoU": precision, "Recall@matchIoU": recall, "F1@matchIoU": f1,
            "Mean_matched_mask_IoU": float(np.mean(c["ious"])) if c["ious"] else 0.0,
            "Mean_matched_mask_Dice": float(np.mean(c["dices"])) if c["dices"] else 0.0,
        })
    instance_df = pd.DataFrame(instance_rows)
    per_class_df = ap_df.merge(instance_df, on=["class_id","class_name"], how="outer")

    pixel_rows = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        c = pixel_counts[cls_idx]
        tp,fp,fn,tn = c["TP_px"],c["FP_px"],c["FN_px"],c["TN_px"]
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 0.0
        dice = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        accuracy = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else 0.0
        pixel_rows.append({
            "class_id": cls_idx, "class_name": name,
            "Pixel_Precision": precision, "Pixel_Recall": recall,
            "Pixel_IoU": iou, "Pixel_Dice": dice, "Pixel_Accuracy": accuracy,
            "TP_pixels": tp, "FP_pixels": fp, "FN_pixels": fn,
        })
    pixel_df = pd.DataFrame(pixel_rows)

    total_tp = int(instance_df["TP"].sum()); total_fp = int(instance_df["FP"].sum()); total_fn = int(instance_df["FN"].sum())
    inst_precision = total_tp/(total_tp+total_fp) if total_tp+total_fp else 0.0
    inst_recall = total_tp/(total_tp+total_fn) if total_tp+total_fn else 0.0
    inst_f1 = 2*inst_precision*inst_recall/(inst_precision+inst_recall) if inst_precision+inst_recall else 0.0
    overall = dict(ap_overall)
    overall.update({
        "instance_precision": inst_precision,
        "instance_recall": inst_recall,
        "instance_F1": inst_f1,
        "pixel_mIoU": float(pixel_df["Pixel_IoU"].mean()),
        "pixel_mDice": float(pixel_df["Pixel_Dice"].mean()),
        "visual_conf": float(VISUAL_CONF),
        "match_mask_iou": float(MATCH_MASK_IOU),
        "images": len(image_paths),
        "GT_instances": total_gt_instances,
        "predictions_COCO": len(export_predictions),
        "runtime_seconds": elapsed,
        "seconds_per_image": elapsed/len(image_paths),
        "metric_method": "Mask2Former scored-mask evaluation; 101-point AP over IoU 0.50:0.95; GT=YOLO segmentation TXT",
    })
    overall_df = pd.DataFrame([overall])
    dataset_df = pd.DataFrame([{
        "images": len(image_paths), "annotations": total_gt_instances, "classes": n_classes,
        "class_names": ", ".join(CLASS_NAMES), "model_classes": ", ".join(CLASS_NAMES),
    }])

    overall_df.to_csv(out_dir / "metrics_overall.csv", index=False)
    per_class_df.to_csv(out_dir / "metrics_per_class.csv", index=False)
    pixel_df.to_csv(out_dir / "pixel_metrics_per_class.csv", index=False)
    labels_bg = list(CLASS_NAMES) + ["background"]
    pd.DataFrame(confusion, index=labels_bg, columns=labels_bg).to_csv(out_dir / "confusion_matrix.csv")

    print("[5/7] Creating plots...")
    cm_path = plots_dir / "confusion_matrix.png"; cmn_path = plots_dir / "confusion_matrix_normalized.png"
    save_confusion_matrix(confusion, labels_bg, cm_path, normalize=False)
    save_confusion_matrix(confusion, labels_bg, cmn_path, normalize=True)
    chart_paths = [cm_path, cmn_path]
    for metric in ["AP50_95","AP50","AP75","Precision@matchIoU","Recall@matchIoU","F1@matchIoU","Mean_matched_mask_IoU"]:
        p = plots_dir / f"per_class_{metric.replace('@','_').replace(':','_')}.png"
        save_per_class_chart(per_class_df, metric, p)
        if p.exists(): chart_paths.append(p)
    for metric in ["Pixel_IoU","Pixel_Dice"]:
        p = plots_dir / f"per_class_{metric}.png"
        save_per_class_chart(pixel_df, metric, p)
        if p.exists(): chart_paths.append(p)

    print("[6/7] Creating visual comparison page...")
    visual_candidates.sort(key=lambda x: x[0])
    if args.max_visuals == 0 or args.max_visuals >= len(visual_candidates):
        selected = visual_candidates
    else:
        n = args.max_visuals; n_worst=max(1,n//2); n_best=max(1,n//4); n_mid=max(0,n-n_worst-n_best)
        worst=visual_candidates[:n_worst]; best=visual_candidates[-n_best:]
        middle_pool=visual_candidates[n_worst:len(visual_candidates)-n_best]
        middle=[middle_pool[i] for i in np.linspace(0,len(middle_pool)-1,n_mid,dtype=int)] if n_mid and middle_pool else []
        selected=worst+middle+best

    visual_rows = []
    for quality, file_name, image_bgr, gt, pred in selected:
        comparison = make_comparison(image_bgr, gt, pred, CLASS_NAMES)
        out_path = visuals_dir / f"{Path(file_name).stem}_comparison.jpg"
        cv2.imwrite(str(out_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])
        visual_rows.append((file_name, out_path, quality))

    print("[7/7] Writing HTML reports...")
    report_path, visual_report_path = generate_reports(
        out_dir, model_path, images_dir, labels_dir, CLASS_NAMES,
        overall_df, per_class_df, pixel_df, dataset_df, chart_paths, visual_rows, elapsed,
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