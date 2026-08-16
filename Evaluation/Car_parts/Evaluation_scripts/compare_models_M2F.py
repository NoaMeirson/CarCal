#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# USER CONFIGURATION
# ============================================================

MODELS = {
    # Point each entry to the OUTPUT_DIR produced by evaluate_mask2former_like_yolo.py
    "M2F_Model_NOA": Path(r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\NOA"),
    "M2F_Model_AVRAHAM": Path(r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\AVRAHAM"),
    "M2F_Model_NOA2": Path(r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\NOA2"),
    "M2F_Model_ChatGPT": Path(r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\ChatGPT"),
    
}


OUTPUT_DIR = Path(r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\COMPARE_MODELS")


# ============================================================
# METRICS TO COMPARE
# ============================================================

OVERALL_METRICS = [
    # Official pycocotools COCOeval segmentation metrics
    "segm_AP50_95",
    "segm_AP50",
    "segm_AP75",
    "segm_AP_small",
    "segm_AP_medium",
    "segm_AP_large",
    "segm_AR_1",
    "segm_AR_10",
    "segm_AR_100",
    "segm_AR_small",
    "segm_AR_medium",
    "segm_AR_large",

    # Additional instance/pixel diagnostics
    "instance_precision",
    "instance_recall",
    "instance_F1",
    "pixel_mIoU",
    "pixel_mDice",
]

PER_CLASS_METRICS = [
    "AP50_95",
    "AP50",
    "AP75",
    "AR50_95",
    "Precision@matchIoU",
    "Recall@matchIoU",
    "F1@matchIoU",
    "Mean_matched_mask_IoU",
    "Mean_matched_mask_Dice",
    "GT_instances",
    "Pred_instances",
    "TP",
    "FP",
    "FN",
]

PIXEL_METRICS = [
    "Pixel_Precision",
    "Pixel_Recall",
    "Pixel_IoU",
    "Pixel_Dice",
    "Pixel_Accuracy",
    "TP_pixels",
    "FP_pixels",
    "FN_pixels",
]


# ============================================================
# HELPERS
# ============================================================

def require_file(folder: Path, filename: str) -> Path:
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    return path


def safe_columns(df: pd.DataFrame, wanted_columns):
    return [c for c in wanted_columns if c in df.columns]


def validate_same_classes(dataframes, table_name):
    reference_name = None
    reference_classes = None

    for model_name, df in dataframes.items():

        if "class_name" not in df.columns:
            continue

        current = set(df["class_name"].astype(str))

        if reference_classes is None:
            reference_classes = current
            reference_name = model_name
            continue

        if current != reference_classes:
            raise ValueError(
                f"Class mismatch in {table_name}: "
                f"{model_name} != {reference_name}"
            )


# ============================================================
# LOAD RESULTS
# ============================================================

def load_all_results():

    overall = {}
    per_class = {}
    pixel = {}

    for model_name, folder in MODELS.items():

        overall_path = require_file(
            folder,
            "metrics_overall.csv"
        )

        per_class_path = require_file(
            folder,
            "metrics_per_class.csv"
        )

        pixel_path = require_file(
            folder,
            "pixel_metrics_per_class.csv"
        )

        overall[model_name] = pd.read_csv(overall_path)
        per_class[model_name] = pd.read_csv(per_class_path)
        pixel[model_name] = pd.read_csv(pixel_path)

    return overall, per_class, pixel


# ============================================================
# OVERALL COMPARISON
# ============================================================

def build_overall_comparison(overall_data):

    rows = []

    for model_name, df in overall_data.items():

        if len(df) != 1:
            raise ValueError(
                f"{model_name}: metrics_overall.csv "
                f"should contain exactly one row."
            )

        source = df.iloc[0]

        row = {
            "model": model_name
        }

        for metric in OVERALL_METRICS:
            row[metric] = (
                source[metric]
                if metric in source.index
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# OVERALL RANKINGS
# ============================================================

def build_overall_rankings(overall_df):

    rows = []

    for metric in OVERALL_METRICS:

        if metric not in overall_df.columns:
            continue

        values = pd.to_numeric(
            overall_df[metric],
            errors="coerce"
        )

        ranks = values.rank(
            ascending=False,
            method="min"
        )

        for model, value, rank in zip(
            overall_df["model"],
            values,
            ranks
        ):
            rows.append({
                "metric": metric,
                "model": model,
                "value": value,
                "rank": rank,
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["metric", "rank", "model"]
        )

    return result


# ============================================================
# BEST MODEL FOR EACH OVERALL METRIC
# ============================================================

def build_best_overall(overall_df):

    rows = []

    for metric in OVERALL_METRICS:

        if metric not in overall_df.columns:
            continue

        values = pd.to_numeric(
            overall_df[metric],
            errors="coerce"
        )

        if values.notna().sum() == 0:
            continue

        best_index = values.idxmax()

        rows.append({
            "metric": metric,
            "best_model": overall_df.loc[
                best_index,
                "model"
            ],
            "best_value": values.loc[
                best_index
            ],
        })

    return pd.DataFrame(rows)


# ============================================================
# PER-CLASS COMPARISON
# ============================================================

def build_per_class_comparison(per_class_data):

    validate_same_classes(
        per_class_data,
        "metrics_per_class.csv"
    )

    rows = []

    for model_name, df in per_class_data.items():

        for _, source in df.iterrows():

            row = {
                "model": model_name,
                "class_id": source.get(
                    "class_id",
                    np.nan
                ),
                "class_name": source.get(
                    "class_name",
                    ""
                ),
            }

            for metric in PER_CLASS_METRICS:
                row[metric] = source.get(
                    metric,
                    np.nan
                )

            rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["class_name", "model"]
    )


# ============================================================
# BEST MODEL PER CLASS / METRIC
# ============================================================

def build_best_per_class(per_class_df):

    higher_is_better = [
        "AP50_95",
        "AP50",
        "AP75",
        "AR50_95",
        "Precision@matchIoU",
        "Recall@matchIoU",
        "F1@matchIoU",
        "Mean_matched_mask_IoU",
        "Mean_matched_mask_Dice",
    ]

    rows = []

    for class_name, group in per_class_df.groupby(
        "class_name"
    ):

        for metric in higher_is_better:

            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce"
            )

            if values.notna().sum() == 0:
                continue

            best_index = values.idxmax()

            rows.append({
                "class_name": class_name,
                "metric": metric,
                "best_model": group.loc[
                    best_index,
                    "model"
                ],
                "best_value": values.loc[
                    best_index
                ],
            })

    return pd.DataFrame(rows)


# ============================================================
# PIXEL COMPARISON
# ============================================================

def build_pixel_comparison(pixel_data):

    validate_same_classes(
        pixel_data,
        "pixel_metrics_per_class.csv"
    )

    rows = []

    for model_name, df in pixel_data.items():

        for _, source in df.iterrows():

            row = {
                "model": model_name,
                "class_id": source.get(
                    "class_id",
                    np.nan
                ),
                "class_name": source.get(
                    "class_name",
                    ""
                ),
            }

            for metric in PIXEL_METRICS:
                row[metric] = source.get(
                    metric,
                    np.nan
                )

            rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["class_name", "model"]
    )


# ============================================================
# BEST PIXEL MODEL PER CLASS
# ============================================================

def build_best_pixel_per_class(pixel_df):

    higher_is_better = [
        "Pixel_Precision",
        "Pixel_Recall",
        "Pixel_IoU",
        "Pixel_Dice",
        "Pixel_Accuracy",
    ]

    rows = []

    for class_name, group in pixel_df.groupby(
        "class_name"
    ):

        for metric in higher_is_better:

            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce"
            )

            if values.notna().sum() == 0:
                continue

            best_index = values.idxmax()

            rows.append({
                "class_name": class_name,
                "metric": metric,
                "best_model": group.loc[
                    best_index,
                    "model"
                ],
                "best_value": values.loc[
                    best_index
                ],
            })

    return pd.DataFrame(rows)


# ============================================================
# MODEL AVERAGES ACROSS CLASSES
# ============================================================

def build_class_averages(
    per_class_df,
    pixel_df
):

    instance_metrics = [
        "AP50_95",
        "AP50",
        "AP75",
        "AR50_95",
        "Precision@matchIoU",
        "Recall@matchIoU",
        "F1@matchIoU",
        "Mean_matched_mask_IoU",
        "Mean_matched_mask_Dice",
    ]

    pixel_metrics = [
        "Pixel_Precision",
        "Pixel_Recall",
        "Pixel_IoU",
        "Pixel_Dice",
        "Pixel_Accuracy",
    ]

    instance_available = [
        c
        for c in instance_metrics
        if c in per_class_df.columns
    ]

    pixel_available = [
        c
        for c in pixel_metrics
        if c in pixel_df.columns
    ]

    instance_avg = (
        per_class_df
        .groupby("model")[
            instance_available
        ]
        .mean()
        .reset_index()
    )

    pixel_avg = (
        pixel_df
        .groupby("model")[
            pixel_available
        ]
        .mean()
        .reset_index()
    )

    result = instance_avg.merge(
        pixel_avg,
        on="model",
        how="outer"
    )

    return result


# ============================================================
# VALIDATE EVALUATION SETTINGS
# ============================================================

def build_validation_table(overall_data):

    rows = []

    for model_name, df in overall_data.items():

        source = df.iloc[0]

        rows.append({
            "model": model_name,
            "images": source.get("images", np.nan),
            "GT_instances": source.get("GT_instances", np.nan),
            "prediction_conf": source.get("prediction_conf", np.nan),
            "visual_conf": source.get("visual_conf", np.nan),
            "mask_threshold": source.get("mask_threshold", np.nan),
            "match_mask_iou": source.get("match_mask_iou", np.nan),
            "max_detections": source.get("max_detections", np.nan),
            "metric_method": source.get("metric_method", ""),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    overall_data, per_class_data, pixel_data = (
        load_all_results()
    )

    overall_df = build_overall_comparison(
        overall_data
    )

    rankings_df = build_overall_rankings(
        overall_df
    )

    best_overall_df = build_best_overall(
        overall_df
    )

    per_class_df = build_per_class_comparison(
        per_class_data
    )

    best_per_class_df = build_best_per_class(
        per_class_df
    )

    pixel_df = build_pixel_comparison(
        pixel_data
    )

    best_pixel_df = build_best_pixel_per_class(
        pixel_df
    )

    class_averages_df = build_class_averages(
        per_class_df,
        pixel_df
    )

    validation_df = build_validation_table(
        overall_data
    )

    overall_df.to_csv(
        OUTPUT_DIR / "01_overall_comparison.csv",
        index=False
    )

    rankings_df.to_csv(
        OUTPUT_DIR / "02_overall_rankings.csv",
        index=False
    )

    best_overall_df.to_csv(
        OUTPUT_DIR / "03_best_model_per_overall_metric.csv",
        index=False
    )

    class_averages_df.to_csv(
        OUTPUT_DIR / "04_model_class_averages.csv",
        index=False
    )

    per_class_df.to_csv(
        OUTPUT_DIR / "05_per_class_comparison.csv",
        index=False
    )

    best_per_class_df.to_csv(
        OUTPUT_DIR / "06_best_model_per_class_metric.csv",
        index=False
    )

    pixel_df.to_csv(
        OUTPUT_DIR / "07_pixel_per_class_comparison.csv",
        index=False
    )

    best_pixel_df.to_csv(
        OUTPUT_DIR / "08_best_pixel_model_per_class.csv",
        index=False
    )

    validation_df.to_csv(
        OUTPUT_DIR / "09_evaluation_settings_check.csv",
        index=False
    )


if __name__ == "__main__":
    main()