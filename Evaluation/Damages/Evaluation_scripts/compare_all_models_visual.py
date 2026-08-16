# ============================================================
# VISUAL COMPARISON:
# Original | GT | Noa | Noa Errors | Tal | Tal Errors |
# Harel | Harel Errors | ChatGPT | ChatGPT Errors
#
# NO BOUNDING BOXES.
# GT is loaded from COCO instances_test2017.json.
# ============================================================

from pathlib import Path
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


# ============================================================
# CONFIGURATION — EDIT THESE PATHS
# ============================================================

TEST_IMAGES_DIR = Path(r"C:\Users\cs513\Desktop\Evaluation\Damages\test_set")
COCO_ANNOTATIONS_JSON = Path(r"C:\Users\cs513\Desktop\Evaluation\Damages\test_set\instances_test2017.json")

# ---- Model paths ----
NOA_MODEL_PATH   = Path(r"C:\Users\cs513\Desktop\CarCal\Engine\artifacts\damages\damage_YOLO_model_NOA.pt")
TAL_MODEL_PATH   = Path(r"C:\Users\cs513\Desktop\CarCal\Engine\artifacts\damages\damage_YOLO_model_TAL.pt")
HAREL_MODEL_PATH = Path(r"C:\Users\cs513\Desktop\CarCal\Engine\artifacts\damages\damage_YOLO_model_HAREL.pt")

# ---- Output directory ----
OUTPUT_DIR = Path(r"C:\Users\cs513\Desktop\Evaluation\Damages\Evaluation_results\VISUAL_COMPARISON")


CHATGPT_RESULTS_PATH = Path(r"C:\Users\cs513\Desktop\Evaluation\Damages\chatGPT_results_on_test_set.jsonl")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 50

SELECTION_MODE = "random"
RANDOM_SEED = 42

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_IOU = 0.70
MAX_DET = 50

DEVICE = 0 if torch.cuda.is_available() else "cpu"

SAVE_DPI = 220

# 2 rows × 5 columns
FIGSIZE = (24, 11)

# ChatGPT confidence filtering
CHATGPT_CONF_THRESHOLD = 0.25


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass shatter",
    4: "lamp broken",
    5: "tire flat",
}

NAME_TO_CLASS_ID = {
    name: idx
    for idx, name in CLASS_NAMES.items()
}


# ============================================================
# COLORS
# ============================================================

# BGR colors for OpenCV
CLASS_COLORS = {
    0: (0, 180, 255),
    1: (255, 0, 255),
    2: (0, 255, 255),
    3: (0, 200, 0),
    4: (255, 165, 0),
    5: (255, 0, 0),
}

# Pixel-error colors
CORRECT_COLOR = (0, 180, 0)   # Green
MISSED_COLOR  = (0, 0, 255)   # Red
FALSE_COLOR   = (255, 140, 0) # Blue-ish / orange

MASK_ALPHA = 0.35


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def list_test_images():
    extensions = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".webp"
    }

    return sorted([
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in extensions
    ])


def choose_images(image_paths):
    if NUM_IMAGES >= len(image_paths):
        return image_paths

    if SELECTION_MODE == "first":
        return image_paths[:NUM_IMAGES]

    random.seed(RANDOM_SEED)
    return random.sample(
        image_paths,
        NUM_IMAGES
    )


def load_rgb(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# COCO GROUND TRUTH
# ============================================================

def coco_segmentation_to_mask(
    segmentation,
    height,
    width
):
    if isinstance(segmentation, list):

        rles = mask_utils.frPyObjects(
            segmentation,
            height,
            width
        )

        rle = mask_utils.merge(rles)

    elif (
        isinstance(segmentation, dict)
        and isinstance(
            segmentation.get("counts"),
            list
        )
    ):

        rle = mask_utils.frPyObjects(
            segmentation,
            height,
            width
        )

    else:
        rle = segmentation

    decoded = mask_utils.decode(rle)

    if decoded.ndim == 3:
        decoded = np.any(
            decoded,
            axis=2
        )

    return decoded.astype(bool)


def build_coco_indexes(coco):
    """
    Build:
        filename -> image COCO metadata
        COCO category ID -> YOLO-style 0..5 index
    """

    image_by_filename = {}

    for info in coco.loadImgs(
        coco.getImgIds()
    ):
        image_by_filename[
            info["file_name"]
        ] = info

    categories = sorted(
        coco.loadCats(
            coco.getCatIds()
        ),
        key=lambda x: x["id"]
    )

    coco_cat_to_class_idx = {}

    for cat in categories:

        name = normalize_name(
            cat["name"]
        )

        if name not in NAME_TO_CLASS_ID:
            raise ValueError(
                f"Unknown COCO class: {name}"
            )

        coco_cat_to_class_idx[
            int(cat["id"])
        ] = NAME_TO_CLASS_ID[name]

    return (
        image_by_filename,
        coco_cat_to_class_idx
    )


def load_gt_instances(
    coco,
    image_info,
    cat_mapping
):
    ann_ids = coco.getAnnIds(
        imgIds=[image_info["id"]]
    )

    annotations = coco.loadAnns(
        ann_ids
    )

    instances = []

    h = int(
        image_info["height"]
    )

    w = int(
        image_info["width"]
    )

    for ann in annotations:

        category_id = int(
            ann["category_id"]
        )

        if category_id not in cat_mapping:
            continue

        mask = coco_segmentation_to_mask(
            ann["segmentation"],
            h,
            w
        )

        if not mask.any():
            continue

        instances.append({
            "class_id":
                cat_mapping[category_id],

            "mask":
                mask
        })

    return instances


# ============================================================
# SEGMENTATION-ONLY VISUALIZATION
# ============================================================

def draw_instances_segmentation_only(
    image_rgb,
    instances,
    show_confidence=False
):
    """
    Draws:
        segmentation fill
        segmentation contour
        class label

    NO bounding boxes.
    """

    image_bgr = cv2.cvtColor(
        image_rgb.copy(),
        cv2.COLOR_RGB2BGR
    )

    overlay = image_bgr.copy()

    # First: fill masks
    for inst in instances:

        cls_id = inst["class_id"]

        mask = inst["mask"].astype(bool)

        color = CLASS_COLORS.get(
            cls_id,
            (255, 255, 255)
        )

        overlay[mask] = color

    image_bgr = cv2.addWeighted(
        overlay,
        MASK_ALPHA,
        image_bgr,
        1 - MASK_ALPHA,
        0
    )

    # Second: contours + labels
    for inst in instances:

        cls_id = inst["class_id"]

        mask = (
            inst["mask"]
            .astype(np.uint8)
        )

        color = CLASS_COLORS.get(
            cls_id,
            (255, 255, 255)
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cv2.drawContours(
            image_bgr,
            contours,
            -1,
            color,
            2
        )

        # Position label near largest contour
        largest = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        label = CLASS_NAMES.get(
            cls_id,
            str(cls_id)
        )

        if (
            show_confidence
            and "confidence" in inst
        ):
            label += (
                f" {inst['confidence']:.2f}"
            )

        # Background behind text only
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        text_y = max(
            th + 5,
            y - 5
        )

        cv2.rectangle(
            image_bgr,
            (
                x,
                text_y - th - 6
            ),
            (
                x + tw + 6,
                text_y + 3
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            image_bgr,
            label,
            (
                x + 3,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# GT MASK UNION
# ============================================================

def union_mask(
    instances,
    image_shape
):
    h, w = image_shape[:2]

    result = np.zeros(
        (h, w),
        dtype=bool
    )

    for inst in instances:
        result |= inst["mask"]

    return result


# ============================================================
# YOLO PREDICTIONS
# ============================================================

def run_yolo_model(
    model,
    image_path,
    original_shape
):
    result = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU,
        max_det=MAX_DET,
        retina_masks=True,
        device=DEVICE,
        verbose=False
    )[0]

    instances = []

    if (
        result.masks is None
        or result.boxes is None
    ):
        return instances

    masks = (
        result.masks.data
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    h, w = original_shape[:2]

    for (
        mask,
        cls_id,
        confidence
    ) in zip(
        masks,
        classes,
        confidences
    ):

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(
                    np.float32
                ),
                (w, h),
                interpolation=
                    cv2.INTER_NEAREST
            )

        binary = mask > 0.5

        if not binary.any():
            continue

        instances.append({
            "class_id":
                int(cls_id),

            "confidence":
                float(confidence),

            "mask":
                binary
        })

    return instances


# ============================================================
# CHATGPT RESULTS
# ============================================================

def load_chatgpt_results(path):
    """
    Returns:
        filename -> full JSON record
    """

    results = {}

    if path.suffix.lower() == ".jsonl":

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                filename = (
                    record.get("fileName")
                )

                if (
                    filename is None
                    and isinstance(
                        record.get("response"),
                        dict
                    )
                ):
                    filename = (
                        record["response"]
                        .get("FileName")
                    )

                if filename:
                    results[filename] = record

    else:

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            records = data
        else:
            records = [data]

        for record in records:

            filename = (
                record.get("fileName")
            )

            if (
                filename is None
                and isinstance(
                    record.get("response"),
                    dict
                )
            ):
                filename = (
                    record["response"]
                    .get("FileName")
                )

            if filename:
                results[filename] = record

    return results


def chatgpt_instances_from_record(
    record,
    image_shape
):
    """
    Converts your ChatGPT polygon format
    into the exact same internal representation
    used for YOLO and GT.

    Expected polygon:
      response.detections[].polygon.points
    """

    instances = []

    if record is None:
        return instances

    response = record.get(
        "response",
        {}
    )

    detections = response.get(
        "detections",
        []
    )

    h, w = image_shape[:2]

    for det in detections:

        confidence = float(
            det.get(
                "confidence",
                0.0
            )
        )

        if (
            confidence
            < CHATGPT_CONF_THRESHOLD
        ):
            continue

        label = normalize_name(
            det.get(
                "label",
                ""
            )
        )

        if (
            label
            not in NAME_TO_CLASS_ID
        ):
            continue

        cls_id = (
            NAME_TO_CLASS_ID[label]
        )

        polygon_obj = det.get(
            "polygon",
            {}
        )

        if not isinstance(
            polygon_obj,
            dict
        ):
            continue

        points = polygon_obj.get(
            "points",
            []
        )

        if len(points) < 3:
            continue

        coords = np.array(
            [
                [
                    float(p["x"]),
                    float(p["y"])
                ]
                for p in points
            ],
            dtype=np.float32
        )

        # In your converted result,
        # coordinates are already pixel coordinates.
        coords[:, 0] = np.clip(
            coords[:, 0],
            0,
            w - 1
        )

        coords[:, 1] = np.clip(
            coords[:, 1],
            0,
            h - 1
        )

        polygon = coords.astype(
            np.int32
        )

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [polygon],
            1
        )

        instances.append({
            "class_id":
                cls_id,

            "confidence":
                confidence,

            "mask":
                mask.astype(bool)
        })

    return instances


# ============================================================
# PIXEL ERROR VISUALIZATION
# ============================================================

def create_pixel_error_image(
    image_rgb,
    gt_mask,
    pred_mask
):
    """
    Correct = prediction overlaps GT
    Missed  = GT pixels not predicted
    False   = predicted pixels outside GT
    """

    h, w = image_rgb.shape[:2]

    correct = (
        gt_mask & pred_mask
    )

    missed = (
        gt_mask & ~pred_mask
    )

    false = (
        ~gt_mask & pred_mask
    )

    # Dark image so errors stand out
    image_bgr = cv2.cvtColor(
        image_rgb.copy(),
        cv2.COLOR_RGB2BGR
    )

    dark = (
        image_bgr.astype(
            np.float32
        ) * 0.35
    ).astype(np.uint8)

    dark[correct] = (
        CORRECT_COLOR
    )

    dark[missed] = (
        MISSED_COLOR
    )

    dark[false] = (
        FALSE_COLOR
    )

    return cv2.cvtColor(
        dark,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# FIGURE
# ============================================================

def save_comparison(
    image_path,
    original,
    gt_visual,
    noa_visual,
    noa_error,
    tal_visual,
    tal_error,
    harel_visual,
    harel_error,
    chatgpt_visual,
    chatgpt_error
):

    fig, axes = plt.subplots(
        2,
        5,
        figsize=FIGSIZE
    )

    panels = [
        (
            "Original",
            original
        ),
        (
            "Ground Truth",
            gt_visual
        ),
        (
            "Noa Prediction",
            noa_visual
        ),
        (
            "Noa Pixel Errors",
            noa_error
        ),
        (
            "Tal Prediction",
            tal_visual
        ),

        (
            "Tal Pixel Errors",
            tal_error
        ),
        (
            "Harel Prediction",
            harel_visual
        ),
        (
            "Harel Pixel Errors",
            harel_error
        ),
        (
            "ChatGPT Prediction",
            chatgpt_visual
        ),
        (
            "ChatGPT Pixel Errors",
            chatgpt_error
        ),
    ]

    for ax, (title, image) in zip(
        axes.flatten(),
        panels
    ):
        ax.imshow(image)
        ax.set_title(
            title,
            fontsize=12
        )
        ax.axis("off")

    # --------------------------------------------------------
    # Global Pixel Error legend
    # --------------------------------------------------------

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=np.array(
                CORRECT_COLOR[::-1]
            ) / 255,
            label="Correct"
        ),

        Patch(
            facecolor=np.array(
                MISSED_COLOR[::-1]
            ) / 255,
            label="Missed"
        ),

        Patch(
            facecolor=np.array(
                FALSE_COLOR[::-1]
            ) / 255,
            label="False"
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.96
        ),
        ncol=3,
        frameon=True,
        title="Pixel Error Legend"
    )

    fig.suptitle(
        f"Model Comparison — {image_path.name}",
        fontsize=16,
        y=1.01
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.91
        ]
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Using device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Load COCO GT
    # --------------------------------------------------------

    print(
        "Loading COCO annotations..."
    )

    coco = COCO(
        str(
            COCO_ANNOTATIONS_JSON
        )
    )

    (
        image_by_filename,
        coco_cat_mapping
    ) = build_coco_indexes(
        coco
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print(
        "Loading models..."
    )

    noa_model = YOLO(
        str(NOA_MODEL_PATH)
    )

    tal_model = YOLO(
        str(TAL_MODEL_PATH)
    )

    harel_model = YOLO(
        str(HAREL_MODEL_PATH)
    )

    # --------------------------------------------------------
    # ChatGPT
    # --------------------------------------------------------

    print(
        "Loading ChatGPT results..."
    )

    chatgpt_results = (
        load_chatgpt_results(
            CHATGPT_RESULTS_PATH
        )
    )

    print(
        "ChatGPT records:",
        len(chatgpt_results)
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    all_images = (
        list_test_images()
    )

    selected_images = (
        choose_images(
            all_images
        )
    )

    print(
        "Total test images:",
        len(all_images)
    )

    print(
        "Selected images:",
        len(selected_images)
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    successful = 0

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        print(
            f"[{index}/{len(selected_images)}] "
            f"{image_path.name}"
        )

        try:

            original = load_rgb(
                image_path
            )

            # -----------------------------------------------
            # COCO GT
            # -----------------------------------------------

            if (
                image_path.name
                not in image_by_filename
            ):
                raise ValueError(
                    "Image does not exist "
                    "in COCO annotations: "
                    f"{image_path.name}"
                )

            image_info = (
                image_by_filename[
                    image_path.name
                ]
            )

            gt_instances = (
                load_gt_instances(
                    coco,
                    image_info,
                    coco_cat_mapping
                )
            )

            gt_visual = (
                draw_instances_segmentation_only(
                    original,
                    gt_instances,
                    show_confidence=False
                )
            )

            gt_mask = union_mask(
                gt_instances,
                original.shape
            )

            # -----------------------------------------------
            # NOA
            # -----------------------------------------------

            noa_instances = (
                run_yolo_model(
                    noa_model,
                    image_path,
                    original.shape
                )
            )

            noa_visual = (
                draw_instances_segmentation_only(
                    original,
                    noa_instances,
                    show_confidence=True
                )
            )

            noa_mask = union_mask(
                noa_instances,
                original.shape
            )

            noa_error = (
                create_pixel_error_image(
                    original,
                    gt_mask,
                    noa_mask
                )
            )

            # -----------------------------------------------
            # TAL
            # -----------------------------------------------

            tal_instances = (
                run_yolo_model(
                    tal_model,
                    image_path,
                    original.shape
                )
            )

            tal_visual = (
                draw_instances_segmentation_only(
                    original,
                    tal_instances,
                    show_confidence=True
                )
            )

            tal_mask = union_mask(
                tal_instances,
                original.shape
            )

            tal_error = (
                create_pixel_error_image(
                    original,
                    gt_mask,
                    tal_mask
                )
            )

            # -----------------------------------------------
            # HAREL
            # -----------------------------------------------

            harel_instances = (
                run_yolo_model(
                    harel_model,
                    image_path,
                    original.shape
                )
            )

            harel_visual = (
                draw_instances_segmentation_only(
                    original,
                    harel_instances,
                    show_confidence=True
                )
            )

            harel_mask = union_mask(
                harel_instances,
                original.shape
            )

            harel_error = (
                create_pixel_error_image(
                    original,
                    gt_mask,
                    harel_mask
                )
            )

            # -----------------------------------------------
            # CHATGPT
            # -----------------------------------------------

            chatgpt_record = (
                chatgpt_results.get(
                    image_path.name
                )
            )

            if chatgpt_record is None:
                print(
                    "  Warning: no ChatGPT "
                    "result found."
                )

            chatgpt_instances = (
                chatgpt_instances_from_record(
                    chatgpt_record,
                    original.shape
                )
            )

            chatgpt_visual = (
                draw_instances_segmentation_only(
                    original,
                    chatgpt_instances,
                    show_confidence=True
                )
            )

            chatgpt_mask = union_mask(
                chatgpt_instances,
                original.shape
            )

            chatgpt_error = (
                create_pixel_error_image(
                    original,
                    gt_mask,
                    chatgpt_mask
                )
            )

            # -----------------------------------------------
            # Save
            # -----------------------------------------------

            output_path = save_comparison(
                image_path,
                original,
                gt_visual,

                noa_visual,
                noa_error,

                tal_visual,
                tal_error,

                harel_visual,
                harel_error,

                chatgpt_visual,
                chatgpt_error
            )

            print(
                "  Saved:",
                output_path
            )

            successful += 1

        except Exception as exc:

            print(
                "  ERROR:",
                exc
            )

    print()
    print(
        "Done."
    )

    print(
        f"Created {successful} "
        f"comparison figures."
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()