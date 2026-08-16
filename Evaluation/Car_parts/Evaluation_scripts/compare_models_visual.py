# ============================================================
# MASK2FORMER + CHATGPT VISUAL COMPARISON
#
# Original | GT | Noa | Noa Errors | Avraham |
# Avraham Errors | Ofek | Ofek Errors | ChatGPT | ChatGPT Errors
#
# GT: YOLO segmentation TXT
# M2F: loaded only from saved local model directories
# ChatGPT: polygons loaded from JSONL
# ============================================================

from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from matplotlib.patches import Patch

from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation
)


# ============================================================
# CONFIGURATION
# ============================================================

# ---- Test set ----
TEST_IMAGES_DIR = Path(
    r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Test_set\images"
)

TEST_LABELS_DIR = Path(
    r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Test_set\labels"
)

# ---- Mask2Former model directories ----
NOA_MODEL_DIR = Path(
    r"C:\Users\cs513\Desktop\CarCal\Engine\artifacts\car_parts\car_parts_M2F_model_NOA"
)

AVRAHAM_MODEL_DIR = Path(
    r"C:\Users\cs513\Desktop\CarCal\Engine\artifacts\car_parts\car_parts_M2F_model_AVRAHAM"
)

OFEK_MODEL_DIR = Path(
    r"C:\Users\cs513\Desktop\CarCal\Engine\artifacts\car_parts\car_parts_M2F_model_OFEK"
)

# ---- ChatGPT predictions ----
CHATGPT_RESULTS_PATH = Path(
    r"C:\Users\cs513\Desktop\Evaluation\Car_parts\chatGPT_results_on_test_set_M2F.jsonl"
)

# ---- Output directory ----
OUTPUT_DIR = Path(
    r"C:\Users\cs513\Desktop\Evaluation\Car_parts\Evaluation_results\COMPARE_MODELS_VISUAL"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# IMAGE SELECTION
# ============================================================

NUM_IMAGES = 20
SELECTION_MODE = "random"   # "random" or "first"
RANDOM_SEED = 42
import random

def list_test_images():

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return sorted([
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in valid_extensions
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

# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

M2F_VIS_THRESHOLD = 0.25
CHATGPT_CONF_THRESHOLD = 0.25

MASK_ALPHA = 0.40
SAVE_DPI = 220

FIGSIZE = (26, 11)


# ============================================================
# CLASS NAMES
# IMPORTANT: must match the class order used during training
# ============================================================

CLASS_NAMES = {
    0: "back_bumper",
    1: "back_door",
    2: "back_glass",
    3: "back_left_door",
    4: "back_left_light",
    5: "back_light",
    6: "back_right_door",
    7: "back_right_light",
    8: "front_bumper",
    9: "front_door",
    10: "front_glass",
    11: "front_left_door",
    12: "front_left_light",
    13: "front_light",
    14: "front_right_door",
    15: "front_right_light",
    16: "hood",
    17: "left_mirror",
    18: "object",
    19: "right_mirror",
    20: "tailgate",
    21: "trunk",
    22: "wheel",
}

NAME_TO_CLASS_ID = {
    name: idx
    for idx, name in CLASS_NAMES.items()
}


# ============================================================
# COLORS
# RGB
# ============================================================

CLASS_COLORS = {
    0:  (255, 140, 0),
    1:  (70, 130, 180),
    2:  (154, 205, 50),
    3:  (218, 112, 214),
    4:  (255, 215, 0),
    5:  (100, 149, 237),
    6:  (205, 92, 92),
    7:  (0, 191, 255),
    8:  (255, 99, 71),
    9:  (60, 179, 113),
    10: (176, 224, 230),
    11: (186, 85, 211),
    12: (255, 165, 0),
    13: (135, 206, 250),
    14: (240, 128, 128),
    15: (0, 250, 154),
    16: (255, 255, 255),
    17: (255, 20, 147),
    18: (128, 128, 128),
    19: (0, 255, 255),
    20: (147, 112, 219),
    21: (210, 180, 140),
    22: (30, 144, 255),
}

# Pixel-error colors
CORRECT_COLOR = np.array([0, 180, 0], dtype=np.uint8)       # Green
MISSED_COLOR = np.array([255, 0, 0], dtype=np.uint8)        # Red
FALSE_COLOR = np.array([255, 140, 0], dtype=np.uint8)       # Orange
WRONG_CLASS_COLOR = np.array([160, 32, 240], dtype=np.uint8)  # Purple


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    print("=" * 70)
    print("VALIDATING PATHS")
    print("=" * 70)

    paths = {
        "Test images": TEST_IMAGES_DIR,
        "Test labels": TEST_LABELS_DIR,
        "Noa model": NOA_MODEL_DIR,
        "Avraham model": AVRAHAM_MODEL_DIR,
        "Ofek model": OFEK_MODEL_DIR,
        "ChatGPT results": CHATGPT_RESULTS_PATH,
    }

    for name, path in paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

        print(f"OK: {name}")

    print("=" * 70)


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_rgb(path):

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Could not read image:\n{path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# GT — YOLO SEGMENTATION TXT
# ============================================================

def load_yolo_gt_instances(
    label_path,
    image_shape
):

    h, w = image_shape[:2]

    instances = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"GT label file not found:\n{label_path}"
        )

    with label_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    for line in lines:

        parts = line.split()

        if len(parts) < 7:
            continue

        class_id = int(
            float(parts[0])
        )

        coords = np.array(
            list(
                map(
                    float,
                    parts[1:]
                )
            ),
            dtype=np.float32
        )

        if (
            len(coords) < 6
            or len(coords) % 2 != 0
        ):
            continue

        coords = coords.reshape(
            -1,
            2
        )

        # YOLO normalized coordinates
        coords[:, 0] *= w
        coords[:, 1] *= h

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

        if not mask.any():
            continue

        instances.append({
            "class_id": class_id,
            "mask": mask.astype(bool),
        })

    return instances


# ============================================================
# MASK2FORMER LOADING
# ============================================================

def load_mask2former_model(
    model_dir
):

    required_files = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ]

    for filename in required_files:

        file_path = (
            model_dir
            / filename
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required model file missing:\n"
                f"{file_path}"
            )

    print(
        f"Loading processor:\n"
        f"{model_dir}"
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            str(model_dir),
            local_files_only=True
        )
    )

    print(
        f"Loading model:\n"
        f"{model_dir}"
    )

    model = (
        Mask2FormerForUniversalSegmentation
        .from_pretrained(
            str(model_dir),
            local_files_only=True
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    return (
        processor,
        model
    )


# ============================================================
# MASK2FORMER PREDICTION
# ============================================================

@torch.no_grad()
def run_mask2former(
    processor,
    model,
    image_rgb
):

    h, w = (
        image_rgb.shape[:2]
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    inputs = processor(
        images=pil_image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v
        in inputs.items()
    }

    outputs = model(
        **inputs
    )

    result = (
        processor
        .post_process_instance_segmentation(
            outputs,
            threshold=M2F_VIS_THRESHOLD,
            target_sizes=[(h, w)]
        )[0]
    )

    instances = []

    if result is None:
        return instances

    segmentation = result.get(
        "segmentation"
    )

    segments_info = result.get(
        "segments_info",
        []
    )

    if segmentation is None:
        return instances

    if torch.is_tensor(
        segmentation
    ):
        segmentation = (
            segmentation
            .detach()
            .cpu()
            .numpy()
        )

    for segment in segments_info:

        segment_id = int(
            segment["id"]
        )

        class_id = int(
            segment["label_id"]
        )

        confidence = float(
            segment.get(
                "score",
                1.0
            )
        )

        mask = (
            segmentation
            == segment_id
        )

        if not mask.any():
            continue

        instances.append({
            "class_id": class_id,
            "confidence": confidence,
            "mask": mask.astype(bool),
        })

    return instances


# ============================================================
# CHATGPT
# ============================================================

def normalize_name(name):

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def load_chatgpt_results(
    path
):

    results = {}

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            filename = (
                record.get(
                    "fileName"
                )
            )

            if (
                filename is None
                and isinstance(
                    record.get(
                        "response"
                    ),
                    dict
                )
            ):
                filename = (
                    record["response"]
                    .get(
                        "FileName"
                    )
                )

            if filename:
                results[
                    filename
                ] = record

    return results


def chatgpt_instances_from_record(
    record,
    image_shape
):

    instances = []

    if record is None:
        return instances

    h, w = image_shape[:2]

    response = record.get(
        "response",
        {}
    )

    detections = response.get(
        "detections",
        []
    )

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

        class_id = (
            NAME_TO_CLASS_ID[
                label
            ]
        )

        polygon_obj = det.get(
            "polygon"
        )

        points = None

        if isinstance(
            polygon_obj,
            dict
        ):
            points = (
                polygon_obj
                .get(
                    "points"
                )
            )

        elif isinstance(
            polygon_obj,
            list
        ):
            points = polygon_obj

        if (
            not points
            or len(points) < 3
        ):
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

        # Handle normalized coordinates if needed
        if (
            np.max(coords[:, 0]) <= 1.0
            and np.max(coords[:, 1]) <= 1.0
        ):

            coords[:, 0] *= w
            coords[:, 1] *= h

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

        if not mask.any():
            continue

        instances.append({
            "class_id": class_id,
            "confidence": confidence,
            "mask": mask.astype(bool),
        })

    return instances


# ============================================================
# DRAW SEGMENTATION
# ============================================================

def draw_instances(
    image_rgb,
    instances,
    show_confidence=False
):

    result = image_rgb.copy()

    color_overlay = (
        image_rgb.copy()
    )

    # Fill masks
    for inst in instances:

        class_id = (
            inst["class_id"]
        )

        mask = (
            inst["mask"]
            .astype(bool)
        )

        color = np.array(
            CLASS_COLORS.get(
                class_id,
                (255, 255, 255)
            ),
            dtype=np.uint8
        )

        color_overlay[
            mask
        ] = color

    result = (
        (
            1 - MASK_ALPHA
        )
        * result
        +
        MASK_ALPHA
        * color_overlay
    ).astype(
        np.uint8
    )

    # Contours + labels
    result_bgr = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    for inst in instances:

        class_id = (
            inst["class_id"]
        )

        mask = (
            inst["mask"]
            .astype(np.uint8)
        )

        color_rgb = (
            CLASS_COLORS.get(
                class_id,
                (255, 255, 255)
            )
        )

        color_bgr = (
            color_rgb[2],
            color_rgb[1],
            color_rgb[0]
        )

        contours, _ = (
            cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        )

        if not contours:
            continue

        cv2.drawContours(
            result_bgr,
            contours,
            -1,
            color_bgr,
            2
        )

        largest = max(
            contours,
            key=cv2.contourArea
        )

        x, y, _, _ = (
            cv2.boundingRect(
                largest
            )
        )

        label = (
            CLASS_NAMES.get(
                class_id,
                str(class_id)
            )
        )

        if (
            show_confidence
            and "confidence" in inst
        ):
            label += (
                f" "
                f"{inst['confidence']:.2f}"
            )

        cv2.putText(
            result_bgr,
            label,
            (
                x,
                max(
                    15,
                    y - 4
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_bgr,
            1,
            cv2.LINE_AA
        )

    return cv2.cvtColor(
        result_bgr,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# SEMANTIC MAP FOR CLASS-AWARE PIXEL ERRORS
# ============================================================

def instances_to_semantic_map(
    instances,
    image_shape
):

    h, w = image_shape[:2]

    semantic = np.full(
        (h, w),
        -1,
        dtype=np.int32
    )

    # Lower-confidence masks first,
    # higher-confidence predictions overwrite them.
    sorted_instances = sorted(
        instances,
        key=lambda x: x.get(
            "confidence",
            1.0
        )
    )

    for inst in sorted_instances:

        semantic[
            inst["mask"]
        ] = (
            inst["class_id"]
        )

    return semantic


# ============================================================
# PIXEL ERROR IMAGE
# ============================================================

def create_pixel_error_image(
    image_rgb,
    gt_instances,
    pred_instances
):

    gt_map = (
        instances_to_semantic_map(
            gt_instances,
            image_rgb.shape
        )
    )

    pred_map = (
        instances_to_semantic_map(
            pred_instances,
            image_rgb.shape
        )
    )

    gt_fg = (
        gt_map != -1
    )

    pred_fg = (
        pred_map != -1
    )

    correct = (
        gt_fg
        & pred_fg
        & (
            gt_map
            == pred_map
        )
    )

    missed = (
        gt_fg
        & ~pred_fg
    )

    false = (
        ~gt_fg
        & pred_fg
    )

    wrong_class = (
        gt_fg
        & pred_fg
        & (
            gt_map
            != pred_map
        )
    )

    # Darken original image
    error_img = (
        image_rgb.astype(
            np.float32
        )
        * 0.30
    ).astype(
        np.uint8
    )

    error_img[
        correct
    ] = CORRECT_COLOR

    error_img[
        missed
    ] = MISSED_COLOR

    error_img[
        false
    ] = FALSE_COLOR

    error_img[
        wrong_class
    ] = WRONG_CLASS_COLOR

    return error_img


# ============================================================
# SAVE FIGURE
# ============================================================

def save_comparison(
    image_path,
    original,
    gt_instances,
    noa_instances,
    avraham_instances,
    ofek_instances,
    chatgpt_instances
):

    gt_visual = draw_instances(
        original,
        gt_instances
    )

    noa_visual = draw_instances(
        original,
        noa_instances,
        show_confidence=True
    )

    avraham_visual = draw_instances(
        original,
        avraham_instances,
        show_confidence=True
    )

    ofek_visual = draw_instances(
        original,
        ofek_instances,
        show_confidence=True
    )

    chatgpt_visual = draw_instances(
        original,
        chatgpt_instances,
        show_confidence=True
    )

    noa_error = (
        create_pixel_error_image(
            original,
            gt_instances,
            noa_instances
        )
    )

    avraham_error = (
        create_pixel_error_image(
            original,
            gt_instances,
            avraham_instances
        )
    )

    ofek_error = (
        create_pixel_error_image(
            original,
            gt_instances,
            ofek_instances
        )
    )

    chatgpt_error = (
        create_pixel_error_image(
            original,
            gt_instances,
            chatgpt_instances
        )
    )

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
            "Avraham Prediction",
            avraham_visual
        ),

        (
            "Avraham Pixel Errors",
            avraham_error
        ),
        (
            "Ofek Prediction",
            ofek_visual
        ),
        (
            "Ofek Pixel Errors",
            ofek_error
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

    for ax, (
        title,
        image
    ) in zip(
        axes.flatten(),
        panels
    ):

        ax.imshow(
            image
        )

        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold"
        )

        ax.axis(
            "off"
        )

    legend_elements = [
        Patch(
            facecolor=
            CORRECT_COLOR / 255,
            label="Correct class"
        ),

        Patch(
            facecolor=
            MISSED_COLOR / 255,
            label="Missed GT"
        ),

        Patch(
            facecolor=
            FALSE_COLOR / 255,
            label="False Positive"
        ),

        Patch(
            facecolor=
            WRONG_CLASS_COLOR / 255,
            label="Wrong Class"
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.965
        ),
        ncol=4,
        title="Pixel Error Legend",
        frameon=True
    )

    fig.suptitle(
        f"Mask2Former and ChatGPT Comparison — {image_path.name}",
        fontsize=16,
        fontweight="bold",
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
        / (
            f"{image_path.stem}"
            f"_comparison.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    validate_paths()

    print(
        "\nDevice:",
        DEVICE
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print(
        "\nLoading Noa model..."
    )

    (
        noa_processor,
        noa_model
    ) = load_mask2former_model(
        NOA_MODEL_DIR
    )

    print(
        "\nLoading Avraham model..."
    )

    (
        avraham_processor,
        avraham_model
    ) = load_mask2former_model(
        AVRAHAM_MODEL_DIR
    )

    print(
        "\nLoading Ofek model..."
    )

    (
        ofek_processor,
        ofek_model
    ) = load_mask2former_model(
        OFEK_MODEL_DIR
    )

    # --------------------------------------------------------
    # ChatGPT
    # --------------------------------------------------------

    print(
        "\nLoading ChatGPT results..."
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
        # --------------------------------------------------------
    # Select test images
    # --------------------------------------------------------

    all_images = list_test_images()

    selected_images = choose_images(
        all_images
    )

    print(
        "\nTotal test images:",
        len(all_images)
    )

    print(
        "Selected images:",
        len(selected_images)
    )

    # --------------------------------------------------------
    # Run comparison
    # --------------------------------------------------------

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        print(
            f"\n[{index}/{len(selected_images)}] "
            f"{image_path.name}"
        )
   

      

        if not image_path.exists():

            print(
                "ERROR: image not found:",
                image_path
            )

            continue

        try:

            original = (
                load_rgb(
                    image_path
                )
            )

            # -----------------------------------------------
            # GT
            # -----------------------------------------------

            label_path = (
                TEST_LABELS_DIR
                / (
                    image_path.stem
                    + ".txt"
                )
            )

            gt_instances = (
                load_yolo_gt_instances(
                    label_path,
                    original.shape
                )
            )

            # -----------------------------------------------
            # NOA
            # -----------------------------------------------

            noa_instances = (
                run_mask2former(
                    noa_processor,
                    noa_model,
                    original
                )
            )

            # -----------------------------------------------
            # AVRAHAM
            # -----------------------------------------------

            avraham_instances = (
                run_mask2former(
                    avraham_processor,
                    avraham_model,
                    original
                )
            )

            # -----------------------------------------------
            # OFEK
            # -----------------------------------------------

            ofek_instances = (
                run_mask2former(
                    ofek_processor,
                    ofek_model,
                    original
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

            if (
                chatgpt_record
                is None
            ):
                print(
                    "WARNING: "
                    "No ChatGPT result "
                    "found for image."
                )

            chatgpt_instances = (
                chatgpt_instances_from_record(
                    chatgpt_record,
                    original.shape
                )
            )

            # -----------------------------------------------
            # SAVE
            # -----------------------------------------------

            output_path = (
                save_comparison(
                    image_path,
                    original,
                    gt_instances,
                    noa_instances,
                    avraham_instances,
                    ofek_instances,
                    chatgpt_instances
                )
            )

            print(
                "Saved:",
                output_path
            )

        except Exception as exc:

            print(
                "ERROR:",
                exc
            )

    print(
        "\nDone."
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()