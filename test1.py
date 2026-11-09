# ============================================================
# DRFP-Net V4 EVALUATION
#
# Uses:
#   E:\DRPF\Val\Val
#
# because this dataset contains ground-truth masks.
#
# Produces:
#   - mIoU
#   - IoU
#   - Precision
#   - Recall
#   - F1
#   - Pixel Accuracy
#   - Boundary Precision
#   - Boundary Recall
#   - Boundary F1
#   - Confusion Matrix
#   - CSV metrics
#   - Prediction images
#
# Prediction visualization:
#
#   ORIGINAL | GROUND TRUTH | PREDICTION | BOUNDARY
# ============================================================

import os
import csv

import numpy as np

import torch
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw

from dataset import LoveDADataset
from model import DRFPNet
from metrics import (
    update_confusion_matrix,
    compute_metrics,
    boundary_metrics,
    mask_to_boundary
)

import config


# ============================================================
# SETTINGS
# ============================================================

EVALUATION_DIR = os.path.join(
    config.ROOT_DIR,
    "evaluation_v4"
)

PREDICTION_DIR = os.path.join(
    EVALUATION_DIR,
    "predictions"
)

CSV_PATH = os.path.join(
    EVALUATION_DIR,
    "metrics.csv"
)

BEST_CHECKPOINT = getattr(
    config,
    "BEST_CHECKPOINT",
    os.path.join(
        config.CHECKPOINT_DIR,
        "E:/DRPF/checkpoints_v3/best_model.pth"
    )
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [

    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture"

]


# ============================================================
# COLOR PALETTE
# ============================================================

PALETTE = np.array(

    [
        [0, 0, 0],

        [255, 0, 0],

        [255, 255, 0],

        [0, 120, 255],

        [210, 140, 70],

        [0, 180, 0],

        [180, 100, 255]
    ],

    dtype=np.uint8
)


# ============================================================
# COLORIZE MASK
# ============================================================

def colorize_mask(
    mask
):

    mask = (
        mask.detach()
        .cpu()
        .numpy()
    )

    image = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3
        ),
        dtype=np.uint8
    )

    for class_id in range(
        config.NUM_CLASSES
    ):

        image[
            mask == class_id
        ] = PALETTE[
            class_id
        ]

    # Ignore pixels -> white
    image[
        mask == config.IGNORE_INDEX
    ] = [255, 255, 255]

    return image


# ============================================================
# BOUNDARY VISUALIZATION
# ============================================================

def boundary_visualization(
    boundary
):

    boundary = (
        boundary.detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    image = np.zeros(
        (
            boundary.shape[0],
            boundary.shape[1],
            3
        ),
        dtype=np.uint8
    )

    image[
        boundary > 0
    ] = [
        255,
        255,
        255
    ]

    return image


# ============================================================
# IMAGE TENSOR -> RGB
# ============================================================

def tensor_to_rgb(
    tensor
):

    image = (
        tensor.detach()
        .cpu()
    )

    # C,H,W -> H,W,C
    image = image.permute(
        1,
        2,
        0
    ).numpy()

    # --------------------------------------------------------
    # If dataset normalization is unknown, use robust
    # min-max conversion.
    #
    # This is only for visualization.
    # It does not affect model evaluation.
    # --------------------------------------------------------

    image = (
        image
        -
        image.min()
    ) / (
        image.max()
        -
        image.min()
        +
        1e-8
    )

    image = (
        image * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    return image


# ============================================================
# CREATE COMPARISON
# ============================================================

def create_comparison(
    original,
    ground_truth,
    prediction,
    gt_boundary,
    pred_boundary,
    sample_index
):

    original = Image.fromarray(
        original
    )

    ground_truth = Image.fromarray(
        ground_truth
    )

    prediction = Image.fromarray(
        prediction
    )

    gt_boundary = Image.fromarray(
        gt_boundary
    )

    pred_boundary = Image.fromarray(
        pred_boundary
    )

    width = original.width

    height = original.height

    label_height = 40

    total_width = (
        width * 5
    )

    total_height = (
        height
        +
        label_height
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            total_height
        ),
        "white"
    )

    panels = [

        (
            original,
            "ORIGINAL"
        ),

        (
            ground_truth,
            "GROUND TRUTH"
        ),

        (
            prediction,
            "PREDICTION"
        ),

        (
            gt_boundary,
            "GT BOUNDARY"
        ),

        (
            pred_boundary,
            "PREDICTED BOUNDARY"
        )

    ]

    draw = ImageDraw.Draw(
        canvas
    )

    for index, (
        image,
        label
    ) in enumerate(panels):

        x = (
            index * width
        )

        canvas.paste(
            image,
            (
                x,
                label_height
            )
        )

        draw.text(
            (
                x + 10,
                10
            ),
            label,
            fill="black"
        )

    return canvas


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    results
):

    os.makedirs(
        EVALUATION_DIR,
        exist_ok=True
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Class",
                "IoU",
                "Precision",
                "Recall",
                "F1"
            ]
        )

        for i in range(
            config.NUM_CLASSES
        ):

            writer.writerow(
                [
                    CLASS_NAMES[i],

                    f"{results['iou'][i] * 100:.4f}",

                    f"{results['precision'][i] * 100:.4f}",

                    f"{results['recall'][i] * 100:.4f}",

                    f"{results['f1'][i] * 100:.4f}"
                ]
            )

        writer.writerow([])

        writer.writerow(
            [
                "OVERALL",
                "",
                "",
                "",
                ""
            ]
        )

        writer.writerow(
            [
                "mIoU",
                results["miou"] * 100
            ]
        )

        writer.writerow(
            [
                "Pixel Accuracy",
                results["pixel_accuracy"] * 100
            ]
        )

        writer.writerow(
            [
                "Macro Precision",
                results["macro_precision"] * 100
            ]
        )

        writer.writerow(
            [
                "Macro Recall",
                results["macro_recall"] * 100
            ]
        )

        writer.writerow(
            [
                "Macro F1",
                results["macro_f1"] * 100
            ]
        )

        writer.writerow(
            [
                "Boundary Precision",
                results["boundary_precision"] * 100
            ]
        )

        writer.writerow(
            [
                "Boundary Recall",
                results["boundary_recall"] * 100
            ]
        )

        writer.writerow(
            [
                "Boundary F1",
                results["boundary_f1"] * 100
            ]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        PREDICTION_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("DRFP-Net V4 EVALUATION")
    print("=" * 70)

    config.print_config()

    device = config.DEVICE

    print()
    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING VALIDATION DATASET")
    print("=" * 70)

    dataset = LoveDADataset(
        config.VAL_DIR,
        image_size=config.IMAGE_SIZE,
        augmentation=False
    )

    print(
        f"Validation samples: "
        f"{len(dataset)}"
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    # --------------------------------------------------------
    # WINDOWS SAFE
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=config.PIN_MEMORY
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = DRFPNet(
        num_classes=config.NUM_CLASSES,
        feature_channels=config.FEATURE_CHANNELS
    ).to(device)

    if not os.path.isfile(
        BEST_CHECKPOINT
    ):

        raise FileNotFoundError(
            f"Best checkpoint not found:\n"
            f"{BEST_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', -1) + 1}"
    )

    print(
        f"Saved best mIoU: "
        f"{checkpoint.get('best_miou', 0.0) * 100:.2f}%"
    )

    model.eval()

    # --------------------------------------------------------
    # CONFUSION
    # --------------------------------------------------------

    confusion_matrix = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.int64,
        device=device
    )

    boundary_precision_total = 0.0
    boundary_recall_total = 0.0
    boundary_f1_total = 0.0

    # --------------------------------------------------------
    # CSV PER SAMPLE
    # --------------------------------------------------------

    per_sample_csv = os.path.join(
        EVALUATION_DIR,
        "per_sample_metrics.csv"
    )

    csv_file = open(
        per_sample_csv,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow(
        [
            "sample",
            "boundary_precision",
            "boundary_recall",
            "boundary_f1"
        ]
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    with torch.no_grad():

        for index, (
            images,
            targets
        ) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(
                    config.AMP
                    and
                    device.type == "cuda"
                )
            ):

                output = model(
                    images
                )

                logits = output[
                    "logits"
                ]

                boundary_logits = output[
                    "boundary_logits"
                ]

            prediction = torch.argmax(
                logits,
                dim=1
            )

            boundary_prediction = (
                torch.sigmoid(
                    boundary_logits
                )
                >
                0.5
            ).squeeze(1)

            # ------------------------------------------------
            # Semantic metrics
            # ------------------------------------------------

            confusion_matrix = (
                update_confusion_matrix(
                    confusion_matrix,
                    prediction,
                    targets,
                    config.NUM_CLASSES,
                    config.IGNORE_INDEX
                )
            )

            # ------------------------------------------------
            # Boundary metrics
            # ------------------------------------------------

            bmetrics = boundary_metrics(
                prediction,
                targets,
                config.IGNORE_INDEX,
                tolerance=2
            )

            bp = float(
                bmetrics[
                    "boundary_precision"
                ]
            )

            br = float(
                bmetrics[
                    "boundary_recall"
                ]
            )

            bf = float(
                bmetrics[
                    "boundary_f1"
                ]
            )

            boundary_precision_total += bp

            boundary_recall_total += br

            boundary_f1_total += bf

            csv_writer.writerow(
                [
                    index,
                    bp * 100,
                    br * 100,
                    bf * 100
                ]
            )

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            original = tensor_to_rgb(
                images[0]
            )

            gt_color = colorize_mask(
                targets[0]
            )

            pred_color = colorize_mask(
                prediction[0]
            )

            gt_boundary = (
                mask_to_boundary(
                    targets,
                    config.IGNORE_INDEX
                )[0]
            )

            gt_boundary_image = (
                boundary_visualization(
                    gt_boundary
                )
            )

            pred_boundary_image = (
                boundary_visualization(
                    boundary_prediction[0]
                )
            )

            comparison = (
                create_comparison(
                    original,
                    gt_color,
                    pred_color,
                    gt_boundary_image,
                    pred_boundary_image,
                    index
                )
            )

            # ------------------------------------------------
            # Save first 50 examples
            # ------------------------------------------------

            if index < 50:

                comparison.save(
                    os.path.join(
                        PREDICTION_DIR,
                        f"sample_{index:04d}.png"
                    )
                )

            # ------------------------------------------------
            # Save raw prediction mask
            # ------------------------------------------------

            if index < 50:

                raw_prediction = (
                    prediction[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                Image.fromarray(
                    raw_prediction
                ).save(
                    os.path.join(
                        PREDICTION_DIR,
                        f"prediction_{index:04d}.png"
                    )
                )

            if (
                index + 1
            ) % 100 == 0:

                print(
                    f"Evaluated "
                    f"{index + 1:04d}/"
                    f"{len(dataset):04d}"
                )

    csv_file.close()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = compute_metrics(
        confusion_matrix
    )

    n = len(dataset)

    boundary_precision = (
        boundary_precision_total
        / n
    )

    boundary_recall = (
        boundary_recall_total
        / n
    )

    boundary_f1 = (
        boundary_f1_total
        / n
    )

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    iou = (
        metrics[
            "iou"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    precision = (
        metrics[
            "precision"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    recall = (
        metrics[
            "recall"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    f1 = (
        metrics[
            "f1"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    final_results = {

        "iou":
            iou,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "miou":
            float(
                metrics[
                    "miou"
                ]
            ),

        "pixel_accuracy":
            float(
                metrics[
                    "pixel_accuracy"
                ]
            ),

        "macro_precision":
            float(
                metrics[
                    "macro_precision"
                ]
            ),

        "macro_recall":
            float(
                metrics[
                    "macro_recall"
                ]
            ),

        "macro_f1":
            float(
                metrics[
                    "macro_f1"
                ]
            ),

        "boundary_precision":
            boundary_precision,

        "boundary_recall":
            boundary_recall,

        "boundary_f1":
            boundary_f1
    }

    save_csv(
        final_results
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"mIoU             : "
        f"{final_results['miou'] * 100:.2f}%"
    )

    print(
        f"Pixel Accuracy   : "
        f"{final_results['pixel_accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision        : "
        f"{final_results['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Recall           : "
        f"{final_results['macro_recall'] * 100:.2f}%"
    )

    print(
        f"F1               : "
        f"{final_results['macro_f1'] * 100:.2f}%"
    )

    print()
    print(
        f"Boundary Precision: "
        f"{boundary_precision * 100:.2f}%"
    )

    print(
        f"Boundary Recall   : "
        f"{boundary_recall * 100:.2f}%"
    )

    print(
        f"Boundary F1       : "
        f"{boundary_f1 * 100:.2f}%"
    )

    print()
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    for i in range(
        config.NUM_CLASSES
    ):

        print(
            f"{CLASS_NAMES[i]:15s} | "
            f"IoU {iou[i] * 100:6.2f}% | "
            f"Precision {precision[i] * 100:6.2f}% | "
            f"Recall {recall[i] * 100:6.2f}% | "
            f"F1 {f1[i] * 100:6.2f}%"
        )

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Comparison images:\n"
        f"{PREDICTION_DIR}"
    )

    print()

    print(
        f"Overall CSV:\n"
        f"{CSV_PATH}"
    )

    print()

    print(
        f"Per-sample CSV:\n"
        f"{per_sample_csv}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()