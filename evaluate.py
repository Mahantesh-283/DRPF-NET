# ============================================================
# DRFP-NET COMPLETE VALIDATION DATASET EVALUATION
# ============================================================

import os
import time
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# IMPORT YOUR MODEL
# ============================================================

# CHANGE THIS if your DRFPNet class is in another file.
# Example:
# from model import DRFPNet

from model import DRFPNet


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = r"E:\DRPF\checkpoints_v2\epoch_039.pth"

# ------------------------------------------------------------
# VALIDATION DATASET
# ------------------------------------------------------------

RURAL_IMAGES = r"E:\DRPF\Val\Val\Rural\images_png"
RURAL_MASKS  = r"E:\DRPF\Val\Val\Rural\masks_png"

URBAN_IMAGES = r"E:\DRPF\Val\Val\Urban\images_png"
URBAN_MASKS  = r"E:\DRPF\Val\Val\Urban\masks_png"

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

NUM_CLASSES = 7
FEATURE_CHANNELS = 96

# ------------------------------------------------------------
# Input size
# ------------------------------------------------------------

IMAGE_SIZE = 512

# ------------------------------------------------------------
# Class names
#
# LoveDA 7 classes:
# 0 Background
# 1 Building
# 2 Road
# 3 Water
# 4 Barren
# 5 Forest
# 6 Agricultural
#
# If your training code uses a different class mapping,
# CHANGE THESE NAMES to match your training labels.
# ------------------------------------------------------------

CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agricultural"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("DRFP-NET COMPLETE VALIDATION DATASET EVALUATION")
print("=" * 75)

print(f"Device     : {DEVICE}")
print(f"Checkpoint : {CHECKPOINT}")


# ============================================================
# LOAD MODEL
# ============================================================

model = DRFPNet(
    num_classes=NUM_CLASSES,
    feature_channels=FEATURE_CHANNELS
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


print()
print("Checkpoint type:", type(checkpoint))


# ------------------------------------------------------------
# Handle training checkpoint
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    print("Checkpoint keys:")
    print(list(checkpoint.keys()))

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# CHECKPOINT VERIFICATION
# ============================================================

model_state = model.state_dict()

print()
print("Checkpoint verification")
print("-" * 75)

print(
    f"Model parameters : {len(model_state)}"
)

print(
    f"Checkpoint params: {len(state_dict)}"
)


missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False
)


print(
    f"Missing keys     : {len(missing_keys)}"
)

print(
    f"Unexpected keys  : {len(unexpected_keys)}"
)


if len(missing_keys) > 0:

    print()
    print("Missing keys:")

    for key in missing_keys[:20]:
        print(" -", key)


if len(unexpected_keys) > 0:

    print()
    print("Unexpected keys:")

    for key in unexpected_keys[:20]:
        print(" -", key)


if len(missing_keys) == 0 and len(unexpected_keys) == 0:

    print()
    print("✓ DRFP-Net checkpoint loaded successfully")

else:

    raise RuntimeError(
        "Checkpoint does not match DRFP-Net architecture."
    )


model.eval()


# ============================================================
# MODEL STATISTICS
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def model_size_mb(model):

    total_params = count_parameters(model)

    return (
        total_params * 4
        /
        (1024 ** 2)
    )


# ============================================================
# MAC CALCULATION
# ============================================================

def calculate_macs(model, input_tensor):

    macs = 0
    hooks = []

    def conv_hook(module, inputs, output):

        nonlocal macs

        if isinstance(output, tuple):
            output = output[0]

        if output.dim() != 4:
            return

        batch_size = output.shape[0]

        out_channels = output.shape[1]

        out_height = output.shape[2]

        out_width = output.shape[3]

        kernel_h = module.kernel_size[0]

        kernel_w = module.kernel_size[1]

        in_channels = module.in_channels

        groups = module.groups

        kernel_ops = (
            kernel_h
            *
            kernel_w
            *
            (in_channels // groups)
        )

        macs += (
            batch_size
            *
            out_channels
            *
            out_height
            *
            out_width
            *
            kernel_ops
        )

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            hooks.append(
                module.register_forward_hook(
                    conv_hook
                )
            )

    model.eval()

    with torch.no_grad():

        model(input_tensor)

    for hook in hooks:

        hook.remove()

    return macs


# ============================================================
# PRINT MODEL STATISTICS
# ============================================================

total_params = count_parameters(model)

size_mb = model_size_mb(model)


dummy = torch.randn(
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=DEVICE
)


if DEVICE.type == "cuda":

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


with torch.no_grad():

    macs = calculate_macs(
        model,
        dummy
    )


flops = macs * 2


print()
print("=" * 75)
print("MODEL STATISTICS")
print("=" * 75)

print(
    f"Total parameters     : {total_params:,}"
)

print(
    f"Trainable parameters : "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

print(
    f"Parameters           : "
    f"{total_params / 1e6:.4f} M"
)

print(
    f"FP32 model size      : "
    f"{size_mb:.2f} MB"
)

print(
    f"MACs @ 512x512       : "
    f"{macs / 1e9:.4f} G"
)

print(
    f"FLOPs @ 512x512      : "
    f"{flops / 1e9:.4f} G"
)


# ============================================================
# FIND IMAGE/MASK PAIRS
# ============================================================

def get_image_mask_pairs(
    image_dir,
    mask_dir,
    region
):

    pairs = []

    if not os.path.isdir(image_dir):

        print(
            f"WARNING: Image directory does not exist:\n"
            f"{image_dir}"
        )

        return pairs

    if not os.path.isdir(mask_dir):

        print(
            f"WARNING: Mask directory does not exist:\n"
            f"{mask_dir}"
        )

        return pairs

    image_files = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            )
        ]
    )

    for image_name in image_files:

        image_path = os.path.join(
            image_dir,
            image_name
        )

        stem = os.path.splitext(
            image_name
        )[0]

        mask_path = None

        # ----------------------------------------------------
        # Exact same filename first
        # ----------------------------------------------------

        candidate = os.path.join(
            mask_dir,
            image_name
        )

        if os.path.isfile(candidate):

            mask_path = candidate

        else:

            # ------------------------------------------------
            # Try common extensions
            # ------------------------------------------------

            for ext in [
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff"
            ]:

                candidate = os.path.join(
                    mask_dir,
                    stem + ext
                )

                if os.path.isfile(candidate):

                    mask_path = candidate
                    break

        if mask_path is not None:

            pairs.append(
                {
                    "image": image_path,
                    "mask": mask_path,
                    "region": region
                }
            )

    return pairs


# ============================================================
# LOAD DATASET
# ============================================================

rural_pairs = get_image_mask_pairs(
    RURAL_IMAGES,
    RURAL_MASKS,
    "Rural"
)

urban_pairs = get_image_mask_pairs(
    URBAN_IMAGES,
    URBAN_MASKS,
    "Urban"
)

pairs = rural_pairs + urban_pairs


print()
print("=" * 75)
print("DATASET")
print("=" * 75)

print(
    f"Rural image/mask pairs : {len(rural_pairs)}"
)

print(
    f"Urban image/mask pairs : {len(urban_pairs)}"
)

print(
    f"Total image/mask pairs : {len(pairs)}"
)


if len(pairs) == 0:

    raise RuntimeError(
        "\nNo validation image/mask pairs found.\n\n"
        "Check these directories:\n"
        f"{RURAL_IMAGES}\n"
        f"{RURAL_MASKS}\n"
        f"{URBAN_IMAGES}\n"
        f"{URBAN_MASKS}\n"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original_size = image.size

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image = torch.from_numpy(
        image
    ).permute(
        2,
        0,
        1
    )

    image = image.unsqueeze(
        0
    )

    return image, original_size


# ============================================================
# MASK PREPROCESSING
# ============================================================

def load_mask(mask_path):

    mask = Image.open(
        mask_path
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Nearest-neighbor is mandatory for segmentation masks.
    # --------------------------------------------------------

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.NEAREST
    )

    mask = np.asarray(
        mask
    )

    # --------------------------------------------------------
    # Handle RGB masks
    # --------------------------------------------------------

    if mask.ndim == 3:

        # If RGB channels are identical,
        # simply take the first channel.

        if (
            np.array_equal(
                mask[:, :, 0],
                mask[:, :, 1]
            )
            and
            np.array_equal(
                mask[:, :, 1],
                mask[:, :, 2]
            )
        ):

            mask = mask[:, :, 0]

        else:

            # ------------------------------------------------
            # If your masks are color-coded, this section
            # needs to match your dataset's color mapping.
            # ------------------------------------------------

            raise RuntimeError(
                f"RGB mask detected:\n{mask_path}\n\n"
                "Your masks appear to be color-coded. "
                "Provide the RGB-to-class mapping used by "
                "your dataset."
            )

    mask = torch.from_numpy(
        mask.astype(np.int64)
    )

    return mask


# ============================================================
# CONFUSION MATRIX
# ============================================================

def update_confusion_matrix(
    confusion_matrix,
    target,
    prediction,
    num_classes
):

    target = target.reshape(-1)

    prediction = prediction.reshape(-1)

    valid = (
        (target >= 0)
        &
        (target < num_classes)
    )

    target = target[valid]

    prediction = prediction[valid]

    indices = (
        target * num_classes
        +
        prediction
    )

    matrix = torch.bincount(
        indices,
        minlength=num_classes * num_classes
    )

    matrix = matrix.reshape(
        num_classes,
        num_classes
    )

    confusion_matrix += (
        matrix.cpu().numpy()
    )

    return confusion_matrix


# ============================================================
# METRICS FROM CONFUSION MATRIX
# ============================================================

def calculate_metrics(
    confusion_matrix
):

    true_positive = np.diag(
        confusion_matrix
    )

    false_positive = (
        confusion_matrix.sum(
            axis=0
        )
        -
        true_positive
    )

    false_negative = (
        confusion_matrix.sum(
            axis=1
        )
        -
        true_positive
    )

    total_pixels = (
        confusion_matrix.sum()
    )

    # --------------------------------------------------------
    # Pixel accuracy
    # --------------------------------------------------------

    pixel_accuracy = (
        true_positive.sum()
        /
        max(total_pixels, 1)
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (
        true_positive
        /
        np.maximum(
            true_positive
            +
            false_positive
            +
            false_negative,
            1
        )
    )

    # --------------------------------------------------------
    # Dice / F1
    # --------------------------------------------------------

    dice = (
        2.0
        *
        true_positive
        /
        np.maximum(
            2.0 * true_positive
            +
            false_positive
            +
            false_negative,
            1
        )
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        true_positive
        /
        np.maximum(
            true_positive
            +
            false_positive,
            1
        )
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        true_positive
        /
        np.maximum(
            true_positive
            +
            false_negative,
            1
        )
    )

    # --------------------------------------------------------
    # Only classes actually present in GT
    # --------------------------------------------------------

    valid_classes = (
        (
            true_positive
            +
            false_negative
        ) > 0
    )

    if valid_classes.any():

        mean_iou = np.mean(
            iou[valid_classes]
        )

        mean_dice = np.mean(
            dice[valid_classes]
        )

    else:

        mean_iou = 0.0

        mean_dice = 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "valid_classes": valid_classes
    }


# ============================================================
# EVALUATE ONE REGION
# ============================================================

def evaluate_region(
    model,
    region_pairs,
    region_name
):

    confusion_matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    inference_times = []

    print()
    print("=" * 75)
    print(f"{region_name.upper()} EVALUATION")
    print("=" * 75)

    print(
        f"Images: {len(region_pairs)}"
    )

    for index, pair in enumerate(
        region_pairs,
        start=1
    ):

        image_path = pair["image"]

        mask_path = pair["mask"]

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image, original_size = load_image(
            image_path
        )

        target = load_mask(
            mask_path
        )

        image = image.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            output = model(
                image
            )

            logits = output[
                "logits"
            ]

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time = (
            end_time
            -
            start_time
        )

        inference_times.append(
            inference_time
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = torch.argmax(
            logits,
            dim=1
        )

        prediction = prediction[
            0
        ].cpu()

        # ----------------------------------------------------
        # Update confusion matrix
        # ----------------------------------------------------

        confusion_matrix = update_confusion_matrix(
            confusion_matrix,
            target,
            prediction,
            NUM_CLASSES
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or
            index % 25 == 0
            or
            index == len(region_pairs)
        ):

            print(
                f"\rProcessed: "
                f"{index}/{len(region_pairs)}",
                end=""
            )

    print()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        confusion_matrix
    )

    average_time = np.mean(
        inference_times
    )

    fps = (
        1.0
        /
        average_time
    )

    return {
        "confusion_matrix": confusion_matrix,
        "metrics": metrics,
        "times": inference_times,
        "average_time": average_time,
        "fps": fps
    }


# ============================================================
# EVALUATE COMPLETE DATASET
# ============================================================

print()
print("=" * 75)
print("DATASET EVALUATION")
print("=" * 75)

# ------------------------------------------------------------
# Rural
# ------------------------------------------------------------

rural_results = evaluate_region(
    model,
    rural_pairs,
    "Rural"
)

# ------------------------------------------------------------
# Urban
# ------------------------------------------------------------

urban_results = evaluate_region(
    model,
    urban_pairs,
    "Urban"
)

# ------------------------------------------------------------
# Overall
# ------------------------------------------------------------

overall_confusion_matrix = (
    rural_results["confusion_matrix"]
    +
    urban_results["confusion_matrix"]
)

overall_metrics = calculate_metrics(
    overall_confusion_matrix
)

all_times = (
    rural_results["times"]
    +
    urban_results["times"]
)

average_inference_time = np.mean(
    all_times
)

overall_fps = (
    1.0
    /
    average_inference_time
)


# ============================================================
# PRINT METRICS FUNCTION
# ============================================================

def print_metrics(
    title,
    results
):

    metrics = results["metrics"]

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)

    print(
        f"Pixel Accuracy : "
        f"{metrics['pixel_accuracy'] * 100:.4f}%"
    )

    print(
        f"Mean IoU       : "
        f"{metrics['mean_iou'] * 100:.4f}%"
    )

    print(
        f"Mean Dice/F1   : "
        f"{metrics['mean_dice'] * 100:.4f}%"
    )

    print()
    print(
        "Per-class metrics"
    )

    print("-" * 75)

    print(
        f"{'Class':<18}"
        f"{'IoU':>12}"
        f"{'Dice/F1':>14}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
    )

    print("-" * 75)

    for i in range(NUM_CLASSES):

        print(
            f"{CLASS_NAMES[i]:<18}"
            f"{metrics['iou'][i] * 100:>11.4f}%"
            f"{metrics['dice'][i] * 100:>13.4f}%"
            f"{metrics['precision'][i] * 100:>13.4f}%"
            f"{metrics['recall'][i] * 100:>13.4f}%"
        )

    print("-" * 75)

    print(
        f"Average inference time : "
        f"{results['average_time'] * 1000:.4f} ms"
    )

    print(
        f"FPS                    : "
        f"{results['fps']:.4f}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

rural_summary = {
    "metrics": rural_results["metrics"],
    "average_time": rural_results["average_time"],
    "fps": rural_results["fps"]
}

urban_summary = {
    "metrics": urban_results["metrics"],
    "average_time": urban_results["average_time"],
    "fps": urban_results["fps"]
}

overall_summary = {
    "metrics": overall_metrics,
    "average_time": average_inference_time,
    "fps": overall_fps
}


print_metrics(
    "RURAL VALIDATION RESULTS",
    rural_summary
)

print_metrics(
    "URBAN VALIDATION RESULTS",
    urban_summary
)

print_metrics(
    "OVERALL VALIDATION RESULTS",
    overall_summary
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 75)
print("OVERALL CONFUSION MATRIX")
print("=" * 75)

print(
    "Rows = Ground Truth"
)

print(
    "Columns = Prediction"
)

print()

print(
    f"{'':<18}",
    end=""
)

for name in CLASS_NAMES:

    print(
        f"{name[:10]:>12}",
        end=""
    )

print()

for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]:<18}",
        end=""
    )

    for j in range(NUM_CLASSES):

        print(
            f"{overall_confusion_matrix[i, j]:>12,}",
            end=""
        )

    print()


# ============================================================
# GPU MEMORY
# ============================================================

print()
print("=" * 75)
print("GPU MEMORY")
print("=" * 75)

if DEVICE.type == "cuda":

    allocated = (
        torch.cuda.max_memory_allocated()
        /
        (1024 ** 2)
    )

    reserved = (
        torch.cuda.max_memory_reserved()
        /
        (1024 ** 2)
    )

    print(
        f"Peak Allocated : "
        f"{allocated:.2f} MB"
    )

    print(
        f"Peak Reserved  : "
        f"{reserved:.2f} MB"
    )

else:

    print(
        "CUDA not available."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("FINAL VALIDATION SUMMARY")
print("=" * 75)

print(
    f"Total images              : "
    f"{len(pairs)}"
)

print(
    f"Rural images              : "
    f"{len(rural_pairs)}"
)

print(
    f"Urban images              : "
    f"{len(urban_pairs)}"
)

print()

print(
    f"Pixel Accuracy            : "
    f"{overall_metrics['pixel_accuracy'] * 100:.4f}%"
)

print(
    f"Mean IoU                  : "
    f"{overall_metrics['mean_iou'] * 100:.4f}%"
)

print(
    f"Mean Dice/F1              : "
    f"{overall_metrics['mean_dice'] * 100:.4f}%"
)

print()

print(
    f"Average inference time    : "
    f"{average_inference_time * 1000:.4f} ms"
)

print(
    f"Average FPS               : "
    f"{overall_fps:.4f}"
)

print()

print(
    f"Parameters                : "
    f"{total_params:,}"
)

print(
    f"Parameters (M)            : "
    f"{total_params / 1e6:.4f} M"
)

print(
    f"Model size                : "
    f"{size_mb:.2f} MB"
)

print(
    f"MACs @ 512x512            : "
    f"{macs / 1e9:.4f} G"
)

print(
    f"FLOPs @ 512x512           : "
    f"{flops / 1e9:.4f} G"
)


# ============================================================
# SAVE CSV RESULTS
# ============================================================

csv_path = r"E:\DRPF\validation_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Region",
            "Class",
            "IoU",
            "Dice_F1",
            "Precision",
            "Recall"
        ]
    )

    for region_name, result in [
        ("Overall", overall_summary),
        ("Rural", rural_summary),
        ("Urban", urban_summary)
    ]:

        metrics = result["metrics"]

        for i in range(NUM_CLASSES):

            writer.writerow(
                [
                    region_name,
                    CLASS_NAMES[i],
                    metrics["iou"][i],
                    metrics["dice"][i],
                    metrics["precision"][i],
                    metrics["recall"][i]
                ]
            )

        writer.writerow(
            [
                region_name,
                "MEAN",
                metrics["mean_iou"],
                metrics["mean_dice"],
                "",
                ""
            ]
        )

    writer.writerow(
        [
            "Overall",
            "Pixel Accuracy",
            overall_metrics[
                "pixel_accuracy"
            ],
            "",
            "",
            ""
        ]
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = r"E:\DRPF\confusion_matrix.csv"

with open(
    cm_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["GroundTruth/Prediction"]
        +
        CLASS_NAMES
    )

    for i in range(NUM_CLASSES):

        writer.writerow(
            [
                CLASS_NAMES[i]
            ]
            +
            list(
                overall_confusion_matrix[i]
            )
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = r"E:\DRPF\evaluation_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "DRFP-NET VALIDATION EVALUATION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Checkpoint: {CHECKPOINT}\n"
    )

    f.write(
        f"Device: {DEVICE}\n\n"
    )

    f.write(
        f"Total images: {len(pairs)}\n"
    )

    f.write(
        f"Rural images: {len(rural_pairs)}\n"
    )

    f.write(
        f"Urban images: {len(urban_pairs)}\n\n"
    )

    f.write(
        f"Pixel Accuracy: "
        f"{overall_metrics['pixel_accuracy'] * 100:.6f}%\n"
    )

    f.write(
        f"Mean IoU: "
        f"{overall_metrics['mean_iou'] * 100:.6f}%\n"
    )

    f.write(
        f"Mean Dice/F1: "
        f"{overall_metrics['mean_dice'] * 100:.6f}%\n\n"
    )

    f.write(
        f"Average inference time: "
        f"{average_inference_time * 1000:.6f} ms\n"
    )

    f.write(
        f"FPS: "
        f"{overall_fps:.6f}\n\n"
    )

    f.write(
        f"Parameters: "
        f"{total_params:,}\n"
    )

    f.write(
        f"Model size: "
        f"{size_mb:.6f} MB\n"
    )

    f.write(
        f"MACs: "
        f"{macs / 1e9:.6f} G\n"
    )

    f.write(
        f"FLOPs: "
        f"{flops / 1e9:.6f} G\n"
    )

    f.write("\n")

    f.write(
        "PER CLASS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    for i in range(NUM_CLASSES):

        f.write(
            f"{CLASS_NAMES[i]}: "
            f"IoU={overall_metrics['iou'][i] * 100:.6f}%, "
            f"Dice={overall_metrics['dice'][i] * 100:.6f}%, "
            f"Precision={overall_metrics['precision'][i] * 100:.6f}%, "
            f"Recall={overall_metrics['recall'][i] * 100:.6f}%\n"
        )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 75)
print("RESULT FILES")
print("=" * 75)

print(
    f"CSV metrics       : {csv_path}"
)

print(
    f"Confusion matrix  : {cm_path}"
)

print(
    f"Summary report    : {summary_path}"
)

print()
print("=" * 75)
print("✓ ALL DRFP-NET VALIDATION TESTS COMPLETED")
print("=" * 75)