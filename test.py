
# ============================================================
# DRFP-Net V3 TEST INFERENCE
#
# Test set has NO ground-truth masks.
#
# This script:
#   1. Loads best_model.pth
#   2. Runs inference on all Test images
#   3. Saves raw prediction masks
#   4. Saves colorized segmentation maps
#   5. Saves input + prediction visualizations
#   6. Reports Rural / Urban image counts
#
# IMPORTANT:
#   No mIoU is calculated because Test has no masks.
# ============================================================

import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from model import DRFPNet

import config


# ============================================================
# PATHS
# ============================================================

TEST_ROOT = Path(config.TEST_DIR)

CHECKPOINT_PATH = Path(
    config.BEST_CHECKPOINT
)

OUTPUT_ROOT = Path(
    config.ROOT_DIR
) / "predictions_v3"


RAW_MASK_ROOT = (
    OUTPUT_ROOT / "raw_masks"
)

COLOR_MASK_ROOT = (
    OUTPUT_ROOT / "color_masks"
)

VIS_ROOT = (
    OUTPUT_ROOT / "visualizations"
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp"
}


# ============================================================
# NORMALIZATION
#
# MUST match dataset.py
# ============================================================

NORMALIZE = transforms.Normalize(
    mean=[
        0.485,
        0.456,
        0.406
    ],
    std=[
        0.229,
        0.224,
        0.225
    ]
)


# ============================================================
# LOVE-DA CLASS COLORS
#
# 7 classes
#
# These colors are ONLY for visualization.
# Raw prediction masks contain class IDs 0-6.
# ============================================================

CLASS_COLORS = np.array(
    [
        [0, 0, 0],          # Class 0
        [255, 0, 0],        # Class 1
        [0, 255, 0],        # Class 2
        [0, 0, 255],        # Class 3
        [255, 255, 0],      # Class 4
        [255, 0, 255],      # Class 5
        [0, 255, 255],      # Class 6
    ],
    dtype=np.uint8
)


# ============================================================
# DEVICE
# ============================================================

device = config.DEVICE


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_output_directories():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    RAW_MASK_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    COLOR_MASK_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    VIS_ROOT.mkdir(
        parents=True,
        exist_ok=True

    )


# ============================================================
# FIND TEST IMAGES
# ============================================================

def find_test_images():

    image_list = []

    # --------------------------------------------------------
    # Expected structure:
    #
    # Test/Test/Rural/images_png
    # Test/Test/Urban/images_png
    # --------------------------------------------------------

    for region in [
        "Rural",
        "Urban"
    ]:

        image_dir = (
            TEST_ROOT
            / region
            / "images_png"
        )

        if not image_dir.exists():

            print()
            print(
                f"WARNING: image directory missing:"
            )

            print(
                image_dir
            )

            continue

        for path in sorted(
            image_dir.iterdir()
        ):

            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                image_list.append(
                    (
                        region,
                        path
                    )
                )

    return image_list


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    original_width, original_height = (
        image.size
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image_resized = image.resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # To tensor
    # --------------------------------------------------------

    image_tensor = transforms.ToTensor()(
        image_resized
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_tensor = NORMALIZE(
        image_tensor
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    return (
        image,
        image_tensor,
        original_width,
        original_height
    )


# ============================================================
# COLORIZE PREDICTION
# ============================================================

def colorize_prediction(
    prediction
):

    color_mask = CLASS_COLORS[
        prediction
    ]

    return Image.fromarray(
        color_mask
    )


# ============================================================
# CREATE SIDE-BY-SIDE VISUALIZATION
# ============================================================

def create_visualization(
    original_image,
    color_mask
):

    # --------------------------------------------------------
    # Resize original to model resolution
    # --------------------------------------------------------

    original_resized = original_image.resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Create combined image
    # --------------------------------------------------------

    combined = Image.new(
        "RGB",
        (
            config.IMAGE_SIZE * 2,
            config.IMAGE_SIZE
        )
    )

    combined.paste(
        original_resized,
        (0, 0)
    )

    combined.paste(
        color_mask,
        (
            config.IMAGE_SIZE,
            0
        )
    )

    return combined


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    print(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = DRFPNet(
        num_classes=config.NUM_CLASSES,
        feature_channels=config.FEATURE_CHANNELS
    ).to(device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    saved_epoch = checkpoint.get(
        "epoch",
        -1
    )

    best_miou = checkpoint.get(
        "best_miou",
        0.0
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Checkpoint epoch : "
        f"{saved_epoch + 1}"
    )

    print(
        f"Best validation mIoU : "
        f"{best_miou * 100:.2f}%"
    )

    print(
        f"Parameters : "
        f"{total_parameters:,}"
    )

    print(
        f"Parameters (M) : "
        f"{total_parameters / 1e6:.4f}"
    )

    return model


# ============================================================
# RUN INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    image_list
):

    print()
    print("=" * 70)
    print("RUNNING TEST INFERENCE")
    print("=" * 70)

    rural_count = 0
    urban_count = 0

    total = len(
        image_list
    )

    if total == 0:

        print()
        print(
            "ERROR: No test images found."
        )

        print()
        print(
            "Expected:"
        )

        print(
            TEST_ROOT
            / "Rural"
            / "images_png"
        )

        print(
            TEST_ROOT
            / "Urban"
            / "images_png"
        )

        return 0

    for index, (
        region,
        image_path
    ) in enumerate(
        image_list,
        start=1
    ):

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        (
            original_image,
            image_tensor,
            original_width,
            original_height
        ) = load_image(
            image_path
        )

        image_tensor = image_tensor.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                config.AMP
                and
                device.type == "cuda"
            )
        ):

            output = model(
                image_tensor
            )

            logits = output[
                "logits"
            ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = torch.argmax(
            logits,
            dim=1
        )

        prediction = prediction[
            0
        ].cpu().numpy().astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Output directories
        # ----------------------------------------------------

        raw_dir = (
            RAW_MASK_ROOT
            / region
        )

        color_dir = (
            COLOR_MASK_ROOT
            / region
        )

        vis_dir = (
            VIS_ROOT
            / region
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        color_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        vis_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        filename = image_path.stem

        # ----------------------------------------------------
        # Raw class-ID mask
        #
        # Values:
        #   0,1,2,3,4,5,6
        # ----------------------------------------------------

        raw_mask_path = (
            raw_dir
            / f"{filename}.png"
        )

        Image.fromarray(
            prediction,
            mode="L"
        ).save(
            raw_mask_path
        )

        # ----------------------------------------------------
        # Color mask
        # ----------------------------------------------------

        color_mask = colorize_prediction(
            prediction
        )

        color_mask_path = (
            color_dir
            / f"{filename}.png"
        )

        color_mask.save(
            color_mask_path
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        visualization = create_visualization(
            original_image,
            color_mask
        )

        visualization_path = (
            vis_dir
            / f"{filename}.png"
        )

        visualization.save(
            visualization_path
        )

        # ----------------------------------------------------
        # Region count
        # ----------------------------------------------------

        if region == "Rural":

            rural_count += 1

        elif region == "Urban":

            urban_count += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or
            index % 100 == 0
            or
            index == total
        ):

            print(
                f"Processed "
                f"{index:05d}/{total:05d} | "
                f"{region:5s} | "
                f"{image_path.name}"
            )

    return total


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DRFP-Net V3 TEST INFERENCE")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Test directory:\n"
        f"{TEST_ROOT}"
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    create_output_directories()

    # ========================================================
    # FIND IMAGES
    # ========================================================

    print()
    print("=" * 70)
    print("SCANNING TEST IMAGES")
    print("=" * 70)

    image_list = find_test_images()

    rural_images = sum(
        1
        for region, path
        in image_list
        if region == "Rural"
    )

    urban_images = sum(
        1
        for region, path
        in image_list
        if region == "Urban"
    )

    print()
    print(
        f"Total test images : "
        f"{len(image_list)}"
    )

    print(
        f"Rural images      : "
        f"{rural_images}"
    )

    print(
        f"Urban images      : "
        f"{urban_images}"
    )

    if len(image_list) == 0:

        print()
        print(
            "NO TEST IMAGES FOUND."
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # INFERENCE
    # ========================================================

    processed = run_inference(
        model,
        image_list
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("TEST INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Images processed : "
        f"{processed}"
    )

    print()
    print(
        f"Raw masks:\n"
        f"{RAW_MASK_ROOT}"
    )

    print()
    print(
        f"Color masks:\n"
        f"{COLOR_MASK_ROOT}"
    )

    print()
    print(
        f"Visualizations:\n"
        f"{VIS_ROOT}"
    )

    print()
    print("=" * 70)

    print(
        "IMPORTANT"
    )

    print("=" * 70)

    print(
        "Test mIoU was NOT calculated because "
        "the Test dataset has no ground-truth masks."
    )

    print()
    print(
        "The predictions are generated from the "
        "BEST validation checkpoint."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
