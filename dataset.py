
# ============================================================
# DRFP-Net V2
# LoveDA DATASET
# ============================================================

import os
import random

import numpy as np

from PIL import Image, ImageEnhance

import torch

from torch.utils.data import Dataset

import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode


# ============================================================
# LOVE-DA DATASET
# ============================================================

class LoveDADataset(Dataset):

    def __init__(
        self,
        root_dir,
        image_size=512,
        augmentation=False
    ):

        self.root_dir = root_dir
        self.image_size = image_size
        self.augmentation = augmentation

        self.samples = []

        # ====================================================
        # DOMAINS
        # ====================================================

        domains = [
            "Rural",
            "Urban"
        ]

        # ====================================================
        # FIND DATA
        # ====================================================

        for domain in domains:

            image_dir = os.path.join(
                root_dir,
                domain,
                "images_png"
            )

            mask_dir = os.path.join(
                root_dir,
                domain,
                "masks_png"
            )

            # ------------------------------------------------
            # Check image directory
            # ------------------------------------------------

            if not os.path.isdir(image_dir):

                print(
                    f"WARNING: Missing image directory:\n"
                    f"{image_dir}"
                )

                continue

            # ------------------------------------------------
            # Check mask directory
            # ------------------------------------------------

            if not os.path.isdir(mask_dir):

                print(
                    f"WARNING: Missing mask directory:\n"
                    f"{mask_dir}"
                )

                continue

            # ------------------------------------------------
            # Find images
            # ------------------------------------------------

            for filename in sorted(
                os.listdir(image_dir)
            ):

                if not filename.lower().endswith(
                    ".png"
                ):
                    continue

                image_path = os.path.join(
                    image_dir,
                    filename
                )

                mask_path = os.path.join(
                    mask_dir,
                    filename
                )

                # ------------------------------------------------
                # Check corresponding mask
                # ------------------------------------------------

                if not os.path.exists(mask_path):

                    print(
                        f"WARNING: Missing mask:\n"
                        f"{mask_path}"
                    )

                    continue

                self.samples.append(
                    (
                        image_path,
                        mask_path,
                        domain
                    )
                )

        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        rural_count = sum(
            d == "Rural"
            for _, _, d in self.samples
        )

        urban_count = sum(
            d == "Urban"
            for _, _, d in self.samples
        )

        print()
        print("=" * 70)
        print("LoveDA DATASET V2")
        print("=" * 70)

        print(
            f"Root directory : {root_dir}"
        )

        print(
            f"Image size     : "
            f"{image_size} x {image_size}"
        )

        print(
            f"Augmentation   : "
            f"{augmentation}"
        )

        print(
            f"Total samples  : "
            f"{len(self.samples)}"
        )

        print(
            f"Rural samples  : "
            f"{rural_count}"
        )

        print(
            f"Urban samples  : "
            f"{urban_count}"
        )

        print("=" * 70)

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path, mask_path, domain = \
            self.samples[index]

        # ====================================================
        # IMAGE
        # ====================================================

        image = Image.open(
            image_path
        ).convert("RGB")

        # ====================================================
        # MASK
        # ====================================================

        mask = Image.open(
            mask_path
        ).convert("L")

        # ====================================================
        # RESIZE IMAGE
        # ====================================================

        image = TF.resize(
            image,
            [
                self.image_size,
                self.image_size
            ],
            interpolation=InterpolationMode.BILINEAR
        )

        # ====================================================
        # RESIZE MASK
        # ====================================================

        mask = TF.resize(
            mask,
            [
                self.image_size,
                self.image_size
            ],
            interpolation=InterpolationMode.NEAREST
        )

        # ====================================================
        # AUGMENTATION
        # ====================================================

        if self.augmentation:

            # ------------------------------------------------
            # Horizontal flip
            # ------------------------------------------------

            if random.random() < 0.5:

                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # ------------------------------------------------
            # Vertical flip
            # ------------------------------------------------

            if random.random() < 0.5:

                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # ------------------------------------------------
            # 90-degree rotation
            # ------------------------------------------------

            rotation = random.randint(
                0,
                3
            )

            if rotation != 0:

                angle = 90 * rotation

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=(
                        InterpolationMode.BILINEAR
                    )
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=(
                        InterpolationMode.NEAREST
                    )
                )

            # ------------------------------------------------
            # Brightness
            # ------------------------------------------------

            if random.random() < 0.30:

                image = ImageEnhance.Brightness(
                    image
                ).enhance(
                    random.uniform(
                        0.90,
                        1.10
                    )
                )

            # ------------------------------------------------
            # Contrast
            # ------------------------------------------------

            if random.random() < 0.30:

                image = ImageEnhance.Contrast(
                    image
                ).enhance(
                    random.uniform(
                        0.90,
                        1.10
                    )
                )

            # ------------------------------------------------
            # Color
            # ------------------------------------------------

            if random.random() < 0.20:

                image = ImageEnhance.Color(
                    image
                ).enhance(
                    random.uniform(
                        0.95,
                        1.05
                    )
                )

        # ====================================================
        # IMAGE → TENSOR
        # ====================================================

        image = TF.to_tensor(
            image
        )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        image = TF.normalize(
            image,
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

        # ====================================================
        # MASK → NUMPY
        # ====================================================

        mask = np.asarray(
            mask,
            dtype=np.int64
        )

        # ====================================================
        # VALIDATE ORIGINAL LABELS
        # ====================================================

        unique_labels = np.unique(
            mask
        )

        invalid = [
            int(x)
            for x in unique_labels
            if x < 0 or x > 7
        ]

        if invalid:

            raise ValueError(
                "\n"
                "Invalid mask labels!\n"
                f"Image: {image_path}\n"
                f"Mask : {mask_path}\n"
                f"Unique labels: "
                f"{unique_labels.tolist()}\n"
                f"Invalid labels: "
                f"{invalid}\n"
                "Expected original labels 0-7."
            )

        # ====================================================
        # LOVE-DA LABEL CONVERSION
        # ====================================================
        #
        # Original LoveDA:
        #
        # 0 = ignore
        # 1 = class 0
        # 2 = class 1
        # 3 = class 2
        # 4 = class 3
        # 5 = class 4
        # 6 = class 5
        # 7 = class 6
        #
        # Model:
        #
        # 255 = ignore
        # 0-6 = classes
        # ====================================================

        mask[
            mask == 0
        ] = 255

        valid = (
            mask != 255
        )

        mask[
            valid
        ] -= 1

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        final_labels = np.unique(
            mask
        )

        invalid_final = [
            int(x)
            for x in final_labels
            if x != 255 and not (
                0 <= x <= 6
            )
        ]

        if invalid_final:

            raise ValueError(
                "\n"
                "Invalid converted labels!\n"
                f"Image: {image_path}\n"
                f"Labels: "
                f"{final_labels.tolist()}\n"
                f"Invalid: "
                f"{invalid_final}"
            )

        # ====================================================
        # RETURN
        # ====================================================

        return (
            image,
            torch.from_numpy(
                mask
            ).long()
        )


# ============================================================
# FAST FULL DATASET LABEL CHECK
# ============================================================

def check_all_labels(dataset):

    print()
    print("=" * 70)
    print("FULL DATASET LABEL CHECK")
    print("=" * 70)

    total_samples = len(
        dataset.samples
    )

    print(
        f"Scanning {total_samples} masks..."
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    class_counts = {
        i: 0
        for i in range(7)
    }

    ignore_count = 0

    invalid_labels = {}

    # ========================================================
    # SCAN MASKS ONLY
    # ========================================================

    for index, (
        image_path,
        mask_path,
        domain
    ) in enumerate(
        dataset.samples
    ):

        # ----------------------------------------------------
        # Open ONLY mask
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = np.asarray(
            mask,
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Count original labels
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            mask,
            return_counts=True
        )

        for label, count in zip(
            unique_labels,
            counts
        ):

            label = int(label)
            count = int(count)

            # ----------------------------------------------
            # Original label 0 = ignore
            # ----------------------------------------------

            if label == 0:

                ignore_count += count

            # ----------------------------------------------
            # Original labels 1-7
            # ----------------------------------------------

            elif 1 <= label <= 7:

                converted_label = label - 1

                class_counts[
                    converted_label
                ] += count

            # ----------------------------------------------
            # Invalid label
            # ----------------------------------------------

            else:

                invalid_labels[label] = (
                    invalid_labels.get(
                        label,
                        0
                    )
                    +
                    count
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 100 == 0
            or index == 0
            or index + 1 == total_samples
        ):

            print(
                f"Scanned "
                f"{index + 1}/"
                f"{total_samples} masks"
            )

    # ========================================================
    # INVALID LABEL CHECK
    # ========================================================

    if invalid_labels:

        print()
        print(
            "ERROR: INVALID LABELS FOUND"
        )

        for label, count in sorted(
            invalid_labels.items()
        ):

            print(
                f"Label {label}: "
                f"{count:,} pixels"
            )

        raise ValueError(
            "Dataset contains invalid mask labels."
        )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print("-" * 70)
    print("CLASS PIXEL DISTRIBUTION")
    print("-" * 70)

    for class_id in range(7):

        print(
            f"Class {class_id}: "
            f"{class_counts[class_id]:,} pixels"
        )

    print(
        f"Ignore (255): "
        f"{ignore_count:,} pixels"
    )

    # ========================================================
    # CLASS PRESENCE
    # ========================================================

    missing_classes = [
        class_id
        for class_id in range(7)
        if class_counts[class_id] == 0
    ]

    print()
    print("-" * 70)
    print("CLASS PRESENCE")
    print("-" * 70)

    if not missing_classes:

        print(
            "All 7 classes are present."
        )

    else:

        print(
            "WARNING: Missing classes:",
            missing_classes
        )

    # ========================================================
    # PIXEL SUMMARY
    # ========================================================

    valid_pixels = sum(
        class_counts.values()
    )

    total_pixels = (
        valid_pixels
        +
        ignore_count
    )

    print()
    print("-" * 70)
    print("PIXEL SUMMARY")
    print("-" * 70)

    print(
        f"Valid pixels : "
        f"{valid_pixels:,}"
    )

    print(
        f"Ignore pixels: "
        f"{ignore_count:,}"
    )

    print(
        f"Total pixels : "
        f"{total_pixels:,}"
    )

    # ========================================================
    # CLASS PERCENTAGES
    # ========================================================

    print()
    print("-" * 70)
    print("CLASS PIXEL PERCENTAGES")
    print("-" * 70)

    if valid_pixels > 0:

        for class_id in range(7):

            percentage = (
                class_counts[class_id]
                /
                valid_pixels
                *
                100.0
            )

            print(
                f"Class {class_id}: "
                f"{percentage:.2f}%"
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 70)

    if not missing_classes:

        print(
            "DATASET LABEL CHECK PASSED"
        )

    else:

        print(
            "DATASET LABEL CHECK WARNING"
        )

    print("=" * 70)


# ============================================================
# SINGLE SAMPLE TEST
# ============================================================

def test_dataset(dataset):

    image, mask = dataset[0]

    print()
    print("=" * 70)
    print("DATASET TEST")
    print("=" * 70)

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Mask shape :",
        mask.shape
    )

    print(
        "Image dtype:",
        image.dtype
    )

    print(
        "Mask dtype :",
        mask.dtype
    )

    print(
        "Sample mask labels:",
        torch.unique(
            mask
        ).tolist()
    )

    print(
        "Image range:",
        image.min().item(),
        "to",
        image.max().item()
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import config

    # ========================================================
    # DATASET
    # ========================================================

    dataset = LoveDADataset(
        root_dir=config.TRAIN_DIR,
        image_size=config.IMAGE_SIZE,
        augmentation=False
    )

    # ========================================================
    # SINGLE SAMPLE TEST
    # ========================================================

    test_dataset(
        dataset
    )

    # ========================================================
    # FULL DATASET LABEL CHECK
    # ========================================================

    check_all_labels(
        dataset
    )
