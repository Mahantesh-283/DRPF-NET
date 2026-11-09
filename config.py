# ============================================================
# DRFP-Net V3 CONFIGURATION
# ============================================================

import os
import torch


# ============================================================
# DIRECTORIES
# ============================================================

ROOT_DIR = r"E:\DRPF"

TRAIN_DIR = os.path.join(
    ROOT_DIR,
    "Train",
    "Train"
)

VAL_DIR = os.path.join(
    ROOT_DIR,
    "Val",
    "Val"
)

TEST_DIR = os.path.join(
    ROOT_DIR,
    "Test",
    "Test"
)


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINT_DIR = os.path.join(
    ROOT_DIR,
    "checkpoints_v3"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "last_checkpoint.pth"
)

# Checkpoint containing the exact state when Ctrl+C occurs.
INTERRUPTED_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "interrupted_checkpoint.pth"
)


# ============================================================
# DATA
# ============================================================

IMAGE_SIZE = 512

NUM_CLASSES = 7

IGNORE_INDEX = 255


# ============================================================
# MODEL
# ============================================================

FEATURE_CHANNELS = 96


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 2

NUM_EPOCHS = 80

LEARNING_RATE = 3e-4

MIN_LR = 1e-6

WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 5

GRAD_CLIP = 1.0


# ============================================================
# LOSS
# ============================================================

CE_WEIGHT = 0.70

DICE_WEIGHT = 0.30


# ============================================================
# AUGMENTATION
# ============================================================

TRAIN_AUGMENTATION = True

VAL_AUGMENTATION = False


# ============================================================
# DATALOADER
# ============================================================

NUM_WORKERS = 2

PIN_MEMORY = True


# ============================================================
# AMP
# ============================================================

AMP = True


# ============================================================
# SEED
# ============================================================

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT CONFIG
# ============================================================

def print_config():

    print("=" * 70)
    print("DRFP-Net V3 CONFIGURATION")
    print("=" * 70)

    print(
        f"Train directory : {TRAIN_DIR}"
    )

    print(
        f"Validation      : {VAL_DIR}"
    )

    print(
        f"Test directory  : {TEST_DIR}"
    )

    print(
        f"Image size      : {IMAGE_SIZE}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Epochs          : {NUM_EPOCHS}"
    )

    print(
        f"Learning rate   : {LEARNING_RATE}"
    )

    print(
        f"Weight decay    : {WEIGHT_DECAY}"
    )

    print(
        f"Feature channels: {FEATURE_CHANNELS}"
    )

    print(
        f"Classes         : {NUM_CLASSES}"
    )

    print(
        f"AMP             : {AMP}"
    )

    print(
        f"Checkpoint dir  : {CHECKPOINT_DIR}"
    )

    print("=" * 70)