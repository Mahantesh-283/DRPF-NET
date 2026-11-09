# ============================================================
# DRFP-Net V4 TRAINING
#
# Features:
#   - Automatic resume
#   - Ctrl+C graceful stop
#   - Best checkpoint
#   - Last checkpoint
#   - AMP scaler restoration
#   - Optimizer restoration
#   - Model restoration
#   - Best mIoU restoration
#   - RNG restoration
#   - Warmup + cosine LR
#   - Gradient clipping
#   - Semantic metrics
#   - Boundary metrics
# ============================================================

import os
import random
import numpy as np

import torch

from torch.utils.data import DataLoader

from dataset import LoveDADataset
from model import DRFPNet
from losses import DRFPLoss
from metrics import (
    update_confusion_matrix,
    compute_metrics,
    boundary_metrics
)

import config


# ============================================================
# STOP FLAG
# ============================================================

STOP_REQUESTED = False


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    torch.backends.cudnn.deterministic = False


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(epoch):

    if epoch < config.WARMUP_EPOCHS:

        alpha = (
            epoch + 1
        ) / max(
            1,
            config.WARMUP_EPOCHS
        )

        return (
            config.LEARNING_RATE
            * alpha
        )

    progress = (
        epoch
        - config.WARMUP_EPOCHS
    ) / max(
        1,
        config.NUM_EPOCHS
        - config.WARMUP_EPOCHS
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    cosine = (
        0.5
        *
        (
            1.0
            +
            np.cos(
                np.pi * progress
            )
        )
    )

    return (
        config.MIN_LR
        +
        (
            config.LEARNING_RATE
            -
            config.MIN_LR
        )
        *
        cosine
    )


# ============================================================
# SET LR
# ============================================================

def set_epoch_lr(
    optimizer,
    epoch
):

    lr = get_learning_rate(
        epoch
    )

    for group in optimizer.param_groups:

        group["lr"] = lr

    return lr


# ============================================================
# CHECKPOINT PATHS
# ============================================================

def get_checkpoint_paths():

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    best_path = getattr(
        config,
        "BEST_CHECKPOINT",
        os.path.join(
            config.CHECKPOINT_DIR,
            "best_model.pth"
        )
    )

    last_path = getattr(
        config,
        "LAST_CHECKPOINT",
        os.path.join(
            config.CHECKPOINT_DIR,
            "last_checkpoint.pth"
        )
    )

    return (
        best_path,
        last_path
    )


# ============================================================
# SAVE
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_miou
):

    checkpoint = {

        "epoch":
            int(epoch),

        "best_miou":
            float(best_miou),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict()
            if scaler is not None
            else None,

        "torch_rng_state":
            torch.get_rng_state(),

        "numpy_rng_state":
            np.random.get_state(),

        "python_rng_state":
            random.getstate(),

        "cuda_rng_state":
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
    }

    temporary_path = (
        path + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path
    )

    os.replace(
        temporary_path,
        path
    )


# ============================================================
# LOAD
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    device
):

    print()
    print("=" * 70)
    print("RESUMING TRAINING")
    print("=" * 70)

    print(
        f"Checkpoint:\n{path}"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if (
        scaler is not None
        and
        scaler_state is not None
    ):

        scaler.load_state_dict(
            scaler_state
        )

    saved_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    best_miou = float(
        checkpoint.get(
            "best_miou",
            0.0
        )
    )

    try:

        torch.set_rng_state(
            checkpoint[
                "torch_rng_state"
            ]
        )

    except Exception:
        pass

    try:

        np.random.set_state(
            checkpoint[
                "numpy_rng_state"
            ]
        )

    except Exception:
        pass

    try:

        random.setstate(
            checkpoint[
                "python_rng_state"
            ]
        )

    except Exception:
        pass

    if torch.cuda.is_available():

        try:

            torch.cuda.set_rng_state_all(
                checkpoint[
                    "cuda_rng_state"
                ]
            )

        except Exception:
            pass

    print(
        f"Completed epoch : "
        f"{saved_epoch + 1}"
    )

    print(
        f"Best mIoU       : "
        f"{best_miou * 100:.2f}%"
    )

    print("=" * 70)

    return (
        saved_epoch + 1,
        best_miou
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch
):

    global STOP_REQUESTED

    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_boundary_bce = 0.0
    total_boundary_dice = 0.0

    confusion_matrix = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.int64,
        device=device
    )

    num_batches = len(loader)

    for batch_index, (
        images,
        targets
    ) in enumerate(loader):

        if STOP_REQUESTED:
            break

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
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

            losses = criterion(
                logits,
                targets,
                boundary_logits
            )

            loss = losses[
                "loss"
            ]

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            loss.item()
        )

        total_ce += (
            losses[
                "ce"
            ].item()
        )

        total_dice += (
            losses[
                "dice"
            ].item()
        )

        total_boundary_bce += (
            losses[
                "boundary_bce"
            ].item()
        )

        total_boundary_dice += (
            losses[
                "boundary_dice"
            ].item()
        )

        prediction = torch.argmax(
            logits,
            dim=1
        )

        confusion_matrix = (
            update_confusion_matrix(
                confusion_matrix,
                prediction,
                targets,
                config.NUM_CLASSES,
                config.IGNORE_INDEX
            )
        )

        if (
            batch_index + 1
        ) % 20 == 0:

            print(
                f"Epoch "
                f"{epoch + 1:03d} | "
                f"Batch "
                f"{batch_index + 1:04d}/"
                f"{num_batches:04d} | "
                f"Loss "
                f"{loss.item():.4f}"
            )

    if STOP_REQUESTED:
        return None

    metrics = compute_metrics(
        confusion_matrix
    )

    return {

        "loss":
            total_loss / num_batches,

        "ce":
            total_ce / num_batches,

        "dice":
            total_dice / num_batches,

        "boundary_bce":
            total_boundary_bce / num_batches,

        "boundary_dice":
            total_boundary_dice / num_batches,

        "metrics":
            metrics
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_boundary_bce = 0.0
    total_boundary_dice = 0.0

    confusion_matrix = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.int64,
        device=device
    )

    boundary_precision_sum = 0.0
    boundary_recall_sum = 0.0
    boundary_f1_sum = 0.0

    count = 0

    for images, targets in loader:

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

            losses = criterion(
                logits,
                targets,
                boundary_logits
            )

        total_loss += (
            losses[
                "loss"
            ].item()
        )

        total_ce += (
            losses[
                "ce"
            ].item()
        )

        total_dice += (
            losses[
                "dice"
            ].item()
        )

        total_boundary_bce += (
            losses[
                "boundary_bce"
            ].item()
        )

        total_boundary_dice += (
            losses[
                "boundary_dice"
            ].item()
        )

        prediction = torch.argmax(
            logits,
            dim=1
        )

        confusion_matrix = (
            update_confusion_matrix(
                confusion_matrix,
                prediction,
                targets,
                config.NUM_CLASSES,
                config.IGNORE_INDEX
            )
        )

        bmetrics = boundary_metrics(
            prediction,
            targets,
            config.IGNORE_INDEX,
            tolerance=2
        )

        boundary_precision_sum += (
            bmetrics[
                "boundary_precision"
            ].item()
        )

        boundary_recall_sum += (
            bmetrics[
                "boundary_recall"
            ].item()
        )

        boundary_f1_sum += (
            bmetrics[
                "boundary_f1"
            ].item()
        )

        count += 1

    metrics = compute_metrics(
        confusion_matrix
    )

    metrics[
        "boundary_precision"
    ] = (
        boundary_precision_sum
        / max(count, 1)
    )

    metrics[
        "boundary_recall"
    ] = (
        boundary_recall_sum
        / max(count, 1)
    )

    metrics[
        "boundary_f1"
    ] = (
        boundary_f1_sum
        / max(count, 1)
    )

    return {

        "loss":
            total_loss / count,

        "ce":
            total_ce / count,

        "dice":
            total_dice / count,

        "boundary_bce":
            total_boundary_bce / count,

        "boundary_dice":
            total_boundary_dice / count,

        "metrics":
            metrics
    }


# ============================================================
# PRINT
# ============================================================

def print_results(
    title,
    results
):

    metrics = results[
        "metrics"
    ]

    print()
    print(title)

    print(
        f"Loss            : "
        f"{results['loss']:.4f}"
    )

    print(
        f"CE              : "
        f"{results['ce']:.4f}"
    )

    print(
        f"Dice            : "
        f"{results['dice']:.4f}"
    )

    print(
        f"Boundary BCE    : "
        f"{results['boundary_bce']:.4f}"
    )

    print(
        f"Boundary Dice   : "
        f"{results['boundary_dice']:.4f}"
    )

    print(
        f"mIoU            : "
        f"{metrics['miou'].item() * 100:.2f}%"
    )

    print(
        f"Pixel Accuracy  : "
        f"{metrics['pixel_accuracy'].item() * 100:.2f}%"
    )

    print(
        f"Precision       : "
        f"{metrics['macro_precision'].item() * 100:.2f}%"
    )

    print(
        f"Recall          : "
        f"{metrics['macro_recall'].item() * 100:.2f}%"
    )

    print(
        f"F1              : "
        f"{metrics['macro_f1'].item() * 100:.2f}%"
    )

    print(
        f"Boundary Prec.  : "
        f"{float(metrics['boundary_precision']) * 100:.2f}%"
    )

    print(
        f"Boundary Recall : "
        f"{float(metrics['boundary_recall']) * 100:.2f}%"
    )

    print(
        f"Boundary F1     : "
        f"{float(metrics['boundary_f1']) * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global STOP_REQUESTED

    set_seed(
        config.SEED
    )

    config.print_config()

    device = config.DEVICE

    best_checkpoint, last_checkpoint = (
        get_checkpoint_paths()
    )

    print()
    print("=" * 70)
    print("DRFP-Net V4 BOUNDARY-AWARE TRAINING")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("Loading training dataset...")

    train_dataset = LoveDADataset(
        config.TRAIN_DIR,
        image_size=config.IMAGE_SIZE,
        augmentation=config.TRAIN_AUGMENTATION
    )

    print()
    print("Loading validation dataset...")

    val_dataset = LoveDADataset(
        config.VAL_DIR,
        image_size=config.IMAGE_SIZE,
        augmentation=config.VAL_AUGMENTATION
    )

    print()
    print(
        f"Training samples  : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # WINDOWS SAFE DATALOADERS
    # --------------------------------------------------------
    #
    # num_workers=0 is intentionally used here.
    #
    # You already hit the Windows multiprocessing problem in
    # test1.py. Training stability matters more than squeezing
    # a little extra dataloader speed.
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=config.PIN_MEMORY
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = DRFPNet(
        num_classes=config.NUM_CLASSES,
        feature_channels=config.FEATURE_CHANNELS
    ).to(device)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = DRFPLoss(
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX,
        ce_weight=0.50,
        dice_weight=0.20,
        boundary_bce_weight=0.15,
        boundary_dice_weight=0.15
    ).to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            config.AMP
            and
            device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Parameters (M)       : "
        f"{total_parameters / 1e6:.4f}"
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    start_epoch = 0

    best_miou = 0.0

    if os.path.isfile(
        last_checkpoint
    ):

        try:

            (
                start_epoch,
                best_miou
            ) = load_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                scaler,
                device
            )

        except Exception as error:

            print()
            print(
                "WARNING: checkpoint could not be loaded."
            )

            print(
                error
            )

            print(
                "Starting fresh."
            )

            start_epoch = 0
            best_miou = 0.0

    else:

        print()
        print("=" * 70)
        print("STARTING NEW TRAINING")
        print("=" * 70)

    # --------------------------------------------------------
    # COMPLETE?
    # --------------------------------------------------------

    if start_epoch >= config.NUM_EPOCHS:

        print(
            "Training already complete."
        )

        return

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    current_epoch = start_epoch

    try:

        for epoch in range(
            start_epoch,
            config.NUM_EPOCHS
        ):

            current_epoch = epoch

            current_lr = set_epoch_lr(
                optimizer,
                epoch
            )

            print()
            print("=" * 70)

            print(
                f"EPOCH "
                f"{epoch + 1}/"
                f"{config.NUM_EPOCHS}"
            )

            print(
                f"Learning rate: "
                f"{current_lr:.8f}"
            )

            print("=" * 70)

            train_results = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch
            )

            if STOP_REQUESTED:

                break

            print_results(
                "TRAIN",
                train_results
            )

            val_results = validate(
                model,
                val_loader,
                criterion,
                device
            )

            print_results(
                "VALIDATION",
                val_results
            )

            current_miou = (
                val_results[
                    "metrics"
                ][
                    "miou"
                ].item()
            )

            # ------------------------------------------------
            # BEST
            # ------------------------------------------------

            if current_miou > best_miou:

                best_miou = current_miou

                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_miou
                )

                print()
                print(
                    "NEW BEST MODEL"
                )

                print(
                    f"Validation mIoU: "
                    f"{best_miou * 100:.2f}%"
                )

                print(
                    f"Saved:\n"
                    f"{best_checkpoint}"
                )

            # ------------------------------------------------
            # LAST
            # ------------------------------------------------

            save_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                scaler,
                epoch,
                best_miou
            )

            print()
            print(
                "LAST CHECKPOINT SAVED"
            )

            print(
                last_checkpoint
            )

    except KeyboardInterrupt:

        STOP_REQUESTED = True

        print()
        print("=" * 70)
        print("CTRL+C DETECTED")
        print("=" * 70)

        # Save using the previous completed epoch.
        completed_epoch = (
            current_epoch - 1
        )

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scaler,
            completed_epoch,
            best_miou
        )

        print(
            "Current state saved."
        )

        print(
            "Run python train.py again to resume."
        )

    finally:

        print()
        print("=" * 70)

        if STOP_REQUESTED:

            print(
                "TRAINING STOPPED SAFELY"
            )

        else:

            print(
                "TRAINING COMPLETE"
            )

        print("=" * 70)

        print(
            f"Best validation mIoU: "
            f"{best_miou * 100:.2f}%"
        )

        print(
            f"Best checkpoint:\n"
            f"{best_checkpoint}"
        )

        print(
            f"Last checkpoint:\n"
            f"{last_checkpoint}"
        )

        print("=" * 70)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()