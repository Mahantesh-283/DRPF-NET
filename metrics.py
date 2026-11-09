# ============================================================
# DRFP-Net V4 METRICS
#
# Semantic:
#   IoU
#   Precision
#   Recall
#   F1
#   Pixel Accuracy
#
# Boundary:
#   Boundary Precision
#   Boundary Recall
#   Boundary F1
# ============================================================

import torch
import torch.nn.functional as F


# ============================================================
# CONFUSION MATRIX
# ============================================================

def update_confusion_matrix(
    confusion_matrix,
    prediction,
    target,
    num_classes,
    ignore_index=255
):

    prediction = (
        prediction.reshape(-1)
    )

    target = (
        target.reshape(-1)
    )

    valid = (
        target != ignore_index
    )

    prediction = prediction[valid]

    target = target[valid]

    valid_prediction = (
        prediction >= 0
    ) & (
        prediction < num_classes
    )

    prediction = (
        prediction[valid_prediction]
    )

    target = (
        target[valid_prediction]
    )

    indices = (
        target * num_classes
        + prediction
    )

    counts = torch.bincount(
        indices,
        minlength=num_classes * num_classes
    )

    matrix = counts.reshape(
        num_classes,
        num_classes
    )

    return (
        confusion_matrix + matrix
    )


# ============================================================
# COMPUTE METRICS
# ============================================================

def compute_metrics(
    confusion_matrix
):

    cm = confusion_matrix.float()

    true_positive = (
        torch.diag(cm)
    )

    actual = (
        cm.sum(dim=1)
    )

    predicted = (
        cm.sum(dim=0)
    )

    false_positive = (
        predicted - true_positive
    )

    false_negative = (
        actual - true_positive
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    union = (
        true_positive
        +
        false_positive
        +
        false_negative
    )

    iou = (
        true_positive
        /
        union.clamp_min(1.0)
    )

    # Classes absent from ground truth and prediction are
    # ignored when calculating mean IoU.
    present = (
        union > 0
    )

    if present.any():

        miou = (
            iou[present].mean()
        )

    else:

        miou = torch.tensor(
            0.0,
            device=cm.device
        )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        true_positive
        /
        predicted.clamp_min(1.0)
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        true_positive
        /
        actual.clamp_min(1.0)
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        ).clamp_min(1e-8)
    )

    # --------------------------------------------------------
    # Pixel accuracy
    # --------------------------------------------------------

    total = cm.sum()

    pixel_accuracy = (
        true_positive.sum()
        /
        total.clamp_min(1.0)
    )

    # --------------------------------------------------------
    # Macro averages
    # --------------------------------------------------------

    macro_precision = (
        precision[present].mean()
        if present.any()
        else torch.tensor(
            0.0,
            device=cm.device
        )
    )

    macro_recall = (
        recall[present].mean()
        if present.any()
        else torch.tensor(
            0.0,
            device=cm.device
        )
    )

    macro_f1 = (
        f1[present].mean()
        if present.any()
        else torch.tensor(
            0.0,
            device=cm.device
        )
    )

    return {

        "iou":
            iou,

        "miou":
            miou,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "pixel_accuracy":
            pixel_accuracy,

        "confusion_matrix":
            confusion_matrix
    }


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================

def compute_iou(
    confusion_matrix
):

    results = compute_metrics(
        confusion_matrix
    )

    return (
        results["iou"],
        results["miou"],
        results["pixel_accuracy"]
    )


# ============================================================
# BOUNDARY MAP
# ============================================================

def mask_to_boundary(
    mask,
    ignore_index=255
):

    valid = (
        mask != ignore_index
    )

    safe = mask.clone()

    safe[~valid] = 0

    boundary = torch.zeros_like(
        safe,
        dtype=torch.bool
    )

    # vertical
    boundary[:, 1:] |= (
        (
            safe[:, 1:]
            !=
            safe[:, :-1]
        )
        &
        valid[:, 1:]
        &
        valid[:, :-1]
    )

    boundary[:, :-1] |= (
        (
            safe[:, :-1]
            !=
            safe[:, 1:]
        )
        &
        valid[:, :-1]
        &
        valid[:, 1:]
    )

    # horizontal
    boundary[:, :, 1:] |= (
        (
            safe[:, :, 1:]
            !=
            safe[:, :, :-1]
        )
        &
        valid[:, :, 1:]
        &
        valid[:, :, :-1]
    )

    boundary[:, :, :-1] |= (
        (
            safe[:, :, :-1]
            !=
            safe[:, :, 1:]
        )
        &
        valid[:, :, :-1]
        &
        valid[:, :, 1:]
    )

    return boundary


# ============================================================
# BOUNDARY METRICS
# ============================================================

def boundary_metrics(
    prediction,
    target,
    ignore_index=255,
    tolerance=2
):

    pred_boundary = mask_to_boundary(
        prediction,
        ignore_index
    )

    gt_boundary = mask_to_boundary(
        target,
        ignore_index
    )

    pred_boundary = (
        pred_boundary.float()
        .unsqueeze(1)
    )

    gt_boundary = (
        gt_boundary.float()
        .unsqueeze(1)
    )

    # --------------------------------------------------------
    # Allow a small spatial tolerance.
    #
    # This is important for satellite boundaries because a
    # prediction shifted by 1-2 pixels should not be treated
    # as a completely failed edge.
    # --------------------------------------------------------

    kernel_size = (
        tolerance * 2 + 1
    )

    gt_dilated = F.max_pool2d(
        gt_boundary,
        kernel_size=kernel_size,
        stride=1,
        padding=tolerance
    )

    pred_dilated = F.max_pool2d(
        pred_boundary,
        kernel_size=kernel_size,
        stride=1,
        padding=tolerance
    )

    pred_hits = (
        pred_boundary
        *
        gt_dilated
    ).sum()

    gt_hits = (
        gt_boundary
        *
        pred_dilated
    ).sum()

    pred_count = (
        pred_boundary.sum()
    )

    gt_count = (
        gt_boundary.sum()
    )

    precision = (
        pred_hits
        /
        pred_count.clamp_min(1.0)
    )

    recall = (
        gt_hits
        /
        gt_count.clamp_min(1.0)
    )

    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        ).clamp_min(1e-8)
    )

    return {

        "boundary_precision":
            precision,

        "boundary_recall":
            recall,

        "boundary_f1":
            f1,

        "pred_boundary_pixels":
            pred_count,

        "gt_boundary_pixels":
            gt_count
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    iou,
    miou,
    pixel_accuracy,
    precision=None,
    recall=None,
    f1=None
):

    print()

    print(
        f"mIoU          : "
        f"{miou.item() * 100:.2f}%"
    )

    print(
        f"Pixel Accuracy: "
        f"{pixel_accuracy.item() * 100:.2f}%"
    )

    if precision is not None:

        print(
            f"Macro Precision: "
            f"{precision.item() * 100:.2f}%"
        )

    if recall is not None:

        print(
            f"Macro Recall   : "
            f"{recall.item() * 100:.2f}%"
        )

    if f1 is not None:

        print(
            f"Macro F1       : "
            f"{f1.item() * 100:.2f}%"
        )

    print()

    print("Per-class IoU:")

    for index, value in enumerate(iou):

        print(
            f"  Class {index}: "
            f"{value.item() * 100:.2f}%"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    cm = torch.tensor(
        [
            [80, 10, 5],
            [8, 70, 12],
            [4, 10, 90]
        ]
    )

    results = compute_metrics(
        cm
    )

    print_metrics(
        results["iou"],
        results["miou"],
        results["pixel_accuracy"],
        results["macro_precision"],
        results["macro_recall"],
        results["macro_f1"]
    )