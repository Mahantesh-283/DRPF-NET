# ============================================================
# DRFP-Net V4 LOSS
#
# Semantic:
#   Cross Entropy
#   Dice
#
# Boundary:
#   BCE
#   Dice
#
# Total:
#   semantic_weight * semantic_loss
#   +
#   boundary_weight * boundary_loss
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BOUNDARY TARGET
# ============================================================

def create_boundary_target(
    target,
    ignore_index=255
):
    """
    Creates a binary boundary map.

    A pixel becomes boundary=1 when its semantic class
    differs from one of its 4-neighbour pixels.

    Ignore pixels are excluded.
    """

    valid = (
        target != ignore_index
    )

    safe = target.clone()

    safe[~valid] = 0

    boundary = torch.zeros_like(
        safe,
        dtype=torch.bool
    )

    # --------------------------------------------------------
    # Up
    # --------------------------------------------------------

    diff_up = torch.zeros_like(
        boundary
    )

    diff_up[:, 1:] = (
        safe[:, 1:] != safe[:, :-1]
    )

    valid_up = torch.zeros_like(
        boundary
    )

    valid_up[:, 1:] = (
        valid[:, 1:]
        &
        valid[:, :-1]
    )

    boundary |= (
        diff_up
        &
        valid_up
    )

    # --------------------------------------------------------
    # Down
    # --------------------------------------------------------

    diff_down = torch.zeros_like(
        boundary
    )

    diff_down[:, :-1] = (
        safe[:, :-1] != safe[:, 1:]
    )

    valid_down = torch.zeros_like(
        boundary
    )

    valid_down[:, :-1] = (
        valid[:, :-1]
        &
        valid[:, 1:]
    )

    boundary |= (
        diff_down
        &
        valid_down
    )

    # --------------------------------------------------------
    # Left
    # --------------------------------------------------------

    diff_left = torch.zeros_like(
        boundary
    )

    diff_left[:, :, 1:] = (
        safe[:, :, 1:]
        !=
        safe[:, :, :-1]
    )

    valid_left = torch.zeros_like(
        boundary
    )

    valid_left[:, :, 1:] = (
        valid[:, :, 1:]
        &
        valid[:, :, :-1]
    )

    boundary |= (
        diff_left
        &
        valid_left
    )

    # --------------------------------------------------------
    # Right
    # --------------------------------------------------------

    diff_right = torch.zeros_like(
        boundary
    )

    diff_right[:, :, :-1] = (
        safe[:, :, :-1]
        !=
        safe[:, :, 1:]
    )

    valid_right = torch.zeros_like(
        boundary
    )

    valid_right[:, :, :-1] = (
        valid[:, :, :-1]
        &
        valid[:, :, 1:]
    )

    boundary |= (
        diff_right
        &
        valid_right
    )

    boundary = (
        boundary
        &
        valid
    )

    return boundary.float()


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=7,
        ignore_index=255,
        smooth=1.0
    ):

        super().__init__()

        self.num_classes = num_classes

        self.ignore_index = ignore_index

        self.smooth = smooth

    def forward(
        self,
        logits,
        target
    ):

        probs = F.softmax(
            logits,
            dim=1
        )

        valid = (
            target != self.ignore_index
        )

        safe_target = target.clone()

        safe_target[~valid] = 0

        one_hot = F.one_hot(
            safe_target,
            num_classes=self.num_classes
        )

        one_hot = one_hot.permute(
            0,
            3,
            1,
            2
        ).float()

        valid = valid.unsqueeze(1)

        probs = probs * valid

        one_hot = one_hot * valid

        intersection = (
            probs * one_hot
        ).sum(
            dim=(0, 2, 3)
        )

        denominator = (
            probs + one_hot
        ).sum(
            dim=(0, 2, 3)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return (
            1.0 - dice.mean()
        )


# ============================================================
# BINARY DICE LOSS
# ============================================================

class BinaryDiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        target,
        valid=None
    ):

        probability = torch.sigmoid(
            logits
        )

        target = target.float()

        if valid is not None:

            probability = (
                probability * valid
            )

            target = (
                target * valid
            )

        intersection = (
            probability * target
        ).sum()

        denominator = (
            probability + target
        ).sum()

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return (
            1.0 - dice
        )


# ============================================================
# DRFP V4 LOSS
# ============================================================

class DRFPLoss(nn.Module):

    def __init__(
        self,
        num_classes=7,
        ignore_index=255,

        ce_weight=0.50,
        dice_weight=0.20,

        boundary_bce_weight=0.15,
        boundary_dice_weight=0.15
    ):

        super().__init__()

        self.num_classes = (
            num_classes
        )

        self.ignore_index = (
            ignore_index
        )

        self.ce_weight = (
            ce_weight
        )

        self.dice_weight = (
            dice_weight
        )

        self.boundary_bce_weight = (
            boundary_bce_weight
        )

        self.boundary_dice_weight = (
            boundary_dice_weight
        )

        # ----------------------------------------------------
        # Semantic class weights
        # ----------------------------------------------------

        class_weights = torch.tensor(
            [
                0.75,
                1.00,
                1.35,
                1.30,
                1.35,
                0.90,
                0.85
            ],
            dtype=torch.float32
        )

        self.register_buffer(
            "class_weights",
            class_weights
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index
        )

        self.boundary_dice = (
            BinaryDiceLoss()
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        logits,
        target,
        boundary_logits=None
    ):

        class_weights = (
            self.class_weights.to(
                device=logits.device,
                dtype=logits.dtype
            )
        )

        # ----------------------------------------------------
        # Semantic CE
        # ----------------------------------------------------

        ce = F.cross_entropy(
            logits,
            target,
            weight=class_weights,
            ignore_index=self.ignore_index
        )

        # ----------------------------------------------------
        # Semantic Dice
        # ----------------------------------------------------

        dice = self.dice(
            logits,
            target
        )

        semantic_loss = (
            self.ce_weight * ce
            +
            self.dice_weight * dice
        )

        # ----------------------------------------------------
        # Boundary loss
        # ----------------------------------------------------

        boundary_bce = (
            logits.new_tensor(0.0)
        )

        boundary_dice = (
            logits.new_tensor(0.0)
        )

        boundary_target = None

        if boundary_logits is not None:

            boundary_target = (
                create_boundary_target(
                    target,
                    self.ignore_index
                )
            )

            valid = (
                target != self.ignore_index
            )

            valid = valid.unsqueeze(1).float()

            # ------------------------------------------------
            # Dynamic positive weighting
            #
            # Boundaries are sparse, so ordinary BCE would
            # mostly learn "everything is background".
            # ------------------------------------------------

            positive = (
                boundary_target * valid
            ).sum()

            negative = (
                (1.0 - boundary_target)
                * valid
            ).sum()

            pos_weight = (
                negative /
                positive.clamp_min(1.0)
            )

            pos_weight = pos_weight.clamp(
                min=1.0,
                max=20.0
            )

            boundary_bce = (
                F.binary_cross_entropy_with_logits(
                    boundary_logits,
                    boundary_target.unsqueeze(1),
                    pos_weight=pos_weight
                )
            )

            boundary_dice = (
                self.boundary_dice(
                    boundary_logits,
                    boundary_target.unsqueeze(1),
                    valid
                )
            )

        total = (
            semantic_loss
            +
            self.boundary_bce_weight
            * boundary_bce
            +
            self.boundary_dice_weight
            * boundary_dice
        )

        return {

            "loss":
                total,

            "ce":
                ce.detach(),

            "dice":
                dice.detach(),

            "boundary_bce":
                boundary_bce.detach(),

            "boundary_dice":
                boundary_dice.detach(),

            "boundary_target":
                boundary_target
        }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

CombinedSegmentationLoss = DRFPLoss


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DRFP-Net V4 LOSS TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logits = torch.randn(
        2,
        7,
        128,
        128,
        device=device,
        requires_grad=True
    )

    boundary_logits = torch.randn(
        2,
        1,
        128,
        128,
        device=device,
        requires_grad=True
    )

    target = torch.randint(
        0,
        7,
        (
            2,
            128,
            128
        ),
        device=device
    )

    target[
        :,
        :10,
        :10
    ] = 255

    criterion = DRFPLoss(
        num_classes=7,
        ignore_index=255
    ).to(device)

    results = criterion(
        logits,
        target,
        boundary_logits
    )

    print(
        f"Total loss       : "
        f"{results['loss'].item():.6f}"
    )

    print(
        f"CE               : "
        f"{results['ce'].item():.6f}"
    )

    print(
        f"Dice             : "
        f"{results['dice'].item():.6f}"
    )

    print(
        f"Boundary BCE     : "
        f"{results['boundary_bce'].item():.6f}"
    )

    print(
        f"Boundary Dice    : "
        f"{results['boundary_dice'].item():.6f}"
    )

    results["loss"].backward()

    print(
        "Gradient exists  :",
        logits.grad is not None
    )

    print("=" * 70)
    print("LOSS TEST PASSED")
    print("=" * 70)