# ============================================================
# DRFP-Net V4
#
# Boundary-Aware Multi-Kernel CNN
#
# Main ideas:
#   - Variable kernel sizes: 3x3, 5x5, 7x7
#   - Dilated convolution for larger receptive field
#   - High-resolution boundary branch
#   - Boundary-guided semantic refinement
#   - Lightweight decoder
#
# Input:
#   B x 3 x H x W
#
# Output:
#   {
#       "logits":          B x C x H x W
#       "boundary_logits": B x 1 x H x W
#   }
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASIC CONV
# ============================================================

class ConvBNAct(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1
    ):

        super().__init__()

        if padding is None:
            padding = (
                kernel_size // 2
            ) * dilation

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# DEPTHWISE SEPARABLE CONV
# ============================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1
    ):

        super().__init__()

        padding = (
            kernel_size // 2
        ) * dilation

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            in_channels
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)

        return x


# ============================================================
# MULTI-KERNEL BLOCK
# ============================================================
#
# Four receptive fields:
#
#   3x3
#   5x5
#   7x7
#   3x3 dilation=2
#
# This is the important part for your requirement of
# variable-size kernels.
# ============================================================

class MultiKernelBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        branch_channels = (
            out_channels // 4
        )

        remainder = (
            out_channels
            - branch_channels * 4
        )

        channels = [
            branch_channels,
            branch_channels,
            branch_channels,
            branch_channels
        ]

        channels[-1] += remainder

        self.branch3 = DepthwiseSeparableConv(
            in_channels,
            channels[0],
            kernel_size=3
        )

        self.branch5 = DepthwiseSeparableConv(
            in_channels,
            channels[1],
            kernel_size=5
        )

        self.branch7 = DepthwiseSeparableConv(
            in_channels,
            channels[2],
            kernel_size=7
        )

        self.branch_dilated = DepthwiseSeparableConv(
            in_channels,
            channels[3],
            kernel_size=3,
            dilation=2
        )

        self.fusion = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )

        self.residual = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.res_bn = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        b3 = self.branch3(x)

        b5 = self.branch5(x)

        b7 = self.branch7(x)

        bd = self.branch_dilated(x)

        features = torch.cat(
            [
                b3,
                b5,
                b7,
                bd
            ],
            dim=1
        )

        features = self.fusion(
            features
        )

        residual = self.res_bn(
            self.residual(x)
        )

        return self.act(
            features + residual
        )


# ============================================================
# ENCODER BLOCK
# ============================================================

class EncoderStage(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.downsample = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2
        )

        self.multi_kernel = MultiKernelBlock(
            out_channels,
            out_channels
        )

    def forward(self, x):

        x = self.downsample(x)

        x = self.multi_kernel(x)

        return x


# ============================================================
# BOUNDARY FEATURE BLOCK
# ============================================================

class BoundaryFeatureBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        feature_channels
    ):

        super().__init__()

        self.conv1 = ConvBNAct(
            in_channels,
            feature_channels,
            kernel_size=3
        )

        self.conv2 = ConvBNAct(
            feature_channels,
            feature_channels,
            kernel_size=5
        )

        self.conv3 = ConvBNAct(
            feature_channels,
            feature_channels,
            kernel_size=3,
            dilation=2
        )

        self.refine = ConvBNAct(
            feature_channels,
            feature_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.refine(x)

        return x


# ============================================================
# BOUNDARY HEAD
# ============================================================

class BoundaryHead(nn.Module):

    def __init__(
        self,
        in_channels
    ):

        super().__init__()

        self.block = BoundaryFeatureBlock(
            in_channels,
            in_channels
        )

        self.head = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

    def forward(self, x):

        features = self.block(x)

        logits = self.head(
            features
        )

        return (
            logits,
            features
        )


# ============================================================
# DECODER
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.conv = MultiKernelBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [
                x,
                skip
            ],
            dim=1
        )

        return self.conv(x)


# ============================================================
# DRFP-NET V4
# ============================================================

class DRFPNet(nn.Module):

    def __init__(
        self,
        num_classes=7,
        feature_channels=96
    ):

        super().__init__()

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        c1 = feature_channels

        c2 = feature_channels * 2

        c3 = feature_channels * 4

        c4 = feature_channels * 6

        self.stem = nn.Sequential(

            ConvBNAct(
                3,
                c1,
                kernel_size=3
            ),

            MultiKernelBlock(
                c1,
                c1
            )
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder2 = EncoderStage(
            c1,
            c2
        )

        self.encoder3 = EncoderStage(
            c2,
            c3
        )

        self.encoder4 = EncoderStage(
            c3,
            c4
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = nn.Sequential(

            MultiKernelBlock(
                c4,
                c4
            ),

            MultiKernelBlock(
                c4,
                c4
            )
        )

        # ----------------------------------------------------
        # Boundary branch
        #
        # High-resolution features are used because boundary
        # localization should not depend only on deep,
        # low-resolution semantic features.
        # ----------------------------------------------------

        self.boundary_head = BoundaryHead(
            c1
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder3 = DecoderBlock(
            c4,
            c3,
            c3
        )

        self.decoder2 = DecoderBlock(
            c3,
            c2,
            c2
        )

        self.decoder1 = DecoderBlock(
            c2,
            c1,
            c1
        )

        # ----------------------------------------------------
        # Boundary-guided semantic fusion
        # ----------------------------------------------------

        self.boundary_projection = nn.Sequential(

            nn.Conv2d(
                c1,
                c1,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                c1
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.semantic_refine = MultiKernelBlock(
            c1 * 2,
            c1
        )

        # ----------------------------------------------------
        # Semantic head
        # ----------------------------------------------------

        self.semantic_head = nn.Sequential(

            ConvBNAct(
                c1,
                c1,
                kernel_size=3
            ),

            nn.Conv2d(
                c1,
                num_classes,
                kernel_size=1
            )
        )

        # ----------------------------------------------------
        # Initialize
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        input_size = x.shape[-2:]

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        s1 = self.stem(x)

        s2 = self.encoder2(s1)

        s3 = self.encoder3(s2)

        s4 = self.encoder4(s3)

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        b = self.bottleneck(s4)

        # ----------------------------------------------------
        # Boundary branch
        # ----------------------------------------------------

        boundary_logits, boundary_features = (
            self.boundary_head(s1)
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        d3 = self.decoder3(
            b,
            s3
        )

        d2 = self.decoder2(
            d3,
            s2
        )

        d1 = self.decoder1(
            d2,
            s1
        )

        # ----------------------------------------------------
        # Boundary-guided fusion
        # ----------------------------------------------------

        boundary_features = (
            self.boundary_projection(
                boundary_features
            )
        )

        semantic_features = torch.cat(
            [
                d1,
                boundary_features
            ],
            dim=1
        )

        semantic_features = (
            self.semantic_refine(
                semantic_features
            )
        )

        # ----------------------------------------------------
        # Semantic prediction
        # ----------------------------------------------------

        logits = self.semantic_head(
            semantic_features
        )

        # ----------------------------------------------------
        # Guarantee original image resolution
        # ----------------------------------------------------

        if logits.shape[-2:] != input_size:

            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        if boundary_logits.shape[-2:] != input_size:

            boundary_logits = F.interpolate(
                boundary_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        return {

            "logits":
                logits,

            "boundary_logits":
                boundary_logits,

            "boundary_features":
                boundary_features,

            "semantic_features":
                semantic_features
        }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:

                    nn.init.constant_(
                        module.bias,
                        0
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d
            ):

                nn.init.constant_(
                    module.weight,
                    1
                )

                nn.init.constant_(
                    module.bias,
                    0
                )


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DRFP-Net V4 MODEL TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = DRFPNet(
        num_classes=7,
        feature_channels=96
    ).to(device)

    x = torch.randn(
        2,
        3,
        512,
        512,
        device=device
    )

    with torch.no_grad():

        output = model(x)

    print(
        "Input:",
        tuple(x.shape)
    )

    print(
        "Semantic logits:",
        tuple(
            output["logits"].shape
        )
    )

    print(
        "Boundary logits:",
        tuple(
            output["boundary_logits"].shape
        )
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Total parameters     : {parameters:,}"
    )

    print(
        f"Trainable parameters : {trainable:,}"
    )

    print(
        f"Parameters (M)       : "
        f"{parameters / 1e6:.4f}"
    )

    print()
    print("=" * 70)
    print("MODEL TEST PASSED")
    print("=" * 70)