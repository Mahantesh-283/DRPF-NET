# ============================================================
# DRFP-Net V2 COMPLETE TEST
# ============================================================

import torch

from model import (
    DRFPNet,
    model_statistics,
    approximate_macs
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("DRFP-Net V2 MODEL TEST")
print("=" * 70)

print(
    "Device:",
    device
)


# ============================================================
# MODEL
# ============================================================

model = DRFPNet(
    num_classes=7,
    feature_channels=96,
    context_channels=64
).to(device)


# ============================================================
# EVAL
# ============================================================

model.eval()


# ============================================================
# INPUT
# ============================================================

x = torch.randn(
    1,
    3,
    512,
    512,
    device=device
)


# ============================================================
# FORWARD
# ============================================================

with torch.no_grad():

    output = model(
        x
    )


# ============================================================
# SHAPES
# ============================================================

print()
print(
    "Input:",
    x.shape
)

print(
    "Logits:",
    output["logits"].shape
)

print(
    "Importance:",
    output["importance"].shape
)

print(
    "Edge:",
    output["edge_energy"].shape
)

print(
    "Variation:",
    output["local_variation"].shape
)

print(
    "Texture:",
    output["texture"].shape
)


# ============================================================
# ASSERTIONS
# ============================================================

assert output["logits"].shape == (
    1,
    7,
    512,
    512
)

assert output["importance"].shape == (
    1,
    1,
    128,
    128
)

assert output["edge_energy"].shape == (
    1,
    1,
    128,
    128
)


# ============================================================
# IMPORTANCE RANGE
# ============================================================

importance = output[
    "importance"
]

assert importance.min() >= 0

assert importance.max() <= 1


# ============================================================
# PARAMETERS
# ============================================================

model_statistics(
    model
)


# ============================================================
# MACS
# ============================================================

macs = approximate_macs(
    model,
    x
)

print()
print("=" * 70)
print("COMPUTATIONAL COST")
print("=" * 70)

print(
    f"MACs : "
    f"{macs / 1e9:.4f} G"
)

print(
    f"FLOPs: "
    f"{2 * macs / 1e9:.4f} G"
)

print("=" * 70)


# ============================================================
# BACKWARD TEST
# ============================================================

print()
print("=" * 70)
print("BACKWARD TEST")
print("=" * 70)

model.train()

x = torch.randn(
    1,
    3,
    512,
    512,
    device=device,
    requires_grad=True
)

output = model(
    x
)

target = torch.randint(
    0,
    7,
    (
        1,
        512,
        512
    ),
    device=device
)

loss = torch.nn.functional.cross_entropy(
    output["logits"],
    target
)

print(
    "Loss:",
    loss.item()
)

loss.backward()

assert x.grad is not None

gradient_count = 0

for parameter in model.parameters():

    if parameter.grad is not None:

        gradient_count += 1

assert gradient_count > 0

print(
    "Parameters with gradients:",
    gradient_count
)

print()
print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)