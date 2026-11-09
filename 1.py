import torch
from model import DRFPNet

model = DRFPNet(num_classes=7)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 50)
print("MODEL PARAMETERS")
print("=" * 50)
print(f"Total parameters     : {total:,}")
print(f"Trainable parameters : {trainable:,}")
print(f"Parameters (M)       : {total / 1e6:.4f}M")
print("=" * 50)