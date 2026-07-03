"""
module6_traffic_sign/train_sign_model.py
=========================================
Train a small CNN on the GTSRB (German Traffic Sign Recognition Benchmark)
dataset to classify selected sign categories used by the ADAS system.

Selected GTSRB class IDs → ADAS label
--------------------------------------
    1  → SPEED LIMIT 30
    2  → SPEED LIMIT 50
    5  → SPEED LIMIT 80
   14  → STOP
   34  → TURN LEFT
   35  → TURN RIGHT

Requirements (already installed with ultralytics):
    pip install torch torchvision

Dataset:
    Downloaded automatically via torchvision.datasets.GTSRB.
    Stored in: ./data/GTSRB/

Output:
    module6_traffic_sign/sign_model.pth

Training:
    python module6_traffic_sign/train_sign_model.py

Inference time: < 2 ms per crop on CPU (32×32 input)
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SELECTED_GTSRB_IDS = {1: 0, 2: 1, 5: 2, 14: 3, 34: 4, 35: 5}
ID_TO_LABEL        = {
    0: "SPEED LIMIT 30",
    1: "SPEED LIMIT 50",
    2: "SPEED LIMIT 80",
    3: "STOP",
    4: "TURN LEFT",
    5: "TURN RIGHT",
}
NUM_CLASSES   = len(SELECTED_GTSRB_IDS)
IMG_SIZE      = 32
BATCH_SIZE    = 64
EPOCHS        = 8
LR            = 1e-3
DATA_DIR      = "./data"
MODEL_SAVE    = os.path.join(os.path.dirname(__file__), "sign_model.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight CNN — ~150 K parameters
# ─────────────────────────────────────────────────────────────────────────────
class SignCNN(nn.Module):
    """
    Small 3-layer CNN for 32×32 traffic-sign classification.
    Fast enough for real-time ROI classification on CPU.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 32×32 → 16×16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2 — 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3 — 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_transform(augment: bool = False):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        ops += [
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomHorizontalFlip(p=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def filter_dataset(dataset, selected_ids: dict):
    """
    Keep only samples whose original GTSRB class ID is in selected_ids,
    and remap the label to a contiguous 0-based index.
    """
    indices     = []
    new_targets = []

    for idx, (_, target) in enumerate(dataset):
        if target in selected_ids:
            indices.append(idx)
            new_targets.append(selected_ids[target])

    subset = Subset(dataset, indices)

    # Patch __getitem__ so it returns the remapped label
    original_getitem = subset.__getitem__

    def patched_getitem(i):
        img, _ = original_getitem(i)
        return img, new_targets[i]

    subset.__getitem__ = patched_getitem
    return subset, len(indices)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TrainSign] Device: {device}")
    print(f"[TrainSign] Classes: {NUM_CLASSES}  →  {list(ID_TO_LABEL.values())}")
    print(f"[TrainSign] Epochs : {EPOCHS}   Batch: {BATCH_SIZE}\n")

    # ── Download / load GTSRB ─────────────────────────────────────────
    print("[TrainSign] Downloading GTSRB (first run only)…")
    raw_train = datasets.GTSRB(root=DATA_DIR, split="train",
                                download=True,
                                transform=get_transform(augment=True))
    raw_val   = datasets.GTSRB(root=DATA_DIR, split="test",
                                download=True,
                                transform=get_transform(augment=False))

    # ── Filter to selected classes ────────────────────────────────────
    train_set, n_train = filter_dataset(raw_train, SELECTED_GTSRB_IDS)
    val_set,   n_val   = filter_dataset(raw_val,   SELECTED_GTSRB_IDS)
    print(f"[TrainSign] Train samples: {n_train} | Val samples: {n_val}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Model, loss, optimizer ────────────────────────────────────────
    model     = SignCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        t0         = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        correct = total_v = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds   = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total_v += labels.size(0)

        acc = correct / max(total_v, 1) * 100
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:2d}/{EPOCHS}  |  "
              f"Loss: {total_loss / len(train_loader):.4f}  |  "
              f"Val Acc: {acc:.1f}%  |  "
              f"Time: {elapsed:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "id_to_label": ID_TO_LABEL,
                "num_classes" : NUM_CLASSES,
                "img_size"    : IMG_SIZE,
            }, MODEL_SAVE)
            print(f"  ✓ Saved best model → {MODEL_SAVE}")

    print(f"\n[TrainSign] Training complete. Best val accuracy: {best_acc:.1f}%")
    print(f"[TrainSign] Model saved: {MODEL_SAVE}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
