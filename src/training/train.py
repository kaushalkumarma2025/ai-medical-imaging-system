import os
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder

from configs.config import *

from src.preprocessing.transforms import get_train_transforms
from src.training.dataset import get_dataloaders
from src.training.model import (
    build_model,
    count_trainable_params
)

# ======================================
# REPRODUCIBILITY
# ======================================

torch.manual_seed(SEED)

np.random.seed(SEED)

random.seed(SEED)

# ======================================
# DEVICE
# ======================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================================
# OUTPUT DIRECTORIES
# ======================================

METRICS_DIR = os.path.join(
    OUTPUT_DIR,
    "metrics"
)

VISUALIZATION_DIR = os.path.join(
    OUTPUT_DIR,
    "visualizations"
)

PREDICTIONS_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)

LOGS_DIR = os.path.join(
    OUTPUT_DIR,
    "logs"
)

# ======================================
# OUTPUT FILE PATHS
# ======================================

MODEL_SAVE_PATH = MODEL_PATH

HISTORY_SAVE_PATH = os.path.join(
    METRICS_DIR,
    "training_history.json"
)

# ======================================
# CLASS WEIGHTS
# ======================================

def compute_class_weights(dataset):

    """
    WHY THIS EXISTS:
    The dataset is imbalanced:
    more PNEUMONIA images than NORMAL images.

    Without weighting:
    the model may predict PNEUMONIA most of the time
    and still achieve misleadingly high accuracy.

    Inverse-frequency weighting penalizes minority
    class mistakes more heavily.
    """

    targets = [
        label for _, label in dataset.samples
    ]

    counts = np.bincount(targets)

    total = len(targets)

    weights = total / (
        len(counts) * counts
    )

    print(f"Class counts : {counts}")

    print(
        f"Loss weights : "
        f"NORMAL={weights[0]:.3f}, "
        f"PNEUMONIA={weights[1]:.3f}"
    )

    return torch.FloatTensor(
        weights
    ).to(DEVICE)

# ======================================
# TRAINING LOOP
# ======================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = outputs.max(1)

        correct += predicted.eq(
            labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# ======================================
# VALIDATION LOOP
# ======================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = outputs.max(1)

            correct += predicted.eq(
                labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# ======================================
# MAIN TRAINING PIPELINE
# ======================================

def train():

    # ----------------------------------
    # CREATE OUTPUT DIRECTORIES
    # ----------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        METRICS_DIR,
        exist_ok=True
    )

    os.makedirs(
        VISUALIZATION_DIR,
        exist_ok=True
    )

    os.makedirs(
        PREDICTIONS_DIR,
        exist_ok=True
    )

    os.makedirs(
        LOGS_DIR,
        exist_ok=True
    )

    # ----------------------------------
    # DEVICE INFO
    # ----------------------------------

    print(f"Device : {DEVICE}")

    # ----------------------------------
    # LOAD DATALOADERS
    # ----------------------------------

    train_loader, val_loader, _, classes = (
        get_dataloaders()
    )

    print(f"Classes: {classes}\n")

    # ----------------------------------
    # CLASS WEIGHTS
    # ----------------------------------

    train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=get_train_transforms()
    )

    class_weights = compute_class_weights(
        train_dataset
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ----------------------------------
    # BUILD MODEL
    # ----------------------------------

    model = build_model(
        num_classes=2,
        freeze_backbone=True
    ).to(DEVICE)

    print(
        f"Trainable params : "
        f"{count_trainable_params(model):,}\n"
    )

    # ----------------------------------
    # OPTIMIZER
    # ----------------------------------

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    # ----------------------------------
    # LEARNING RATE SCHEDULER
    # ----------------------------------

    scheduler = StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    # ----------------------------------
    # TRAINING HISTORY
    # ----------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    # ==================================
    # EPOCH LOOP
    # ==================================

    for epoch in range(EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        elapsed_time = (
            time.time() - start_time
        )

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  "
            f"Acc: {val_acc:.4f}  |  "
            f"{elapsed_time:.1f}s"
        )

        # ------------------------------
        # SAVE HISTORY
        # ------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_acc"].append(
            val_acc
        )

        # ------------------------------
        # SAVE BEST MODEL
        # ------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print(
                f"  ✓ Saved best model "
                f"(val_loss: {best_val_loss:.4f})"
            )

    # ==================================
    # SAVE TRAINING HISTORY
    # ==================================

    with open(
        HISTORY_SAVE_PATH,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=2
        )

    # ==================================
    # FINAL SUMMARY
    # ==================================

    print(
        f"\nDone. "
        f"Best val_loss: {best_val_loss:.4f}"
    )

    print(
        f"Model → {MODEL_SAVE_PATH}"
    )

    print(
        f"History → {HISTORY_SAVE_PATH}"
    )

# ======================================
# ENTRY POINT
# ======================================

if __name__ == "__main__":

    train()