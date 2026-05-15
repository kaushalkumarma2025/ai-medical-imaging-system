import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from src.preprocessing.transforms import get_train_transforms
from src.training.dataset import get_dataloaders
from src.training.model import build_model, count_trainable_params

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS     = 10
LEARNING_RATE  = 1e-3
MODEL_SAVE     = "models/best_model.pth"
HISTORY_SAVE   = "outputs/training_history.json"


def compute_class_weights(dataset):
    """
    WHY THIS EXISTS:
    Train set: ~3875 PNEUMONIA vs ~1341 NORMAL (roughly 3:1 imbalance)
    Without correction: model predicts PNEUMONIA always → 74% accuracy, clinically useless
    Inverse-frequency weighting makes the loss penalize
    minority class (NORMAL) misclassifications more heavily.
    """
    targets = [label for _, label in dataset.samples]
    counts  = np.bincount(targets)
    total   = len(targets)
    weights = total / (len(counts) * counts)
    print(f"Class counts : {counts}")
    print(f"Loss weights : NORMAL={weights[0]:.3f}, PNEUMONIA={weights[1]:.3f}")
    return torch.FloatTensor(weights).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)          # shape: (batch_size, 2)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct       += predicted.eq(labels).sum().item()
        total         += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

    return running_loss / total, correct / total


def train():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print(f"Device : {DEVICE}")

    train_loader, val_loader, _, classes = get_dataloaders()
    print(f"Classes: {classes}\n")

    # Class weights from train dataset
    train_dataset  = ImageFolder("data/chest_xray/train", transform=get_train_transforms())
    class_weights  = compute_class_weights(train_dataset)
    criterion      = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(num_classes=2, freeze_backbone=True).to(DEVICE)
    print(f"Trainable params : {count_trainable_params(model):,}\n")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # LR decays by half every 5 epochs
    # Prevents overshooting once model is close to convergence
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    history        = {"train_loss": [], "val_loss": [],
                      "train_acc":  [], "val_acc":  []}
    best_val_loss  = float("inf")

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = validate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")

    with open(HISTORY_SAVE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Model → {MODEL_SAVE}")
    print(f"History → {HISTORY_SAVE}")


if __name__ == "__main__":
    train()