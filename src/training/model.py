import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_model()
    total     = sum(p.numel() for p in model.parameters())
    trainable = count_trainable_params(model)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {total - trainable:,}")
    print(f"FC layer         : {model.fc}")