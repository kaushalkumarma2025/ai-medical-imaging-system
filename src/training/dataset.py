import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.preprocessing.transforms import get_train_transforms, get_val_transforms

DATA_ROOT   = "data/chest_xray"
BATCH_SIZE  = 32
NUM_WORKERS = 0  # keep 0 on Windows


def get_dataloaders():
    train_dataset = ImageFolder(
        root=os.path.join(DATA_ROOT, "train"),
        transform=get_train_transforms()
    )
    val_dataset = ImageFolder(
        root=os.path.join(DATA_ROOT, "val"),
        transform=get_val_transforms()
    )
    test_dataset = ImageFolder(
        root=os.path.join(DATA_ROOT, "test"),
        transform=get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train : {len(train_dataset)} images")
    print(f"Val   : {len(val_dataset)} images")
    print(f"Test  : {len(test_dataset)} images")
    print(f"Classes → {train_dataset.class_to_idx}")

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    get_dataloaders()