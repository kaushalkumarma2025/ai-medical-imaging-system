"""
Dataset Setup Script for Chest X-Ray Pneumonia Classification

Features:
- Verify dataset structure
- Download via Kaggle API (optional)
- Deterministic stratified rebalance to 80/10/10 (train/val/test)
- Save reproducibility metadata

Usage:
  python scripts/setup_dataset.py --verify
  python scripts/setup_dataset.py --rebalance-split --seed 42
  python scripts/setup_dataset.py --kaggle-api --rebalance-split
"""

import os
import sys
import json
import shutil
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

# --------------------------------------
# Logging
# --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------
# Constants
# --------------------------------------
DATA_DIR = "data/chest_xray"
DATASET_METADATA_FILE = "data/dataset_metadata.json"
CLASSES = ["NORMAL", "PNEUMONIA"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Kaggle dataset slug
KAGGLE_DATASET = "paul-mooney/chest-xray-pneumonia"


# --------------------------------------
# Helpers
# --------------------------------------
def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTS)


def split_dir(data_dir: str, split: str, class_name: str) -> str:
    return os.path.join(data_dir, split, class_name)


def ensure_split_structure(data_dir: str) -> None:
    for split in SPLITS:
        for class_name in CLASSES:
            os.makedirs(split_dir(data_dir, split, class_name), exist_ok=True)


def count_images_in_dir(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return sum(
        1 for f in os.listdir(path)
        if is_image_file(f) and os.path.isfile(os.path.join(path, f))
    )


def get_dataset_stats(data_dir: str) -> dict:
    stats = {}
    total = 0
    for split in SPLITS:
        split_total = 0
        for class_name in CLASSES:
            key = f"{split}/{class_name}"
            c = count_images_in_dir(split_dir(data_dir, split, class_name))
            stats[key] = c
            split_total += c
            total += c
        stats[f"{split}/TOTAL"] = split_total
    stats["TOTAL"] = total
    return stats


def print_dataset_stats(stats: dict) -> None:
    logger.info("\n=== Dataset Statistics ===")
    for split in SPLITS:
        logger.info(
            "%s -> NORMAL=%d, PNEUMONIA=%d, TOTAL=%d",
            split.upper(),
            stats[f"{split}/NORMAL"],
            stats[f"{split}/PNEUMONIA"],
            stats[f"{split}/TOTAL"],
        )
    logger.info("TOTAL IMAGES: %d", stats["TOTAL"])
    logger.info("=" * 32)


def verify_dataset_structure(data_dir: str, enforce_reasonable_val: bool = False) -> bool:
    # Existence + non-empty checks
    for split in SPLITS:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            logger.error("Missing split directory: %s", split_path)
            return False

        for class_name in CLASSES:
            class_path = split_dir(data_dir, split, class_name)
            if not os.path.exists(class_path):
                logger.error("Missing class directory: %s", class_path)
                return False

            image_count = count_images_in_dir(class_path)
            if image_count == 0:
                logger.error("No images found in: %s", class_path)
                return False

    stats = get_dataset_stats(data_dir)
    print_dataset_stats(stats)

    # Scientific sanity check (optional)
    if enforce_reasonable_val:
        val_total = stats["val/TOTAL"]
        total = stats["TOTAL"]
        val_ratio = (val_total / total) if total > 0 else 0.0

        # Expect roughly >=8% validation after rebalance
        if val_ratio < 0.08:
            logger.error(
                "Validation split too small: %d/%d (%.2f%%). "
                "Run with --rebalance-split.",
                val_total, total, val_ratio * 100
            )
            return False

    return True


def save_dataset_metadata(data_dir: str, stats: dict, rebalanced: bool, seed: int) -> None:
    metadata = {
        "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "description": "Chest X-Ray Images (Pneumonia)",
        "generated_at": datetime.now().isoformat(),
        "data_dir": data_dir,
        "classes": CLASSES,
        "splits": stats,
        "split_strategy": "stratified_80_10_10" if rebalanced else "original_dataset_split",
        "seed": seed,
        "notes": (
            "Original Kaggle split has very small validation set (16 total). "
            "Use --rebalance-split for reproducible scientific validation."
        ),
    }

    os.makedirs(os.path.dirname(DATASET_METADATA_FILE), exist_ok=True)
    with open(DATASET_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset metadata saved to %s", DATASET_METADATA_FILE)


def collect_all_images_by_class(data_dir: str) -> dict:
    """
    Collect images across train/val/test into per-class lists.
    Returns:
      {
        "NORMAL": [abs_path1, abs_path2, ...],
        "PNEUMONIA": [...]
      }
    """
    result = {c: [] for c in CLASSES}

    for split in SPLITS:
        for class_name in CLASSES:
            class_path = split_dir(data_dir, split, class_name)
            if not os.path.exists(class_path):
                continue

            for name in os.listdir(class_path):
                src = os.path.join(class_path, name)
                if os.path.isfile(src) and is_image_file(name):
                    result[class_name].append(src)

    return result


def stratified_split(paths: list, seed: int, train_ratio: float, val_ratio: float):
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # remainder to test
    n_test = n - n_train - n_val

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train:n_train + n_val]
    test_paths = shuffled[n_train + n_val:n_train + n_val + n_test]
    return train_paths, val_paths, test_paths


def unique_dest_name(dest_dir: str, original_name: str, prefix: str) -> str:
    """
    Avoid filename collisions across merged splits.
    """
    stem, ext = os.path.splitext(original_name)
    candidate = f"{prefix}_{original_name}"
    out = os.path.join(dest_dir, candidate)

    i = 1
    while os.path.exists(out):
        candidate = f"{prefix}_{stem}_{i}{ext}"
        out = os.path.join(dest_dir, candidate)
        i += 1

    return out


def rebalance_split(data_dir: str, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> bool:
    """
    Rebuild train/val/test as deterministic stratified 80/10/10 by class.
    Uses temp directory then atomic replace of split folders.
    """
    logger.info("Starting stratified rebalance (train=%.0f%%, val=%.0f%%, test=%.0f%%, seed=%d)",
                train_ratio * 100, val_ratio * 100, (1 - train_ratio - val_ratio) * 100, seed)

    by_class = collect_all_images_by_class(data_dir)

    for class_name in CLASSES:
        if len(by_class[class_name]) == 0:
            logger.error("No images found for class '%s'. Cannot rebalance.", class_name)
            return False

    tmp_root = os.path.join(data_dir, "_tmp_rebalanced")
    backup_root = os.path.join(data_dir, "_backup_before_rebalance")

    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    # Build temp split structure
    for split in SPLITS:
        for class_name in CLASSES:
            os.makedirs(os.path.join(tmp_root, split, class_name), exist_ok=True)

    # Copy files into tmp according to stratified split
    for class_name in CLASSES:
        train_paths, val_paths, test_paths = stratified_split(
            by_class[class_name], seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
        )

        mapping = [
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ]

        for split_name, src_paths in mapping:
            dest_dir = os.path.join(tmp_root, split_name, class_name)
            for idx, src in enumerate(src_paths):
                base = os.path.basename(src)
                # Prefix with index for deterministic, collision-safe naming
                dest = unique_dest_name(dest_dir, base, f"{idx:06d}")
                shutil.copy2(src, dest)

    # Backup old split dirs
    if os.path.exists(backup_root):
        shutil.rmtree(backup_root)
    os.makedirs(backup_root, exist_ok=True)

    for split in SPLITS:
        src_split = os.path.join(data_dir, split)
        if os.path.exists(src_split):
            shutil.move(src_split, os.path.join(backup_root, split))

    # Move new split dirs in place
    for split in SPLITS:
        shutil.move(os.path.join(tmp_root, split), os.path.join(data_dir, split))

    # Cleanup temp
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)

    logger.info("Rebalance complete. Backup saved at: %s", backup_root)

    # Verify result with val sanity enforcement
    return verify_dataset_structure(data_dir, enforce_reasonable_val=True)


def download_with_kaggle() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        logger.error("Kaggle package not available. Install with: pip install kaggle")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        os.makedirs("data/raw", exist_ok=True)
        logger.info("Downloading dataset using Kaggle API...")
        api.dataset_download_files(KAGGLE_DATASET, path="data/raw", unzip=True)
        logger.info("Download complete.")
        return True
    except Exception as exc:
        logger.error("Kaggle API download failed: %s", exc)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup, verify, and rebalance Chest X-Ray dataset"
    )
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset only")
    parser.add_argument("--kaggle-api", action="store_true", help="Download via Kaggle API")
    parser.add_argument("--rebalance-split", action="store_true", help="Rebuild split to stratified 80/10/10")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Dataset path (default: data/chest_xray)")

    args = parser.parse_args()
    data_dir = args.data_dir

    # Download path (optional)
    if args.kaggle_api:
        ok = download_with_kaggle()
        if not ok:
            return 1

    # Basic existence
    if not os.path.exists(data_dir):
        logger.error("Dataset directory not found: %s", data_dir)
        logger.info("Expected structure: data/chest_xray/train|val|test/{NORMAL,PNEUMONIA}")
        return 1

    # Rebalance first if requested
    rebalanced = False
    if args.rebalance_split:
        ok = rebalance_split(data_dir, seed=args.seed, train_ratio=0.8, val_ratio=0.1)
        if not ok:
            logger.error("Rebalance failed.")
            return 1
        rebalanced = True

    # Verify
    ok = verify_dataset_structure(data_dir, enforce_reasonable_val=rebalanced)
    if not ok:
        logger.error("Dataset verification failed.")
        return 1

    stats = get_dataset_stats(data_dir)
    save_dataset_metadata(data_dir, stats, rebalanced=rebalanced, seed=args.seed)
    logger.info("✓ Dataset setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())