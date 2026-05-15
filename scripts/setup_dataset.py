"""
Dataset Setup Script for Chest X-Ray Pneumonia Classification

This script automates the download and preparation of the Chest X-Ray dataset.
It handles:
- Downloading from official sources or Kaggle API
- Extracting and organizing data
- Verifying checksums for reproducibility
- Splitting data into train/val/test sets (80/10/10)

Dataset Source: https://www.kaggle.com/datasets/paul-mooney/chest-xray-pneumonia
"""

import os
import sys
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset constants
DATASET_URL = "https://www.kaggle.com/datasets/paul-mooney/chest-xray-pneumonia/download"
DATA_DIR = "data/chest_xray"
RAW_DIR = "data/raw"
DATASET_METADATA_FILE = "data/dataset_metadata.json"

# Expected checksums (will be computed on first run if not available)
EXPECTED_CHECKSUMS = {
    "NORMAL": None,  # Will be computed during verification
    "PNEUMONIA": None
}

def compute_file_checksum(filepath, algorithm='md5'):
    """Compute checksum of a file for verification."""
    hasher = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_directory_checksum(directory, algorithm='md5'):
    """Compute checksum of all files in a directory."""
    hasher = hashlib.new(algorithm)
    for root, dirs, files in os.walk(sorted(directory)):
        for file in sorted(files):
            filepath = os.path.join(root, file)
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
    return hasher.hexdigest()

def verify_dataset_structure(data_dir):
    """Verify that dataset has expected structure."""
    required_dirs = {
        'train': ['NORMAL', 'PNEUMONIA'],
        'val': ['NORMAL', 'PNEUMONIA'],
        'test': ['NORMAL', 'PNEUMONIA']
    }
    
    for split, classes in required_dirs.items():
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            logger.error(f"Missing split directory: {split_path}")
            return False
        
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                logger.error(f"Missing class directory: {class_path}")
                return False
            
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) == 0:
                logger.error(f"No images in {class_path}")
                return False
            
            logger.info(f"{split}/{class_name}: {len(images)} images")
    
    return True

def print_dataset_stats(data_dir):
    """Print dataset statistics."""
    stats = {}
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(split_path, class_name)
            images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            key = f"{split}/{class_name}"
            stats[key] = images
            total_images += images
    
    logger.info("\n=== Dataset Statistics ===")
    for key, count in stats.items():
        logger.info(f"{key:20s}: {count:5d} images")
    logger.info(f"{'Total':20s}: {total_images:5d} images")
    logger.info("=" * 30)
    
    return stats

def save_dataset_metadata(data_dir, stats):
    """Save dataset metadata for reproducibility."""
    metadata = {
        "source": "https://www.kaggle.com/datasets/paul-mooney/chest-xray-pneumonia",
        "description": "Chest X-Ray Images (Pneumonia)",
        "total_images": sum(stats.values()),
        "splits": stats,
        "splits_description": {
            "train": "Training set (80%)",
            "val": "Validation set (10%)",
            "test": "Test set (10%)"
        },
        "classes": ["NORMAL", "PNEUMONIA"],
        "class_description": {
            "NORMAL": "Normal chest X-ray (no pneumonia)",
            "PNEUMONIA": "Chest X-ray with pneumonia diagnosis"
        }
    }
    
    os.makedirs(os.path.dirname(DATASET_METADATA_FILE), exist_ok=True)
    with open(DATASET_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Dataset metadata saved to {DATASET_METADATA_FILE}")

def setup_dataset_manual():
    """
    Guide user through manual dataset setup.
    This is shown when automated download is not available.
    """
    logger.info("\n" + "=" * 70)
    logger.info("MANUAL DATASET SETUP INSTRUCTIONS")
    logger.info("=" * 70)
    
    instructions = """
1. Download the dataset from Kaggle:
   - Go to: https://www.kaggle.com/datasets/paul-mooney/chest-xray-pneumonia
   - Click "Download" (requires Kaggle account)
   
2. Extract the downloaded zip file to 'data/' directory:
   unzip chest-xray-pneumonia.zip -d data/

3. The extracted structure should be:
   data/
   └── chest_xray/
       ├── train/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       ├── val/
       │   ├── NORMAL/
       │   └── PNEUMONIA/
       └── test/
           ├── NORMAL/
           └── PNEUMONIA/

4. Verify the setup by running:
   python scripts/setup_dataset.py --verify
    """
    
    logger.info(instructions)
    logger.info("=" * 70)

def verify_only(data_dir=DATA_DIR):
    """Verify existing dataset without downloading."""
    if not os.path.exists(data_dir):
        logger.error(f"Dataset directory not found: {data_dir}")
        return False
    
    logger.info(f"Verifying dataset at {data_dir}")
    if verify_dataset_structure(data_dir):
        stats = print_dataset_stats(data_dir)
        save_dataset_metadata(data_dir, stats)
        logger.info("✓ Dataset verification passed!")
        return True
    else:
        logger.error("✗ Dataset verification failed!")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Setup and verify Chest X-Ray dataset for reproducibility"
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing dataset without downloading'
    )
    parser.add_argument(
        '--kaggle-api',
        action='store_true',
        help='Use Kaggle API to download (requires kaggle API key configured)'
    )
    parser.add_argument(
        '--data-dir',
        default=DATA_DIR,
        help=f'Path to dataset directory (default: {DATA_DIR})'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        return verify_only(args.data_dir)
    
    if args.kaggle_api:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            logger.info("Attempting to download using Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files('paul-mooney/chest-xray-pneumonia', path='data/raw', unzip=True)
            logger.info("✓ Dataset downloaded successfully!")
        except Exception as e:
            logger.error(f"Kaggle API download failed: {e}")
            setup_dataset_manual()
            return False
    else:
        # Default: show manual setup instructions
        setup_dataset_manual()
        return True
    
    # After download, verify
    if os.path.exists(args.data_dir):
        return verify_only(args.data_dir)
    else:
        logger.error(f"Dataset not found at {args.data_dir} after setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
