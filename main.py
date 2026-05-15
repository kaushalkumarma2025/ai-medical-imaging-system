"""
Unified Training & Evaluation Orchestrator

Runs the complete ML pipeline in one command:
  1. Dataset setup (download/verify)
  2. Model training (ResNet18 with weighted loss)
  3. Model evaluation (test metrics, confusion matrix, ROC curve)
  4. Generate summary report

Usage:
  python main.py                    # Full pipeline
  python main.py --skip-dataset     # Skip dataset setup (use existing)
  python main.py --skip-train       # Skip training (use existing model)
  python main.py --evaluate-only    # Only evaluate existing model
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import OUTPUT_DIR, MODEL_PATH
from scripts.setup_dataset import verify_only, print_dataset_stats, save_dataset_metadata
from src.training.train import train
from src.training.evaluate import evaluate_model

# ======================================
# OUTPUT DIRECTORIES
# ======================================

REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

PIPELINE_REPORT_PATH = os.path.join(
    REPORT_DIR,
    f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

# ======================================
# PIPELINE FUNCTIONS
# ======================================

def print_header(section):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {section}")
    print("=" * 80 + "\n")

def step_1_setup_dataset():
    """Step 1: Verify/setup dataset."""
    print_header("STEP 1: DATASET SETUP & VERIFICATION")
    
    data_dir = "data/chest_xray"
    
    if os.path.exists(data_dir):
        print(f"✓ Dataset directory exists at {data_dir}")
        if verify_only(data_dir):
            print("✓ Dataset verification passed!")
            stats = {}
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(data_dir, split)
                for class_name in ['NORMAL', 'PNEUMONIA']:
                    class_path = os.path.join(split_path, class_name)
                    images = len([f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    stats[f"{split}/{class_name}"] = images
            save_dataset_metadata(data_dir, stats)
            return True
        else:
            print("✗ Dataset verification failed!")
            return False
    else:
        print(f"✗ Dataset not found at {data_dir}")
        print("Please run: python scripts/setup_dataset.py --verify")
        return False

def step_2_train_model():
    """Step 2: Train model."""
    print_header("STEP 2: MODEL TRAINING")
    
    if os.path.exists(MODEL_PATH):
        print(f"⚠ Model already exists at {MODEL_PATH}")
        response = input("Overwrite and retrain? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping training. Using existing model.")
            return True
    
    try:
        train()
        print(f"✓ Model training complete! Saved to {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

def step_3_evaluate_model():
    """Step 3: Evaluate model."""
    print_header("STEP 3: MODEL EVALUATION")
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found at {MODEL_PATH}")
        print("Please train a model first: python main.py --skip-dataset")
        return False
    
    try:
        results = evaluate_model()
        print("✓ Model evaluation complete!")
        return results
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return None

def generate_pipeline_report(dataset_ok, train_ok, eval_results):
    """Generate final pipeline report."""
    print_header("PIPELINE SUMMARY REPORT")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_status": "success" if (dataset_ok and train_ok and eval_results) else "partial",
        "steps": {
            "dataset_setup": "passed" if dataset_ok else "skipped/failed",
            "model_training": "passed" if train_ok else "skipped/failed",
            "model_evaluation": "passed" if eval_results else "skipped/failed"
        }
    }
    
    if eval_results:
        report["evaluation_results"] = {
            "test_accuracy": eval_results["test_accuracy"],
            "test_loss": eval_results["test_loss"],
            "metrics": eval_results["metrics"]
        }
    
    # Save report
    with open(PIPELINE_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {PIPELINE_REPORT_PATH}")
    
    # Print summary
    print("\nPipeline Summary:")
    print(f"  Dataset Setup  : {report['steps']['dataset_setup']}")
    print(f"  Model Training : {report['steps']['model_training']}")
    print(f"  Evaluation     : {report['steps']['model_evaluation']}")
    print(f"  Overall Status : {report['pipeline_status'].upper()}")
    
    if eval_results:
        print(f"\nModel Performance:")
        print(f"  Test Accuracy : {eval_results['test_accuracy']:.4f}")
        print(f"  Test Loss     : {eval_results['test_loss']:.4f}")
    
    print("\n" + "=" * 80)
    print("  All artifacts saved to: outputs/")
    print("  - metrics/: training_history.json, test_metrics.json, classification_report.json")
    print("  - visualizations/: confusion_matrix.png, roc_curve.png")
    print("  - reports/: pipeline_report_*.json")
    print("=" * 80 + "\n")

# ======================================
# MAIN PIPELINE
# ======================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified ML Pipeline: Dataset Setup → Train → Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full pipeline
  python main.py --skip-dataset     # Skip dataset setup
  python main.py --skip-train       # Skip training (use existing model)
  python main.py --evaluate-only    # Only evaluate existing model
        """
    )
    
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip dataset setup verification'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip model training (use existing model)'
    )
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only run evaluation on existing model'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  PNEUMONIA CLASSIFICATION ML PIPELINE")
    print("  Dataset → Training → Evaluation → Report")
    print("=" * 80)
    
    # Step 1: Dataset
    if args.evaluate_only:
        dataset_ok = True
        print("\n[SKIPPED] Dataset setup (--evaluate-only)")
    elif args.skip_dataset:
        dataset_ok = True
        print("\n[SKIPPED] Dataset setup (--skip-dataset)")
    else:
        dataset_ok = step_1_setup_dataset()
        if not dataset_ok:
            print("\n✗ Pipeline aborted: Dataset setup failed")
            return 1
    
    # Step 2: Train
    if args.evaluate_only:
        train_ok = True
        print("\n[SKIPPED] Model training (--evaluate-only)")
    elif args.skip_train:
        train_ok = True
        print("\n[SKIPPED] Model training (--skip-train)")
    else:
        train_ok = step_2_train_model()
        if not train_ok:
            print("\n✗ Pipeline aborted: Training failed")
            return 1
    
    # Step 3: Evaluate
    eval_results = step_3_evaluate_model()
    if not eval_results:
        print("\n✗ Pipeline aborted: Evaluation failed")
        return 1
    
    # Generate report
    generate_pipeline_report(dataset_ok, train_ok, eval_results)
    
    print("✓ Pipeline complete!\n")
    return 0

# ======================================
# ENTRY POINT
# ======================================

if __name__ == "__main__":
    sys.exit(main())