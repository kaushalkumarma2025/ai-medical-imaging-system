"""
Evaluation Pipeline for Chest X-Ray Pneumonia Classification

This module provides comprehensive evaluation metrics including:
- Test accuracy and loss
- Confusion matrix
- Precision, recall, F1-score
- ROC-AUC curve
- Per-class metrics
- Detailed classification report

Outputs:
- outputs/metrics/test_metrics.json
- outputs/metrics/classification_report.json
- outputs/visualizations/confusion_matrix.png
- outputs/visualizations/roc_curve.png
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_fscore_support
)

from configs.config import CLASS_NAMES, MODEL_PATH, OUTPUT_DIR
from src.training.model import build_model
from src.training.dataset import get_dataloaders

# ======================================
# DEVICE
# ======================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================================
# OUTPUT DIRECTORIES
# ======================================

METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# ======================================
# OUTPUT FILE PATHS
# ======================================

TEST_METRICS_PATH = os.path.join(
    METRICS_DIR,
    "test_metrics.json"
)

CLASSIFICATION_REPORT_PATH = os.path.join(
    METRICS_DIR,
    "classification_report.json"
)

CONFUSION_MATRIX_PATH = os.path.join(
    VISUALIZATION_DIR,
    "confusion_matrix.png"
)

ROC_CURVE_PATH = os.path.join(
    VISUALIZATION_DIR,
    "roc_curve.png"
)

# ======================================
# EVALUATION FUNCTIONS
# ======================================

def evaluate(model, test_loader, criterion):
    """
    Evaluate model on test set.
    
    Returns:
    --------
    dict: Contains loss, accuracy, and all predictions/labels
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(1)
            
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (PNEUMONIA)
    
    test_loss = running_loss / total
    test_accuracy = correct / total
    
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probabilities)
    }

def compute_metrics(predictions, labels, probabilities):
    """
    Compute comprehensive metrics from predictions and labels.
    
    Returns:
    --------
    dict: Comprehensive metrics including per-class and overall metrics
    """
    cm = confusion_matrix(labels, predictions)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Overall metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    # ROC-AUC
    roc_auc = roc_auc_score(labels, probabilities)
    
    # Sensitivity (Recall for PNEUMONIA) and Specificity (Recall for NORMAL)
    sensitivity = recall[1]  # TPR for PNEUMONIA (positive class)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        "overall": {
            "test_accuracy": float(np.mean(predictions == labels)),
            "roc_auc": float(roc_auc),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1)
        },
        "per_class": {
            CLASS_NAMES[0]: {  # NORMAL
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1_score": float(f1[0]),
                "support": int(support[0])
            },
            CLASS_NAMES[1]: {  # PNEUMONIA
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1_score": float(f1[1]),
                "support": int(support[1])
            }
        },
        "clinical_metrics": {
            "sensitivity": float(sensitivity),  # True positive rate
            "specificity": float(specificity),  # True negative rate
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
    }
    
    return metrics, cm

def plot_confusion_matrix(cm, output_path):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {output_path}")

def plot_roc_curve(labels, probabilities, output_path):
    """
    Plot and save ROC curve.
    """
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve saved to {output_path}")

def print_metrics_summary(metrics):
    """
    Print nicely formatted metrics summary.
    """
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 70)
    
    overall = metrics['overall']
    clinical = metrics['clinical_metrics']
    per_class = metrics['per_class']
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy   : {overall['test_accuracy']:.4f}")
    print(f"  ROC-AUC    : {overall['roc_auc']:.4f}")
    print(f"  Macro F1   : {overall['macro_f1']:.4f}")
    print(f"  Weighted F1: {overall['weighted_f1']:.4f}")
    
    print(f"\nClinical Metrics (important for medical diagnosis):")
    print(f"  Sensitivity (TPR) : {clinical['sensitivity']:.4f}  (catches PNEUMONIA cases)")
    print(f"  Specificity (TNR) : {clinical['specificity']:.4f}  (correctly identifies NORMAL)")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives  : {clinical['true_positives']} (correctly predicted PNEUMONIA)")
    print(f"  True Negatives  : {clinical['true_negatives']} (correctly predicted NORMAL)")
    print(f"  False Positives : {clinical['false_positives']} (incorrectly predicted PNEUMONIA)")
    print(f"  False Negatives : {clinical['false_negatives']} (missed PNEUMONIA cases)")
    
    print(f"\nPer-Class Metrics:")
    for class_name in CLASS_NAMES:
        class_metrics = per_class[class_name]
        print(f"\n  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall   : {class_metrics['recall']:.4f}")
        print(f"    F1-Score : {class_metrics['f1_score']:.4f}")
        print(f"    Support  : {class_metrics['support']} samples")
    
    print("\n" + "=" * 70)

def save_metrics_json(metrics, test_loss, test_accuracy, output_path):
    """
    Save metrics to JSON file with timestamp and metadata.
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "metrics": metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Test metrics saved to {output_path}")

def save_classification_report(labels, predictions, output_path):
    """
    Save sklearn classification report as JSON.
    """
    report_dict = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    report_dict = convert_to_serializable(report_dict)
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"✓ Classification report saved to {output_path}")

# ======================================
# MAIN EVALUATION PIPELINE
# ======================================

def evaluate_model(model_path=MODEL_PATH):
    """
    Full evaluation pipeline:
    1. Load model
    2. Load test data
    3. Run inference
    4. Compute metrics
    5. Generate visualizations
    6. Save results
    """
    print(f"Device: {DEVICE}")
    print(f"Model path: {model_path}\n")
    
    # Load model
    print("Loading model...")
    model = build_model(num_classes=2).to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    
    # Load dataloaders
    print("Loading test data...")
    _, _, test_loader, _ = get_dataloaders()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    print("Running evaluation on test set...\n")
    results = evaluate(model, test_loader, criterion)
    
    test_loss = results['test_loss']
    test_accuracy = results['test_accuracy']
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    # Compute metrics
    metrics, cm = compute_metrics(predictions, labels, probabilities)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save results
    print("\nSaving results...")
    save_metrics_json(metrics, test_loss, test_accuracy, TEST_METRICS_PATH)
    save_classification_report(labels, predictions, CLASSIFICATION_REPORT_PATH)
    
    # Generate visualizations
    plot_confusion_matrix(cm, CONFUSION_MATRIX_PATH)
    plot_roc_curve(labels, probabilities, ROC_CURVE_PATH)
    
    print("\n✓ Evaluation complete!")
    
    return {
    "test_accuracy": test_accuracy,
    "test_loss":     test_loss,
    "metrics": {
        "roc_auc":   metrics["overall"]["roc_auc"],
        "f1_macro":  metrics["overall"]["macro_f1"],
        "recall":    metrics["clinical_metrics"]["sensitivity"],
        "precision": metrics["overall"]["macro_precision"],
    }
}
# ======================================
# ENTRY POINT
# ======================================

if __name__ == "__main__":
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train a model first: python src/training/train.py")
    else:
        evaluate_model()
