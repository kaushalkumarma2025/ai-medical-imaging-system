"""
Evaluation module for comprehensive model assessment.

Contains:
- evaluate_model(): Full evaluation pipeline
- evaluate(): Test set evaluation with metrics
- compute_metrics(): Compute comprehensive metrics (confusion matrix, ROC, etc.)
- plot_confusion_matrix(): Visualize confusion matrix
- plot_roc_curve(): Visualize ROC curve
"""

# Importing from training for now since we created it there
# In the future, this can be moved to src/evaluation/evaluate.py
from src.training.evaluate import evaluate_model
