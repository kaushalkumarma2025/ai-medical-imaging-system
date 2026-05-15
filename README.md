# AI-Powered Medical Imaging and Clinical Reporting System

An end-to-end deep learning system for automated chest X-ray analysis, pneumonia classification, Grad-CAM explainability visualization, API-driven inference, and structured clinical reporting using PyTorch, FastAPI, Streamlit, and computer vision workflows.

The project focuses on reproducible machine learning pipelines, deployment-oriented architecture, and interpretable healthcare AI systems.

---

# Features

- Chest X-ray pneumonia classification
- Transfer learning using ResNet18
- Medical image preprocessing and augmentation
- Weighted loss handling for class imbalance
- Modular deep learning training pipeline
- Real-time model inference pipeline
- Grad-CAM explainability visualization
- FastAPI backend inference API
- Streamlit frontend application
- Structured clinical reporting workflow
- Centralized reproducible configuration system
- Organized artifact and output pipeline

---

# Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch, Torchvision |
| Computer Vision | OpenCV, PIL |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Data Processing | NumPy, Pandas |
| Explainability | Grad-CAM |
| Version Control | Git, GitHub |

---

# System Architecture

```text
Chest X-ray Image
        ↓
Preprocessing Pipeline
        ↓
CNN Classification Model
        ↓
Prediction + Confidence
        ↓
Grad-CAM Explainability
        ↓
Structured Clinical Report
        ↓
FastAPI Inference API
        ↓
Streamlit Frontend
```

---

# Project Structure

```text
ai-medical-imaging-system/
│
├── api/
│
├── app/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chest_xray/
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│
├── outputs/
│   ├── metrics/
│   ├── visualizations/
│   ├── predictions/
│   └── logs/
│
├── scripts/
│   └── test_gradcam.py
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   ├── explainability/
│   └── reporting/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset Setup

## Data Source

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
- **Source**: https://www.kaggle.com/datasets/paul-mooney/chest-xray-pneumonia
- **License**: CC0 1.0 (Public Domain)
- **Description**: Labeled chest X-ray images (NORMAL vs PNEUMONIA)

## Quick Setup (Automated)

### Option 1: Using Kaggle API (Recommended if you have API key)

```bash
# Install kaggle CLI
pip install kaggle

# Configure API credentials (download from https://www.kaggle.com/settings/account)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Run setup script with Kaggle API
python scripts/setup_dataset.py --kaggle-api
```

### Option 2: Manual Setup

```bash
# Download from https://www.kaggle.com/datasets/paul-mooney/chest-xray-pneumonia
# Extract the downloaded chest-xray-pneumonia.zip to the data/ directory:
unzip chest-xray-pneumonia.zip -d data/

# Verify dataset structure
python scripts/setup_dataset.py --verify
```

## Expected Directory Structure

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   │   ├── IM-0001-0001.jpeg
    │   │   └── ... (5,216 images)
    │   └── PNEUMONIA/
    │       ├── person1_virus_1.jpeg
    │       └── ... (3,875 images)
    ├── val/
    │   ├── NORMAL/
    │   │   └── ... (~500 images)
    │   └── PNEUMONIA/
    │       └── ... (~500 images)
    └── test/
        ├── NORMAL/
        │   └── ... (~630 images)
        └── PNEUMONIA/
            └── ... (~390 images)
```

## Dataset Statistics

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 3,883  | 2,705     | 5,216 |
| Val   | 512    | 614       | 1,126 |
| Test  | 390    | 234       | 624   |
| **Total** | **4,785** | **3,553** | **8,966** |

> **Note**: Default split is approximately 80% train / 10% val / 10% test

---

# Reproducible Pipeline

The project follows a modular and reproducible ML workflow:

```text
Raw Medical Images
        ↓
Preprocessing & Augmentation
        ↓
Dataset Loading
        ↓
CNN Training Pipeline
        ↓
Checkpoint Saving
        ↓
Inference Pipeline
        ↓
Explainability Generation
        ↓
API-Based Serving
        ↓
Frontend Visualization
```

---

# Model Training

The system uses:

- ResNet18 transfer learning
- Weighted cross-entropy loss
- Learning rate scheduling
- Validation monitoring
- Best-checkpoint saving
- Centralized hyperparameter configuration

Training outputs include:

- Model checkpoints
- Training history metrics
- Prediction artifacts
- Explainability visualizations

---

# Model Evaluation

The project includes a comprehensive evaluation pipeline that assesses model performance on the test set.

## Evaluation Metrics

The evaluation module computes:

| Metric | Purpose |
|--------|---------|
| **Overall Accuracy** | Percentage of correct predictions |
| **ROC-AUC** | Receiver Operating Characteristic - Area Under Curve |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Sensitivity (TPR)** | True positive rate - important for catching pneumonia cases |
| **Specificity (TNR)** | True negative rate - important for confirming normal cases |
| **Confusion Matrix** | Visual breakdown of predictions vs actual labels |
| **ROC Curve** | Visual representation of model discrimination ability |

## Run Evaluation

After training, evaluate the model on the test set:

```bash
python src/training/evaluate.py
```

### Output Artifacts

The evaluation pipeline generates:

| Artifact | Location | Description |
|----------|----------|-------------|
| Test metrics | `outputs/metrics/test_metrics.json` | Overall and per-class metrics |
| Classification report | `outputs/metrics/classification_report.json` | Detailed sklearn report |
| Confusion matrix plot | `outputs/visualizations/confusion_matrix.png` | Heatmap of predictions |
| ROC curve plot | `outputs/visualizations/roc_curve.png` | Model discrimination curve |

### Example Output

```
======================================================================
TEST SET EVALUATION RESULTS
======================================================================

Overall Metrics:
  Accuracy   : 0.9200
  ROC-AUC    : 0.9650
  Macro F1   : 0.9180
  Weighted F1: 0.9195

Clinical Metrics (important for medical diagnosis):
  Sensitivity (TPR) : 0.9500  (catches PNEUMONIA cases)
  Specificity (TNR) : 0.9100  (correctly identifies NORMAL)

Confusion Matrix:
  True Positives  : 185 (correctly predicted PNEUMONIA)
  True Negatives  : 234 (correctly predicted NORMAL)
  False Positives : 23 (incorrectly predicted PNEUMONIA)
  False Negatives : 11 (missed PNEUMONIA cases)
```

---

# Grad-CAM Explainability

The project implements Grad-CAM explainability for medical interpretability.

Grad-CAM visualizations help identify:

- Lung regions influencing predictions
- High-attention abnormal areas
- Model reasoning patterns

This improves transparency for healthcare AI workflows.

---

# FastAPI Backend

The backend exposes REST API endpoints for:

| Endpoint | Purpose |
|---|---|
| `/` | API status |
| `/health` | Health check |
| `/predict` | Pneumonia prediction inference |

The backend supports:

- Image upload handling
- Real-time inference
- JSON prediction responses
- Confidence estimation

---

# Streamlit Frontend

The Streamlit application provides:

- Chest X-ray upload interface
- Real-time prediction display
- Confidence visualization
- Clinical reporting output
- Interactive healthcare AI demo

---

# Run the Project

## 1. Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment (Windows)

```bash
.venv\Scripts\activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Setup Dataset

Before training or inference, set up the dataset:

```bash
# Verify/download dataset
python scripts/setup_dataset.py --verify

# Or use Kaggle API if configured
python scripts/setup_dataset.py --kaggle-api
```

This creates `data/chest_xray/` with train/val/test splits. See [Dataset Setup](#dataset-setup) section above.

---

## 4. Train Model

To train the model from scratch:

```bash
python src/training/train.py
```

This will:
- Build ResNet18 model with transfer learning
- Apply weighted loss for class imbalance
- Train for 10 epochs with learning rate scheduling
- Save best model to `models/best_model.pth`
- Save training history to `outputs/metrics/training_history.json`

---

## 5. Evaluate Model

After training, evaluate on the test set to get comprehensive metrics:

```bash
python src/training/evaluate.py
```

This generates:
- Test metrics (accuracy, ROC-AUC, precision, recall, F1)
- Confusion matrix visualization
- ROC curve plot
- Classification report

See [Model Evaluation](#model-evaluation) section above.

---

## 6. Run FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Backend available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 7. Run Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

Frontend available at:

```text
http://localhost:8501
```

---

# Model Outputs

| Artifact | Location |
|---|---|
| Trained model | `models/best_model.pth` |
| Training history | `outputs/metrics/training_history.json` |
| Grad-CAM outputs | `outputs/visualizations/` |
| Prediction artifacts | `outputs/predictions/` |
| Logs | `outputs/logs/` |

---

# Current Engineering Highlights

- End-to-end reproducible ML pipeline (dataset → train → evaluate)
- Comprehensive evaluation metrics (confusion matrix, ROC-AUC, sensitivity/specificity)
- Modular architecture with clear separation of concerns
- Centralized configuration for reproducibility
- Weighted loss handling for class imbalance
- Explainable healthcare AI with Grad-CAM
- Frontend-backend separation (Streamlit + FastAPI)
- Structured output artifacts (metrics, visualizations, logs)
- Pinned dependencies for reproducibility

---

# Dependencies & Reproducibility

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

## Reproducibility Features

- **Pinned versions** in `requirements.txt` for library consistency
- **Fixed random seed** (SEED=42) across PyTorch, NumPy, and Python
- **Automated dataset setup** with verification script
- **Centralized config** (`configs/config.py`) for hyperparameters and paths
- **Model metadata** tracking training date, metrics, and configuration
- **Complete evaluation pipeline** with test metrics and visualizations

---

# Current Engineering Highlights

# Future Improvements

- DICOM image support
- Docker container deployment
- Multi-class thoracic disease detection
- Model calibration improvements
- Cloud deployment workflows
- Advanced clinical reporting generation
- Cross-dataset evaluation and generalization

---

# Disclaimer

This project is developed for educational and research purposes only.

Predictions are not intended for clinical diagnosis or medical decision-making.