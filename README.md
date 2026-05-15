Created 0 todos

```markdown
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
│   ├── setup_dataset.py
│   └── test_gradcam.py
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   ├── explainability/
│   └── reporting/
│
├── main.py
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
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Dataset Statistics

**Current Kaggle split (actual):**

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |
| **Total** | **1,583** | **4,273** | **5,856** |

> **Important:** Validation has only 16 images, so validation accuracy is unstable (1 image = 6.25%).

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

# Quick Start: Complete Pipeline

Run the entire workflow in one command:

```bash
# Full pipeline: dataset → train → evaluate → report
python main.py

# Skip dataset setup (use existing data)
python main.py --skip-dataset

# Skip training (use existing model)
python main.py --skip-train

# Evaluate only (no training)
python main.py --evaluate-only
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

| Metric | Purpose |
|--------|---------|
| **Overall Accuracy** | Percentage of correct predictions |
| **ROC-AUC** | Receiver Operating Characteristic - Area Under Curve |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Sensitivity (TPR)** | True positive rate (catching pneumonia cases) |
| **Specificity (TNR)** | True negative rate (correctly identifying normal cases) |
| **Confusion Matrix** | Visual breakdown of predictions vs actual labels |
| **ROC Curve** | Visual representation of model discrimination ability |

## Run Evaluation

```bash
python src/training/evaluate.py
```

### Output Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Test metrics | `outputs/metrics/test_metrics.json` | Overall and per-class metrics |
| Classification report | `outputs/metrics/classification_report.json` | Detailed sklearn report |
| Confusion matrix plot | `outputs/visualizations/confusion_matrix.png` | Heatmap of predictions |
| ROC curve plot | `outputs/visualizations/roc_curve.png` | Model discrimination curve |

---

# Grad-CAM Explainability

The project implements Grad-CAM explainability for medical interpretability.

Grad-CAM visualizations help identify:

- Lung regions influencing predictions
- High-attention abnormal areas
- Model reasoning patterns

---

# FastAPI Backend

The backend exposes REST API endpoints for:

| Endpoint | Purpose |
|---|---|
| `/` | API status |
| `/health` | Health check |
| `/predict` | Pneumonia prediction inference |

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

### Activate Environment (Linux/Mac)

```bash
source .venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Setup Dataset

```bash
python scripts/setup_dataset.py --verify
# or
python scripts/setup_dataset.py --kaggle-api
```

---

## 4. Run Complete Pipeline (Recommended)

```bash
python main.py
```

---

## 5. Train Model (Manual)

```bash
python src/training/train.py
```

---

## 6. Evaluate Model (Manual)

```bash
python src/training/evaluate.py
```

---

## 7. Run FastAPI Backend

```bash
uvicorn api.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

---

## 8. Run Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

- Frontend: `http://localhost:8501`

---

# Model Outputs

| Artifact | Location |
|---|---|
| Trained model | `models/best_model.pth` |
| Training history | `outputs/metrics/training_history.json` |
| Test metrics | `outputs/metrics/test_metrics.json` |
| Classification report | `outputs/metrics/classification_report.json` |
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

---

# Dependencies & Reproducibility

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducibility Features

- Pinned versions in `requirements.txt`
- Fixed random seed (`SEED=42`) across PyTorch/NumPy/Python
- Automated dataset setup and verification script
- Centralized config in `configs/config.py`
- Formal evaluation pipeline with test artifacts

---

# Future Improvements

- Validation split cleanup (current val=16 issue)
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
```