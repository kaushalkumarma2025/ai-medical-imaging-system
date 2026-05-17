# Chest X-Ray Pneumonia Classification System

An end-to-end deep learning pipeline for chest X-ray analysis, pneumonia classification, Grad-CAM explainability, REST API inference, and structured clinical reporting — built with PyTorch, FastAPI, and Streamlit.

---

## Features

- Binary chest X-ray classification: NORMAL vs PNEUMONIA
- Transfer learning with ResNet18 (ImageNet pretrained)
- Medical image preprocessing and augmentation pipeline
- Weighted cross-entropy loss for class imbalance handling
- Modular, reproducible training pipeline with checkpoint saving
- Grad-CAM heatmap generation for prediction explainability
- FastAPI REST backend with Swagger documentation
- Streamlit frontend with upload, heatmap, and report display
- Structured template-based clinical report generation
- Centralized configuration via `configs/config.py`
- Full evaluation suite: accuracy, ROC-AUC, sensitivity, specificity, confusion matrix

---

## Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch 2.12, Torchvision 0.27 |
| Computer Vision | OpenCV 4.13, Pillow |
| Explainability | Grad-CAM 1.5.5 |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Data Processing | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Containerization | Docker |
| Version Control | Git, GitHub |

---

## System Architecture

```text
Chest X-ray Image
        ↓
Preprocessing Pipeline
(resize 224×224, normalize to ImageNet stats, augmentation on train only)
        ↓
ResNet18 Classification Model
(frozen conv1–layer3, trainable layer4 + FC head)
        ↓
Prediction + Confidence Score
        ↓
Grad-CAM Heatmap
(gradient-weighted class activation map via layer4)
        ↓
Structured Clinical Report
(template-based: findings, impression, recommendations)
        ↓
FastAPI Inference API
        ↓
Streamlit Frontend
```

---

## Project Structure

```text
medical-imaging-ai/
│
├── api/
│   └── main.py                  # FastAPI backend
│
├── app/
│   └── streamlit_app.py         # Streamlit frontend
│
├── configs/
│   └── config.py                # Centralized hyperparameters and paths
│
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── models/
│   └── best_model.pth           # Saved best checkpoint
│
├── notebooks/
│
├── outputs/
│   ├── metrics/                 # JSON metric files
│   ├── visualizations/          # Confusion matrix, ROC curve, Grad-CAM
│   ├── predictions/
│   ├── logs/
│   └── reports/                 # Pipeline summary reports
│
├── scripts/
│   ├── setup_dataset.py         # Dataset download, verify, rebalance
│   └── test_gradcam.py
│
├── src/
│   ├── preprocessing/
│   │   └── transforms.py        # Train/val augmentation pipelines
│   ├── training/
│   │   ├── dataset.py           # DataLoader construction
│   │   ├── model.py             # ResNet18 with transfer learning
│   │   ├── train.py             # Training loop with weighted loss
│   │   └── evaluate.py          # Full test set evaluation
│   ├── inference/
│   ├── explainability/
│   │   └── gradcam.py           # Grad-CAM heatmap generation
│   └── reporting/
│       └── report_generator.py  # Structured clinical report
│
├── main.py                      # Unified pipeline orchestrator
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Source**: [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paul-mooney/chest-xray-pneumonia)
**License**: CC0 1.0 (Public Domain)

### Rebalanced Split (80/10/10 stratified, seed=42)

The default Kaggle split has only 16 validation images, making validation metrics meaningless. A stratified rebalance script is included:

```bash
python scripts/setup_dataset.py --rebalance-split --seed 42
```

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,266 | 3,418 | 4,684 |
| Val | 158 | 427 | 585 |
| Test | 159 | 428 | 587 |
| **Total** | **1,583** | **4,273** | **5,856** |

Class imbalance ratio: ~2.7:1 (PNEUMONIA:NORMAL). Handled via inverse-frequency weighted loss.

### Manual Setup

```bash
# Download from Kaggle link above
# Extract to data/ then verify:
python scripts/setup_dataset.py --verify

# Apply stratified rebalance (recommended):
python scripts/setup_dataset.py --rebalance-split --seed 42
```

---

## Training Pipeline

### Configuration

All hyperparameters are centralized in `configs/config.py`:

| Parameter | Value |
|---|---|
| Model | ResNet18 (ImageNet pretrained) |
| Frozen layers | conv1, layer1, layer2, layer3 |
| Trainable layers | layer4 + FC head |
| Loss | CrossEntropyLoss with class weights |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | StepLR (γ=0.5, step=5 epochs) |
| Epochs | 10 |
| Batch size | 32 |
| Seed | 42 |

### Augmentation (train split only)

| Transform | Rationale |
|---|---|
| RandomHorizontalFlip | Pathology appears on both sides |
| RandomRotation(±10°) | Patient positioning variation |
| ColorJitter(brightness, contrast) | Equipment exposure variation |
| Normalize (ImageNet mean/std) | Required for pretrained weights |

### Run Training

```bash
python -m src.training.train
```

Outputs saved to:
- `models/best_model.pth` — best checkpoint by validation loss
- `outputs/metrics/training_history.json` — per-epoch loss and accuracy

---

## Model Evaluation

```bash
python -m src.training.evaluate
```

### Results on Test Set (587 images)

| Metric | Value |
|---|---|
| Accuracy | 95.74% |
| ROC-AUC | **98.75%** |
| Sensitivity (Recall) | 96.03% |
| Specificity | 94.97% |
| Macro F1 | 94.70% |
| Weighted F1 | 95.78% |

### Confusion Matrix

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **Actual NORMAL** | 151 (TN) | 8 (FP) |
| **Actual PNEUMONIA** | 17 (FN) | 411 (TP) |

> **Why sensitivity matters most**: A false negative (missed pneumonia) sends a sick patient home untreated. A false positive leads to additional tests — costly but recoverable. The model is optimized to minimize false negatives.

### Output Artifacts

| Artifact | Path |
|---|---|
| Test metrics | `outputs/metrics/test_metrics.json` |
| Classification report | `outputs/metrics/classification_report.json` |
| Confusion matrix | `outputs/visualizations/confusion_matrix.png` |
| ROC curve | `outputs/visualizations/roc_curve.png` |

---

## Grad-CAM Explainability

Grad-CAM computes gradient-weighted class activation maps from the final convolutional layer (layer4), producing a heatmap that highlights which lung regions drove the prediction.

```bash
python -m src.explainability.gradcam "path/to/xray.jpeg"
```

Output saved to `outputs/visualizations/gradcam_output.png`.

Red/yellow regions = high attention. For a valid model, these must overlap with lung fields — not image borders or annotation text.

---

## API Usage

```bash
uvicorn api.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/health` | GET | Health check + model load status |
| `/predict` | POST | Upload X-ray → prediction + confidence + clinical report |
| `/heatmap` | POST | Upload X-ray → Grad-CAM overlay image |

### Example `/predict` Response

```json
{
  "status": "success",
  "filename": "xray.jpeg",
  "predicted_class": "PNEUMONIA",
  "confidence": 99.9,
  "probabilities": {
    "NORMAL": 0.1,
    "PNEUMONIA": 99.9
  },
  "report": {
    "findings": "Imaging findings are strongly consistent with pneumonia...",
    "impression": "HIGH suspicion for pneumonia...",
    "recommendations": ["Correlate with patient symptoms...", "..."]
  }
}
```

---

## Streamlit Frontend

```bash
streamlit run app/streamlit_app.py
```

Available at `http://localhost:8501`

Features:
- X-ray image upload
- Side-by-side: original image | Grad-CAM heatmap | prediction
- Confidence score with tier (HIGH / MODERATE / LOW)
- Structured clinical report with tabbed view
- Downloadable plain-text report

---

## Docker

```bash
# Build
docker build -t pneumonia-classifier .

# Run FastAPI backend
docker run -p 8000:8000 pneumonia-classifier

# Run Streamlit frontend
docker run -p 8501:8501 pneumonia-classifier \
  streamlit run app/streamlit_app.py --server.address 0.0.0.0
```

---

## Complete Pipeline (One Command)

```bash
# Full pipeline: verify dataset → train → evaluate → report
python main.py

# Skip dataset verification
python main.py --skip-dataset

# Skip training, use existing model
python main.py --skip-train

# Evaluate only
python main.py --evaluate-only
```

---

## Screenshots

*Add screenshots here after running the Streamlit app*

---

## Reproducibility

| Feature | Implementation |
|---|---|
| Fixed seed | SEED=42 across PyTorch, NumPy, Python |
| Pinned dependencies | `requirements.txt` with exact versions |
| Centralized config | `configs/config.py` for all hyperparameters |
| Stratified split | 80/10/10 with seed, class ratio preserved |
| Checkpoint saving | Best model by validation loss |

---

## Future Improvements

- DICOM image format support
- Multi-class thoracic disease detection (beyond binary)
- Model calibration (Platt scaling / temperature scaling)
- LLM-powered clinical report generation
- Cloud deployment (AWS / GCP)
- Cross-dataset generalization testing

---

## Disclaimer

This project is developed for educational and research purposes only.
Model predictions are not intended for clinical diagnosis or medical decision-making.
All outputs must be reviewed by a qualified radiologist before any clinical use.
