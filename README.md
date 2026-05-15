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
| Deployment | Docker-ready architecture |
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

## 3. Run FastAPI Backend

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

## 4. Run Streamlit Frontend

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

- End-to-end deployable ML architecture
- Frontend-backend separation
- Centralized reproducible configuration
- Modular inference system
- Explainable healthcare AI pipeline
- Structured output management
- Production-oriented project organization

---

# Future Improvements

- Larger validation and test evaluation
- DICOM image support
- Docker container deployment
- Multi-class thoracic disease detection
- Model calibration improvements
- Cloud deployment workflows
- Advanced clinical reporting generation

---

# Disclaimer

This project is developed for educational and research purposes only.

Predictions are not intended for clinical diagnosis or medical decision-making.