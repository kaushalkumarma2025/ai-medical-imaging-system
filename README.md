# AI-Powered Medical Imaging and Clinical Reporting System

An end-to-end deep learning system for automated chest X-ray analysis,
pneumonia classification, explainability visualization, and structured
clinical report generation using PyTorch, FastAPI, and computer vision workflows.

The project is designed with a modular machine learning engineering architecture
focused on scalability, reproducibility, deployment readiness, and healthcare AI applications.

---

## Features

- Chest X-ray pneumonia classification
- Transfer learning using ResNet18
- Medical image preprocessing and augmentation
- Weighted loss handling for class imbalance
- Modular training and validation pipeline
- Grad-CAM explainability visualization
- FastAPI backend scaffolding
- Streamlit frontend scaffolding
- Deployment-oriented project architecture

---

## Tech Stack

### Machine Learning and Deep Learning
- PyTorch
- Torchvision
- Scikit-learn

### Computer Vision
- OpenCV
- Pillow

### Backend and Frontend
- FastAPI
- Streamlit

### Data and Utilities
- NumPy
- Pandas
- Matplotlib

### Deployment and Version Control
- Docker
- Git/GitHub

---

## System Architecture

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

## Project Structure

```bash
ai-medical-imaging-system/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│
├── notebooks/
│
├── outputs/
│
├── src/
│   ├── preprocessing/
│   │   └── transforms.py
│   │
│   ├── training/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   ├── inference/
│   │
│   ├── explainability/
│   │
│   └── reporting/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

The project uses the Chest X-ray Pneumonia dataset containing
normal and pneumonia chest radiographs for binary image classification.

Dataset structure:

```bash
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## Model Pipeline

### Preprocessing
- Image resizing
- Tensor conversion
- Normalization
- Data augmentation

### Training
- Transfer learning using ResNet18
- Frozen backbone with trainable final layers
- Weighted cross-entropy loss for imbalance correction
- Adam optimizer

### Evaluation
- Training and validation accuracy tracking
- Validation loss monitoring
- Best model checkpoint saving

---

## Current Progress

- Dataset preprocessing pipeline completed
- Transfer learning architecture implemented
- Weighted loss handling implemented
- Modular training pipeline completed
- Validation workflow implemented
- Model checkpoint saving implemented
- Backend and frontend scaffolding created

---

## Planned Improvements

- Grad-CAM heatmap generation
- FastAPI inference endpoint
- Streamlit clinical dashboard
- Structured clinical report generation
- Docker containerization
- ROC-AUC and F1-score evaluation
- Model monitoring dashboard
- Cloud deployment support

---

## Current Training Snapshot

- Training Images: 5216
- Validation Images: 16
- Test Images: 624
- Trainable Parameters: 8.39M

Example initial training result:

```text
Epoch [10/10]
Train Accuracy: 94.67%
Validation Accuracy: 81.25%
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/kaushalkumarma2025/ai-medical-imaging-system.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training pipeline:

```bash
python -m src.training.train
```

---

## Future Goals

The long-term objective is to build a deployment-ready healthcare AI system
capable of:
- medical image classification
- explainable AI visualization
- automated structured reporting
- scalable inference APIs
- clinical workflow integration

---
## Demo Screenshots

### Upload Interface

![Dark themed web app page titled AI-Powered Pneumonia Detection showing a user uploading chest X ray file person1_virus_8.jpeg and previewing the chest X ray below in a browser at localhost:8501. Visible text includes AI-Powered Pneumonia Detection, Upload a chest X-ray image to detect signs of pneumonia using a deep learning model, Upload Chest X-ray, and person1_virus_8.jpeg 44.5KB. The wider environment is a desktop browser window with tabs and a taskbar visible, creating a focused clinical and technical tone.](outputs/screenshots/upload_ui.png)

### Prediction Result

![Dark themed prediction results page in a browser at localhost:8501 showing an uploaded chest X ray analysis outcome. Main panel text reads Uploaded Chest X-ray, Image uploaded successfully., Prediction, Diagnosis, PNEUMONIA, Confidence, 99.9%, Clinical Report, Findings: Signs consistent with pneumonia detected in the chest X-ray image., and Recommendation: Further radiological evaluation and clinical consultation recommended. The wider environment is a desktop browser window with navigation tabs and taskbar visible, and the emotional tone is serious, urgent, and clinically informative.](outputs/screenshots/prediction_ui.png)
## Author

Kaushal Kumar  
MA Economics, Ashoka University  
Machine Learning | Healthcare AI | Quantitative Modeling