from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from src.inference.predict import predict_image

app = FastAPI(
    title="Medical Imaging AI API",
    version="1.0",
    description="Pneumonia detection system using deep learning"
)

# ======================================
# ROOT ENDPOINT
# ======================================

@app.get("/")
def home():

    return {
        "message": "Medical Imaging AI API Running"
    }

# ======================================
# HEALTH ENDPOINT
# ======================================

@app.get("/health")
def health():

    return {
        "status": "running"
    }

# ======================================
# PREDICTION ENDPOINT
# ======================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        result = predict_image(image)

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }

    except Exception as e:

        return {
            "error": str(e)
        }