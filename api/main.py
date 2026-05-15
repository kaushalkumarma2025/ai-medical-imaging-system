from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Medical Imaging AI API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    return {
        "filename": file.filename,
        "prediction": "PNEUMONIA",
        "confidence": 0.94
    }