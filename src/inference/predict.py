import torch
import torch.nn.functional as F
from PIL import Image

from src.training.model import build_model
from src.preprocessing.transforms import get_val_transforms

# ======================================
# CONFIG
# ======================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/best_model.pth"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ======================================
# LOAD MODEL
# ======================================

model = build_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)

model.eval()

# ======================================
# PREDICTION FUNCTION
# ======================================

def predict_image(image: Image.Image):
    
    image = image.convert("RGB")

    transform = get_val_transforms()

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    prediction = CLASS_NAMES[predicted_class.item()]

    confidence_score = min(
    round(confidence.item() * 100, 2),
    99.9
    )

    return {
        "prediction": prediction,
        "confidence": confidence_score
    }