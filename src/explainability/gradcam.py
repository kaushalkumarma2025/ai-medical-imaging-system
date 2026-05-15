import torch
import numpy as np
import cv2

from PIL import Image
from torchvision.transforms.functional import to_pil_image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.training.model import build_model
from src.preprocessing.transforms import get_val_transforms

# ======================================
# CONFIG
# ======================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/best_model.pth"

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
# TARGET LAYER
# ======================================

target_layer = model.layer4[-1]

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

# ======================================
# GENERATE HEATMAP
# ======================================

def generate_gradcam(image: Image.Image):

    image = image.convert("RGB")

    transform = get_val_transforms()

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    rgb_image = np.array(image.resize((224, 224))) / 255.0

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    heatmap = Image.fromarray(visualization)

    return heatmap