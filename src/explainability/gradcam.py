"""
Grad-CAM (Gradient-weighted Class Activation Mapping)

WHY THIS MATTERS FOR CHEST X-RAY DIAGNOSIS:
- A model that says "PNEUMONIA" with 96% confidence is useless
  if it's looking at the wrong part of the image
- Grad-CAM shows WHICH pixels drove the decision
- For chest X-rays: heatmap should highlight lung regions
  If it highlights the image border or text annotations → model is cheating
- This is called a "sanity check" on model behavior
- Without explainability, no hospital will trust your model

HOW GRAD-CAM WORKS (one paragraph explanation):
- Forward pass: image goes through CNN → prediction
- We pick the target class (PNEUMONIA = 1)
- Backward pass: compute gradients of that class score
  with respect to the LAST convolutional layer's feature maps
- Global average pool the gradients → importance weight per feature map
- Weighted sum of feature maps → heatmap
- Regions with high activation = regions that drove the prediction
- Upsample heatmap to original image size → overlay
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.training.model import build_model

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_model():
    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path: str) -> tuple:
    """
    Returns:
        tensor : preprocessed tensor ready for model (1, 3, 224, 224)
        original: original PIL image for overlay
    """
    original = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(original).unsqueeze(0)  # add batch dim → (1, 3, 224, 224)
    return tensor, original


def generate_gradcam(model, tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
    """
    Generates Grad-CAM heatmap for the given input tensor.

    Args:
        model       : loaded ResNet18
        tensor      : preprocessed image tensor (1, 3, 224, 224)
        target_class: class index to explain. If None, uses predicted class.

    Returns:
        heatmap: numpy array (224, 224) with values in [0, 1]
    """
    # Storage for forward activations and backward gradients
    activations = {}
    gradients   = {}

    # Hook 1: capture output of layer4 (last conv block)
    def forward_hook(module, input, output):
        activations["layer4"] = output.detach()

    # Hook 2: capture gradients flowing back through layer4
    def backward_hook(module, grad_input, grad_output):
        gradients["layer4"] = grad_output[0].detach()

    # Register hooks on layer4
    handle_fwd = model.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.layer4.register_full_backward_hook(backward_hook)

    # Forward pass
    tensor = tensor.to(DEVICE)
    tensor.requires_grad_(True)
    output = model(tensor)                          # shape: (1, 2)
    probs  = torch.softmax(output, dim=1)

    # Use predicted class if target not specified
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass for target class only
    model.zero_grad()
    output[0, target_class].backward()

    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # --- Compute Grad-CAM ---
    acts  = activations["layer4"].squeeze(0)   # (512, 7, 7)
    grads = gradients["layer4"].squeeze(0)     # (512, 7, 7)

    # Global average pool gradients → importance weight per channel
    weights = grads.mean(dim=(1, 2))           # (512,)

    # Weighted sum of activation maps
    cam = torch.zeros(acts.shape[1:], device=DEVICE)  # (7, 7)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU: only keep positive contributions
    # Negative values mean "evidence against this class" — not useful for visualization
    cam = F.relu(cam)

    # Normalize to [0, 1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Upsample from 7×7 to 224×224
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

    return cam, probs, target_class


def overlay_heatmap(cam: np.ndarray, original_image: Image.Image) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on the original X-ray image.

    Returns:
        overlaid: numpy array (H, W, 3) in BGR format for cv2 saving
                  and RGB format for PIL/Streamlit display
    """
    # Resize original to 224x224
    orig_np = np.array(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))

    # Convert cam to color heatmap (COLORMAP_JET: blue=low, red=high attention)
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # convert to RGB

    # Blend: 60% original + 40% heatmap
    overlaid = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

    return overlaid


def run_gradcam_pipeline(image_path: str, save_path: str = None) -> dict:
    """
    Full pipeline: image path → prediction + heatmap overlay

    Args:
        image_path: path to chest X-ray image
        save_path : optional path to save overlay image

    Returns:
        dict with keys: predicted_class, confidence, heatmap, overlay
    """
    model  = load_model()
    tensor, original = preprocess_image(image_path)

    cam, probs, pred_class = generate_gradcam(model, tensor)
    overlay = overlay_heatmap(cam, original)

    confidence = probs[0, pred_class].item()

    if save_path:
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, overlay_bgr)
        print(f"Heatmap saved → {save_path}")

    return {
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence":      round(confidence * 100, 2),
        "heatmap":         cam,
        "overlay":         overlay,
        "probabilities": {
            "NORMAL":    round(probs[0, 0].item() * 100, 2),
            "PNEUMONIA": round(probs[0, 1].item() * 100, 2),
        }
    }


if __name__ == "__main__":
    import sys
    import os

    os.makedirs("outputs/visualizations", exist_ok=True)

    # Usage: python -m src.explainability.gradcam <image_path>
    if len(sys.argv) < 2:
        print("Usage: python -m src.explainability.gradcam <path_to_xray_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = run_gradcam_pipeline(
        image_path,
        save_path="outputs/visualizations/gradcam_output.png"
    )

    print(f"\nPrediction  : {result['predicted_class']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"NORMAL      : {result['probabilities']['NORMAL']}%")
    print(f"PNEUMONIA   : {result['probabilities']['PNEUMONIA']}%")