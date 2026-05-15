from PIL import Image

from src.explainability.gradcam import generate_gradcam

image = Image.open(
    "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
)

heatmap = generate_gradcam(image)

heatmap.save("outputs/visualizations/gradcam_result.png")

print("Grad-CAM saved successfully.")