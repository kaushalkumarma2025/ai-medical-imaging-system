from PIL import Image

from src.inference.predict import predict_image

# LOAD TEST IMAGE


image = Image.open(
    "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
)


# RUN PREDICTION


result = predict_image(image)

print(result)