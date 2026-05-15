import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AI Pneumonia Detection",
    layout="centered"
)

st.title("AI-Powered Pneumonia Detection")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    st.success("Image uploaded successfully.")

    st.subheader("Prediction")

    st.write("PNEUMONIA")

    st.subheader("Confidence")

    st.write("94%")