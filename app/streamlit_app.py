import sys
import os

# ======================================
# FIX IMPORT PATH
# ======================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ======================================
# IMPORTS
# ======================================

import streamlit as st
from PIL import Image

from src.inference.predict import predict_image

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="AI Pneumonia Detection",
    layout="centered"
)

# ======================================
# TITLE
# ======================================

st.title("AI-Powered Pneumonia Detection")

st.write(
    """
    Upload a chest X-ray image to detect
    signs of pneumonia using a deep learning model.
    """
)

# ======================================
# FILE UPLOAD
# ======================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# ======================================
# INFERENCE
# ======================================

if uploaded_file:

    # Load image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    st.success("Image uploaded successfully.")

    # Run prediction
    result = predict_image(image)

    # ======================================
    # PREDICTION DISPLAY
    # ======================================

    st.subheader("Prediction")

    st.metric(
        label="Diagnosis",
        value=result["prediction"]
    )

    # ======================================
    # CONFIDENCE DISPLAY
    # ======================================

    st.subheader("Confidence")

    st.progress(
        int(result["confidence"])
    )

    st.write(
        f"{result['confidence']}%"
    )

    # ======================================
    # CLINICAL REPORT
    # ======================================

    st.subheader("Clinical Report")

    if result["prediction"] == "PNEUMONIA":

        st.error(
            """
            Findings:
            Signs consistent with pneumonia detected
            in the chest X-ray image.

            Recommendation:
            Further radiological evaluation and
            clinical consultation recommended.
            """
        )

    else:

        st.success(
            """
            Findings:
            No strong radiological evidence of
            pneumonia detected.

            Recommendation:
            Continue routine clinical monitoring
            if symptoms persist.
            """
        )