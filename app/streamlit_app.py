import streamlit as st
from PIL import Image
import requests

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
# FILE UPLOADER
# ======================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# ======================================
# PREDICTION
# ======================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    st.success("Image uploaded successfully.")

    # ======================================
    # SEND IMAGE TO FASTAPI
    # ======================================

    files = {
        "file": uploaded_file.getvalue()
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files=files
    )

    result = response.json()

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    st.subheader("Prediction")

    if result["prediction"] == "PNEUMONIA":

        st.error("PNEUMONIA DETECTED")

    else:

        st.success("NORMAL")

    st.metric(
        "Confidence",
        f"{result['confidence']}%"
    )

    st.progress(
        int(result["confidence"])
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