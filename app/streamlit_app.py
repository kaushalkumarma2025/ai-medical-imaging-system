import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from PIL import Image
import tempfile

from src.explainability.gradcam import run_gradcam_pipeline
from src.reporting.report_generator import generate_report

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
)

# ─── Title ─────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Pneumonia Analysis")
st.markdown("Upload a chest X-ray image to get classification, Grad-CAM heatmap, and clinical report.")
st.divider()

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model**: ResNet18 (transfer learning)

    **Task**: Binary classification
    - NORMAL
    - PNEUMONIA

    **Metrics (test set)**:
    - Accuracy : 95.74%
    - ROC-AUC  : 98.75%
    - Sensitivity: 96.03%
    - Specificity: 94.97%
    """)
    st.divider()
    st.markdown("⚠️ For research purposes only. Not for clinical use.")

# ─── Upload ────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:

    # Save to temp file — pipeline needs a file path
    ext     = os.path.splitext(uploaded_file.name)[1].lower()
    tmp     = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.read())
    tmp.close()

    # ─── Layout: 3 columns ─────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original X-Ray")
        original_img = Image.open(tmp.name).convert("RGB")
        st.image(original_img, use_column_width=True)

    with st.spinner("Running analysis..."):
        result = run_gradcam_pipeline(tmp.name)
        report = generate_report(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            image_filename=uploaded_file.name,
        )

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(result["overlay"], use_column_width=True)
        st.caption("Red = high attention regions driving prediction")

    with col3:
        st.subheader("Prediction")

        pred  = result["predicted_class"]
        conf  = result["confidence"]
        probs = result["probabilities"]

        # Color-coded result
        if pred == "PNEUMONIA":
            st.error(f"**{pred}**")
        else:
            st.success(f"**{pred}**")

        st.metric("Confidence", f"{conf:.2f}%")

        st.markdown("**Class Probabilities**")
        st.progress(probs["NORMAL"] / 100,
                    text=f"NORMAL: {probs['NORMAL']:.2f}%")
        st.progress(probs["PNEUMONIA"] / 100,
                    text=f"PNEUMONIA: {probs['PNEUMONIA']:.2f}%")

        # Confidence tier
        if conf >= 85:
            tier_color = "🔴" if pred == "PNEUMONIA" else "🟢"
            st.markdown(f"{tier_color} Confidence tier: **HIGH**")
        elif conf >= 60:
            st.markdown("🟡 Confidence tier: **MODERATE**")
        else:
            st.markdown("⚪ Confidence tier: **LOW**")

    st.divider()

    # ─── Clinical Report ───────────────────────────────────────
    st.subheader("Clinical Report")

    tab1, tab2 = st.tabs(["Structured View", "Plain Text"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Findings**")
            st.info(report["findings"])
            st.markdown("**Impression**")
            st.warning(report["impression"])
        with c2:
            st.markdown("**Recommendations**")
            for i, rec in enumerate(report["recommendations"], 1):
                st.markdown(f"{i}. {rec}")
            st.markdown("**Metadata**")
            st.json(report["metadata"])

    with tab2:
        st.code(report["plain_text"], language=None)
        st.download_button(
            label="Download Report",
            data=report["plain_text"],
            file_name=f"report_{uploaded_file.name}.txt",
            mime="text/plain",
        )

    # Cleanup temp file
    os.remove(tmp.name)

else:
    # Placeholder when no image uploaded
    st.info("👆 Upload a chest X-ray image to begin analysis.")
    st.markdown("""
    **What this system does:**
    1. Classifies the X-ray as NORMAL or PNEUMONIA
    2. Generates a Grad-CAM heatmap showing which lung regions influenced the prediction
    3. Produces a structured clinical report with findings and recommendations
    """)