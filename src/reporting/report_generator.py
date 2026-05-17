"""
Structured Clinical Report Generator.

Design philosophy:
- Template-based first. No LLM dependency.
- Input: prediction dict from Grad-CAM pipeline
- Output: structured text report mimicking radiology report format

WHY TEMPLATE-BASED (interview answer):
- LLMs add latency, cost, and hallucination risk
- In medical AI, every output must be deterministic and auditable
- Template-based reports are explainable, testable, and fast
- LLM upgrade is a clear future improvement — shows you understand tradeoffs
"""

from datetime import datetime
from typing import Optional


# --- Threshold Configuration ---
# These thresholds determine report language severity
# In production: these would be calibrated on validation data
CONFIDENCE_HIGH   = 85.0   # > 85%  → strong language
CONFIDENCE_MEDIUM = 60.0   # 60-85% → moderate language
                            # < 60%  → uncertain language


def _get_finding_statement(predicted_class: str, confidence: float) -> str:
    """
    Maps prediction + confidence → clinical finding language.
    Mirrors how radiologists grade certainty in reports.
    """
    if predicted_class == "PNEUMONIA":
        if confidence >= CONFIDENCE_HIGH:
            return (
                "Imaging findings are strongly consistent with pneumonia. "
                "Increased opacity and consolidation patterns are identified "
                "in the lung fields, suggestive of an active infectious process."
            )
        elif confidence >= CONFIDENCE_MEDIUM:
            return (
                "Imaging findings are moderately suggestive of pneumonia. "
                "Some opacity changes are noted in the lung fields. "
                "Clinical correlation is recommended."
            )
        else:
            return (
                "Imaging findings show possible early or mild pneumonia. "
                "Findings are inconclusive at this confidence level. "
                "Clinical correlation and follow-up imaging are strongly recommended."
            )
    else:  # NORMAL
        if confidence >= CONFIDENCE_HIGH:
            return (
                "No significant acute cardiopulmonary abnormality detected. "
                "Lung fields appear clear with no consolidation or opacity patterns "
                "suggestive of pneumonia."
            )
        elif confidence >= CONFIDENCE_MEDIUM:
            return (
                "No definitive signs of pneumonia detected. "
                "Lung fields appear largely clear. "
                "Clinical correlation is advised if symptoms persist."
            )
        else:
            return (
                "Findings are within normal limits, though confidence is moderate. "
                "Clinical correlation is recommended if symptoms persist."
            )


def _get_impression(predicted_class: str, confidence: float) -> str:
    if predicted_class == "PNEUMONIA":
        if confidence >= CONFIDENCE_HIGH:
            return "HIGH suspicion for pneumonia. Further clinical evaluation and treatment planning recommended."
        elif confidence >= CONFIDENCE_MEDIUM:
            return "MODERATE suspicion for pneumonia. Radiological review and clinical correlation advised."
        else:
            return "LOW-to-MODERATE suspicion for pneumonia. Follow-up imaging recommended."
    else:
        if confidence >= CONFIDENCE_HIGH:
            return "No evidence of pneumonia. Chest X-ray within normal limits."
        else:
            return "No definitive pneumonia detected. Monitor if symptomatic."


def _get_recommendation(predicted_class: str, confidence: float) -> list:
    if predicted_class == "PNEUMONIA":
        recs = [
            "Correlate with patient symptoms (fever, cough, dyspnea).",
            "Consider laboratory workup (CBC, CRP, blood culture).",
            "Antibiotic therapy may be indicated pending clinical assessment.",
        ]
        if confidence < CONFIDENCE_HIGH:
            recs.append("Repeat chest X-ray in 24-48 hours if symptoms persist.")
        return recs
    else:
        recs = [
            "No immediate radiological intervention required.",
            "Routine clinical follow-up as indicated by symptoms.",
        ]
        if confidence < CONFIDENCE_HIGH:
            recs.append("Consider follow-up imaging if clinical presentation worsens.")
        return recs


def generate_report(
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    image_filename: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> dict:
    """
    Generates a structured clinical report.

    Args:
        predicted_class : "NORMAL" or "PNEUMONIA"
        confidence      : float, 0-100
        probabilities   : {"NORMAL": float, "PNEUMONIA": float}
        image_filename  : original image filename (optional)
        patient_id      : patient identifier (optional)

    Returns:
        dict with full report structure + plain text version
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    finding     = _get_finding_statement(predicted_class, confidence)
    impression  = _get_impression(predicted_class, confidence)
    recs        = _get_recommendation(predicted_class, confidence)

    # Confidence tier label
    if confidence >= CONFIDENCE_HIGH:
        confidence_tier = "HIGH"
    elif confidence >= CONFIDENCE_MEDIUM:
        confidence_tier = "MODERATE"
    else:
        confidence_tier = "LOW"

    report = {
        "metadata": {
            "report_generated_at": timestamp,
            "patient_id":          patient_id or "N/A",
            "image_file":          image_filename or "N/A",
            "model":               "ResNet18 (Transfer Learning)",
            "disclaimer":          (
                "Not intended for clinical diagnosis or medical decision-making."
            ),
        },
        "ai_analysis": {
            "predicted_class":  predicted_class,
            "confidence":       f"{confidence:.2f}%",
            "confidence_tier":  confidence_tier,
            "probabilities": {
                "NORMAL":    f"{probabilities.get('NORMAL', 0):.2f}%",
                "PNEUMONIA": f"{probabilities.get('PNEUMONIA', 0):.2f}%",
            },
        },
        "findings":        finding,
        "impression":      impression,
        "recommendations": recs,
    }

    # Plain text version for display in Streamlit / API response
    rec_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(recs))

    report["plain_text"] = f"""
================================================================================
                    CHEST X-RAY ANALYSIS REPORT
================================================================================
Date/Time    : {timestamp}
Patient ID   : {patient_id or 'N/A'}
Image File   : {image_filename or 'N/A'}
Model        : ResNet18 (Transfer Learning, ImageNet pretrained)

--------------------------------------------------------------------------------
CLASSIFICATION
--------------------------------------------------------------------------------
  Prediction       : {predicted_class}
  Confidence       : {confidence:.2f}% ({confidence_tier})
  P(NORMAL)        : {probabilities.get('NORMAL', 0):.2f}%
  P(PNEUMONIA)     : {probabilities.get('PNEUMONIA', 0):.2f}%

--------------------------------------------------------------------------------
FINDINGS
--------------------------------------------------------------------------------
  {finding}

--------------------------------------------------------------------------------
IMPRESSION
--------------------------------------------------------------------------------
  {impression}

--------------------------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------------------------
{rec_text}

--------------------------------------------------------------------------------
DISCLAIMER
--------------------------------------------------------------------------------
  It is NOT intended for clinical diagnosis or medical decision-making.
  All findings must be reviewed and confirmed by a qualified radiologist.
================================================================================
""".strip()

    return report


if __name__ == "__main__":
    # Test with a mock prediction
    mock_result = {
        "predicted_class": "PNEUMONIA",
        "confidence":      96.43,
        "probabilities":   {"NORMAL": 3.57, "PNEUMONIA": 96.43},
    }

    report = generate_report(
        predicted_class=mock_result["predicted_class"],
        confidence=mock_result["confidence"],
        probabilities=mock_result["probabilities"],
        image_filename="test_xray.jpeg",
        patient_id="TEST-001",
    )

    print(report["plain_text"])