# utils.py — helper functions for the backend

from config import MODEL_NAMES

def safe_encode(encoder, value, fallback=0):
    """Encode a categorical value safely — return fallback if unseen."""
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return fallback

def get_severity(prediction: int, probability: float):
    """Return severity label, color key, and recommended action."""
    if prediction == 1:
        if probability >= 0.75:
            return (
                "Critical",
                "critical",
                "Immediate deployment required — pre-position emergency shelter, food, and medical teams now."
            )
        return (
            "High",
            "high",
            "High risk — alert field teams and prepare logistics for rapid response."
        )
    else:
        if probability <= 0.30:
            return (
                "Low",
                "low",
                "Low risk — monitor the situation and keep standard readiness."
            )
        return (
            "Moderate",
            "moderate",
            "Moderate risk — continue monitoring and keep response teams on standby."
        )

def get_model_name(key: str) -> str:
    return MODEL_NAMES.get(key, key)
