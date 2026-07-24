from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
models = {
    "lr":  joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl")),
    "rf":  joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl")),
    "xgb": joblib.load(os.path.join(MODELS_DIR, "best_model.pkl")),
}

encoders   = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
config     = joblib.load(os.path.join(MODELS_DIR, "model_config.pkl"))

le_combined = encoders["combined_type"]
le_displace = encoders["displacement_type"]
le_region   = encoders["region"]

THRESHOLD = config["threshold"]   # 0.35 — tuned for recall
FEATURES  = config["features"]
METRICS   = config["metrics"]

MODEL_NAMES = {
    "lr":  "Logistic Regression",
    "rf":  "Random Forest",
    "xgb": "XGBoost",
}

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="Somalia Displacement Severity API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────
class EventInput(BaseModel):
    combined_type:     str       # Conflict | Drought | Flood
    displacement_type: str       # Conflict | Disaster
    region:            str       # e.g. Banaadir
    month:             int       # 1–12
    duration_days:     float
    latitude:          float
    longitude:         float
    year:              int
    model:             Optional[str] = "xgb"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def safe_encode(encoder, value, fallback=0):
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return fallback

def get_season(month: int) -> int:
    if month in [3, 4, 5]:  return 1   # Gu — long rains
    if month in [10, 11]:   return 2   # Deyr — short rains
    return 0                            # dry season

def get_severity(prediction: int, probability: float):
    if prediction == 1:
        if probability >= 0.75:
            return "Critical", "critical", \
                "Immediate deployment required — pre-position emergency shelter, food, and medical teams now."
        return "High", "high", \
            "High risk — alert field teams and prepare logistics for rapid response."
    else:
        if probability <= 0.30:
            return "Low", "low", \
                "Low risk — monitor the situation and keep standard readiness."
        return "Moderate", "moderate", \
            "Moderate risk — continue monitoring and keep response teams on standby."

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Somalia Displacement Severity API v2.0",
        "threshold": THRESHOLD,
    }

@app.get("/metrics")
def metrics():
    return {
        key: {
            "name":      MODEL_NAMES[key],
            "recall":    round(METRICS[key]["recall"], 3),
            "precision": round(METRICS[key]["precision"], 3),
            "f1":        round(METRICS[key]["f1"], 3),
            "accuracy":  round(METRICS[key]["accuracy"], 3),
            "auc":       round(METRICS[key]["auc"], 3),
        }
        for key in ["lr", "rf", "xgb"]
    }

@app.post("/predict")
def predict(event: EventInput):
    model_key = event.model if event.model in models else "xgb"
    model     = models[model_key]

    # Encode categoricals
    combined_enc = safe_encode(le_combined, event.combined_type)
    displace_enc = safe_encode(le_displace, event.displacement_type)
    region_enc   = safe_encode(le_region,   event.region)

    # Derived features — must match training
    is_flood      = 1 if event.combined_type == "Flood"    else 0
    is_conflict   = 1 if event.combined_type == "Conflict" else 0
    rainy_season  = get_season(event.month)
    lat_lon_inter = event.latitude * event.longitude
    week_of_year  = 1   # default — not available from UI input

    # Approximate week from month
    week_of_year = (event.month - 1) * 4 + 2

    # Build input dataframe — same column order as training
    row = {
        'combined_enc':      combined_enc,
        'displace_enc':      displace_enc,
        'region_enc':        region_enc,
        'month':             event.month,
        'week_of_year':      week_of_year,
        'duration_days':     event.duration_days,
        'latitude':          event.latitude,
        'longitude':         event.longitude,
        'lat_lon_interact':  lat_lon_inter,
        'rainy_season':      rainy_season,
        'is_flood':          is_flood,
        'is_conflict':       is_conflict,
        'year':              event.year,
    }

    df_input = pd.DataFrame([row])[FEATURES]

    # Predict with tuned threshold
    probability = float(model.predict_proba(df_input)[0][1])
    prediction  = 1 if probability >= THRESHOLD else 0

    severity, color, action = get_severity(prediction, probability)

    return {
        "prediction":  "Large Event" if prediction == 1 else "Small Event",
        "is_large":    prediction,
        "probability": round(probability, 3),
        "severity":    severity,
        "color":       color,
        "action":      action,
        "model_used":  model_key,
        "model_name":  MODEL_NAMES[model_key],
        "threshold":   THRESHOLD,
    }
