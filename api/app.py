from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Load all 3 models + shared artifacts
models = {
    "lr":  joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl")),
    "rf":  joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl")),
    "xgb": joblib.load(os.path.join(MODELS_DIR, "best_model.pkl")),  # best = xgb
}

scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))

le_combined = encoders["combined_type"]
le_displace = encoders["displacement_type"]
le_region   = encoders["region"]

# Metrics — update these after running 02_modeling.ipynb
METRICS = {
    "lr":  {"recall": 0.0, "accuracy": 0.0, "f1": 0.0},
    "rf":  {"recall": 0.0, "accuracy": 0.0, "f1": 0.0},
    "xgb": {"recall": 0.0, "accuracy": 0.0, "f1": 0.0},
}

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="Somalia Displacement Severity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/ui",
    StaticFiles(directory=BASE_DIR, html=True),
    name="ui"
)

# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────
class EventInput(BaseModel):
    combined_type:     str
    displacement_type: str
    region:            str
    month:             int
    duration_days:     float
    latitude:          float
    longitude:         float
    year:              int
    model:             Optional[str] = "xgb"   # which model to use

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def safe_encode(encoder, value, fallback=0):
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return fallback

def get_severity(prediction, probability):
    if prediction == 1:
        if probability >= 0.75:
            return "Critical", "critical", "Immediate deployment required — pre-position emergency shelter, food, and medical teams now."
        return "High", "high", "High risk — alert field teams and prepare logistics for rapid response."
    else:
        if probability <= 0.30:
            return "Low", "low", "Low risk — monitor the situation and keep standard readiness."
        return "Moderate", "moderate", "Moderate risk — continue monitoring and keep response teams on standby."

FEATURE_NAMES = [
    'combined_type_enc', 'displacement_type_enc', 'region_enc',
    'month', 'duration_days', 'latitude', 'longitude', 'year'
]
SCALE_COLS = ['month', 'duration_days', 'latitude', 'longitude', 'year']

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Somalia Displacement Severity API"}

@app.get("/metrics")
def metrics():
    return METRICS

@app.post("/predict")
def predict(event: EventInput):
    # Pick model
    model_key = event.model if event.model in models else "xgb"
    model     = models[model_key]

    # Encode
    combined_enc = safe_encode(le_combined, event.combined_type)
    displace_enc = safe_encode(le_displace, event.displacement_type)
    region_enc   = safe_encode(le_region,   event.region)

    # Build dataframe
    df_input = pd.DataFrame([[
        combined_enc, displace_enc, region_enc,
        event.month, event.duration_days,
        event.latitude, event.longitude, event.year
    ]], columns=FEATURE_NAMES)

    # Scale continuous cols
    df_input[SCALE_COLS] = scaler.transform(df_input[SCALE_COLS])

    # Predict
    prediction  = int(model.predict(df_input)[0])
    probability = float(model.predict_proba(df_input)[0][1])

    severity, color, action = get_severity(prediction, probability)

    return {
        "prediction":  "Large Event" if prediction == 1 else "Small Event",
        "is_large":    prediction,
        "probability": round(probability, 3),
        "severity":    severity,
        "color":       color,
        "action":      action,
        "model_used":  model_key,
    }
