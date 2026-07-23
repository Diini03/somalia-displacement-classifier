from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

from config import MODELS_DIR, FEATURE_NAMES, SCALE_COLS
from utils import safe_encode, get_severity, get_model_name

# ─────────────────────────────────────────────
# Load artifacts — all paths relative to this file
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS  = os.path.join(BASE_DIR, "models")

models = {
    "lr":  joblib.load(os.path.join(_MODELS, "lr_model.pkl")),
    "rf":  joblib.load(os.path.join(_MODELS, "rf_model.pkl")),
    "xgb": joblib.load(os.path.join(_MODELS, "best_model.pkl")),
}

scaler   = joblib.load(os.path.join(_MODELS, "scaler.pkl"))
encoders = joblib.load(os.path.join(_MODELS, "label_encoders.pkl"))

le_combined = encoders["combined_type"]
le_displace = encoders["displacement_type"]
le_region   = encoders["region"]

# Update these after running 02_modeling.ipynb
METRICS = {
    "lr":  {"name": "Logistic Regression", "recall": 0.0, "accuracy": 0.0, "f1": 0.0},
    "rf":  {"name": "Random Forest",       "recall": 0.0, "accuracy": 0.0, "f1": 0.0},
    "xgb": {"name": "XGBoost",             "recall": 0.0, "accuracy": 0.0, "f1": 0.0},
}

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="Somalia Displacement Severity API", version="1.0.0")

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
    combined_type:     str
    displacement_type: str
    region:            str
    month:             int
    duration_days:     float
    latitude:          float
    longitude:         float
    year:              int
    model:             Optional[str] = "xgb"

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Somalia Displacement Severity API v1.0"}

@app.get("/metrics")
def metrics():
    return METRICS

@app.post("/predict")
def predict(event: EventInput):
    model_key = event.model if event.model in models else "xgb"
    model     = models[model_key]

    combined_enc = safe_encode(le_combined, event.combined_type)
    displace_enc = safe_encode(le_displace, event.displacement_type)
    region_enc   = safe_encode(le_region,   event.region)

    df_input = pd.DataFrame([[
        combined_enc, displace_enc, region_enc,
        event.month, event.duration_days,
        event.latitude, event.longitude, event.year
    ]], columns=FEATURE_NAMES)

    df_input[SCALE_COLS] = scaler.transform(df_input[SCALE_COLS])

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
        "model_name":  get_model_name(model_key),
    }
