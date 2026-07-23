# config.py — shared constants for the backend

FEATURE_NAMES = [
    'combined_type_enc',
    'displacement_type_enc',
    'region_enc',
    'month',
    'duration_days',
    'latitude',
    'longitude',
    'year',
]

SCALE_COLS = ['month', 'duration_days', 'latitude', 'longitude', 'year']

MODEL_NAMES = {
    "lr":  "Logistic Regression",
    "rf":  "Random Forest",
    "xgb": "XGBoost",
}
