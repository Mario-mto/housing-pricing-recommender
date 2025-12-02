from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "listings.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "pricing_model.joblib"

TARGET_COLUMN = "price"

NUMERIC_FEATURES = [
    "surface",
    "rooms",
    "bathrooms",
    "floor",
    "building_age",
]

CATEGORICAL_FEATURES = [
    "neighbourhood",
    "city",
    "furnished",
    "has_elevator",
]
