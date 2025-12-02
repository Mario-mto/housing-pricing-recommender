# src/ml/config.py

from pathlib import Path

# Chemin racine du projet (remonte depuis ce fichier)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "listings.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "pricing_model.joblib"

# Nom de la colonne cible (prix)
TARGET_COLUMN = "price"

# Liste des features (à adapter à TES colonnes)
NUMERIC_FEATURES = [
    "surface",
    "rooms",
    "bathrooms",
    "floor",
    "building_age",
    "distance_metro",
    "price_per_m2_neighbourhood",
]

CATEGORICAL_FEATURES = [
    "neighbourhood",
    "city",
    "furnished",
    "has_elevator",
    "season",   # ex: 'winter', 'summer'
]

# Paramètres de base du modèle (RandomForest pour commencer)
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": 42,
}
