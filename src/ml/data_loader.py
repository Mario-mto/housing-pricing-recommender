# src/ml/data_loader.py

import pandas as pd
from .config import RAW_DATA_PATH, TARGET_COLUMN

def load_data():
    """
    Charge les données brutes depuis le CSV et retourne:
    - X : features
    - y : cible (prix)
    """
    df = pd.read_csv(RAW_DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET_COLUMN}' introuvable dans le CSV.")

    # On sépare X et y
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y
