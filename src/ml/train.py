# src/ml/train.py

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import MODEL_DIR, MODEL_PATH, RANDOM_FOREST_PARAMS
from .data_loader import load_data
from .features import build_preprocessing_pipeline


def train_model(test_size: float = 0.2, random_state: int = 42):
    """
    Entraîne un modèle de régression (RandomForest) pour prédire le prix.
    - Split train/test
    - Entraînement
    - Évaluation (MAE, RMSE, R²)
    - Sauvegarde du pipeline complet (preprocessing + modèle)
    """
    # 1. Charger les données
    X, y = load_data()

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # 3. Pipeline de préprocessing
    preprocessor = build_preprocessing_pipeline()

    # 4. Modèle
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)

    # 5. Pipeline complet (preprocessing + modèle)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # 6. Entraînement
    clf.fit(X_train, y_train)

    # 7. Évaluation
    y_pred = clf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R²   (Coefficient de détermination): {r2:.3f}")

    # 8. Sauvegarde du modèle
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans: {MODEL_PATH}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


if __name__ == "__main__":
    # Permet de lancer l'entraînement avec:
    # python -m src.ml.train
    train_model()
