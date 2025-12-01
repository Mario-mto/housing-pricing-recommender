# src/ml/train.py

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from .config import MODEL_DIR, MODEL_PATH
from .data_loader import load_data
from .features import build_preprocessing_pipeline
from .model_selection import optimize_random_forest, optimize_xgboost


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

    return mae, rmse, r2


def train_model(use_grid=False, model_type="rf"):
    """
    Entraîne un modèle ML avec ou sans GridSearch.
    model_type : "rf" ou "xgb"
    """

    # 1. Charger données
    X, y = load_data()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Pipeline de préprocessing
    preprocessor = build_preprocessing_pipeline()

    # 4. Choix du modèle
    if model_type == "rf":
        model = RandomForestRegressor(random_state=42)
        param_grid = optimize_random_forest() if use_grid else None

    elif model_type == "xgb":
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
        )
        param_grid = optimize_xgboost() if use_grid else None

    else:
        raise ValueError("model_type doit être 'rf' ou 'xgb'.")

    # 5. Pipeline complet (preprocess + modèle)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # 6. Si GridSearch
    if use_grid:
        print("🔍 Optimisation des hyperparamètres avec GridSearchCV...")
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("🔥 Meilleurs paramètres :", grid.best_params_)
    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    # 7. Prédictions
    y_pred = best_model.predict(X_test)

    # 8. Évaluer
    mae, rmse, r2 = evaluate_model(y_test, y_pred)

    # 9. Sauvegarde modèle
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans : {MODEL_PATH}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


if __name__ == "__main__":
    # Exemple 1 : RandomForest normal
    # train_model(use_grid=False, model_type="rf")

    # Exemple 2 : RandomForest avec GridSearch
    # train_model(use_grid=True, model_type="rf")

    # Exemple 3 : XGBoost avec GridSearch (recommandé)
    train_model(use_grid=True, model_type="xgb")
