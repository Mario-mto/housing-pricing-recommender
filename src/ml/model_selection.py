# src/ml/model_selection.py

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def optimize_random_forest():
    """
    Optimise un RandomForest avec GridSearchCV.
    Retourne le meilleur modèle + les meilleurs paramètres.
    """

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    return param_grid


def optimize_xgboost():
    """
    Optimise un XGBoostRegressor avec GridSearchCV.
    """

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    }

    return param_grid
