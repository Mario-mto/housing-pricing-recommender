# src/ml/model_selection.py

"""
Ce module définit les grilles d'hyperparamètres
utilisées par GridSearchCV pour RandomForest et XGBoost.

ATTENTION:
- Pour RandomForest, le modèle est directement dans la step 'model'
  du pipeline, donc on utilise 'model__param'.
- Pour XGBoost, on l'a enveloppé dans un TransformedTargetRegressor,
  donc le XGBRegressor est dans 'model.regressor' :
  => on utilise 'model__regressor__param'.
"""


def optimize_random_forest():
    """
    Grille d'hyperparamètres pour RandomForestRegressor.
    Utilisée quand model_type='rf' et use_grid=True dans train.py.
    """
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    return param_grid


def optimize_xgboost():
    """
    Grille d'hyperparamètres pour XGBRegressor
    quand il est enveloppé dans un TransformedTargetRegressor
    dans la step 'model' du pipeline.
    """

    param_grid = {
        "model__regressor__n_estimators": [100, 200, 300],
        "model__regressor__max_depth": [3, 5, 7],
        "model__regressor__learning_rate": [0.01, 0.05, 0.1],
        "model__regressor__subsample": [0.7, 0.9, 1.0],
        "model__regressor__colsample_bytree": [0.7, 0.9, 1.0],
    }

    return param_grid

