# src/ml/predict.py

import joblib
import pandas as pd

from .config import MODEL_PATH


def load_trained_model():
    """
    Charge le pipeline entraîné (preprocessing + modèle).
    """
    model = joblib.load(MODEL_PATH)
    return model


def predict_price(sample: dict) -> float:
    """
    Prend un dictionnaire représentant un logement
    et retourne le prix prédit.
    
    Exemple de sample:
    {
        "surface": 45,
        "rooms": 2,
        "bathrooms": 1,
        "floor": 3,
        "building_age": 15,
        "neighbourhood": "Plateau",
        "city": "Montreal",
        "furnished": "yes",
        "has_elevator": "no"
    }
    """
    model = load_trained_model()

    # On convertit le dict en DataFrame (1 ligne)
    df = pd.DataFrame([sample])

    price_pred = model.predict(df)[0]
    return float(price_pred)


if __name__ == "__main__":
    # Petit test manuel
    example = {
        "surface": 45,
        "rooms": 2,
        "bathrooms": 1,
        "floor": 3,
        "building_age": 15,
        "neighbourhood": "Plateau",
        "city": "Montreal",
        "furnished": "yes",
        "has_elevator": "no"
    }

    print("Prévision de prix:", predict_price(example))
