# src/api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

import pandas as pd

from src.ml.predict import load_trained_model

# -----------------------------
# 1. Création de l'app FastAPI
# -----------------------------
app = FastAPI(
    title="Housing Pricing Recommender API",
    description="API pour prédire le prix optimal d'un logement à partir d'un modèle ML.",
    version="1.0.0",
)


# ----------------------------------------
# 2. Schémas Pydantic (entrée / sortie)
# ----------------------------------------

class HousingInput(BaseModel):
    """
    Représente les caractéristiques d'un logement envoyées à l'API.
    Les champs doivent correspondre aux colonnes utilisées par ton modèle.
    """
    surface: float
    rooms: int
    bathrooms: int
    floor: int
    building_age: int

    neighbourhood: str
    city: str

    # On peut garder des str simples pour yes/no
    furnished: Literal["yes", "no"]
    has_elevator: Literal["yes", "no"]


class PricePrediction(BaseModel):
    """
    Réponse de l'API : prix prédit (pour l'instant un seul nombre).
    Tu pourras ajouter low/high range plus tard.
    """
    predicted_price: float


# ----------------------------------------
# 3. Chargement du modèle au démarrage
# ----------------------------------------

# On charge le pipeline (preprocessing + modèle) UNE seule fois
# pour éviter de le recharger à chaque requête.
model = load_trained_model()


# ----------------------------------------
# 4. Endpoints
# ----------------------------------------

@app.get("/health")
def health_check():
    """
    Endpoint de santé très simple pour vérifier que l'API tourne.
    """
    return {"status": "ok"}


@app.post("/predict-price", response_model=PricePrediction)
def predict_price_endpoint(housing: HousingInput):
    """
    Endpoint principal de prédiction.
    - Reçoit un logement (HousingInput)
    - Transforme en DataFrame
    - Passe dans le pipeline ML
    - Retourne le prix prédit
    """

    # 1) On transforme l'objet Pydantic en dict
    data_dict = housing.dict()

    # 2) On crée un DataFrame avec une seule ligne
    df = pd.DataFrame([data_dict])

    # 3) On appelle le pipeline ML
    price_pred = model.predict(df)[0]

    # 4) On retourne la réponse avec le schéma Pydantic
    return PricePrediction(predicted_price=float(price_pred))
