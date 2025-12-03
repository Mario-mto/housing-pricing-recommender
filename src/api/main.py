# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

import pandas as pd
import logging
from pathlib import Path    

from src.ml.predict import load_trained_model

# -----------------------------
# 0. Constante de dispersion
# -----------------------------
# On utilise le RMSE observé (~80) comme "rayon" de la fourchette de prix.
ESTIMATED_RMSE = 80.0

# -----------------------------
# 0. Logging
# -----------------------------

# Dossier des logs à la racine du projet
ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "api.log"

logger = logging.getLogger("pricing_api")
logger.setLevel(logging.INFO)

# Éviter d'ajouter plusieurs handlers si l'app est rechargée (uvicorn --reload)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
    Réponse de l'API : prix prédit + fourchette estimée.
    """
    predicted_price: float
    low_range: float
    high_range: float


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
    logger.info("Health check called")
    return {"status": "ok"}


@app.post("/predict-price", response_model=PricePrediction)
def predict_price_endpoint(housing: HousingInput):
    """
    Endpoint principal de prédiction.
    - Reçoit un logement (HousingInput)
    - Transforme en DataFrame
    - Passe dans le pipeline ML
    - Retourne le prix prédit + une fourchette
    """

    # 1) On transforme l'objet Pydantic en dict
    data_dict = housing.dict()
    logger.info(f"New prediction request: {data_dict}")

    # 2) On crée un DataFrame avec une seule ligne
    df = pd.DataFrame([data_dict])

    try:
        # 3) On appelle le pipeline ML
        price_pred = float(model.predict(df)[0])
    except Exception as e:
        logger.exception(f"Error during prediction for data: {data_dict}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

    # 4) On calcule une fourchette autour du prix
    low = max(price_pred - ESTIMATED_RMSE, 0.0)  # prix minimal plausible
    high = price_pred + ESTIMATED_RMSE

    logger.info(
        f"Prediction result: price={price_pred:.2f}, "
        f"low={low:.2f}, high={high:.2f}"
    )

    # 5) On retourne la réponse avec le schéma Pydantic
    return PricePrediction(
        predicted_price=price_pred,
        low_range=low,
        high_range=high,
    )

