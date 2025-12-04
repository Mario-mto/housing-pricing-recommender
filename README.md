
# 🏡 Housing Pricing Recommender

### Machine Learning Price Recommendation Engine + FastAPI

---

## 📌 Project Overview

This project implements a **recommendation engine capable of predicting the optimal rental price of a housing unit**, based on features such as:

* surface area
* number of rooms
* neighbourhood
* building age
* furnishing conditions
* elevator availability
* and more

The system includes:

* a full **Machine Learning pipeline** (preprocessing, feature engineering, XGBoost tuning)
* a reproducible **training and evaluation framework**
* a **FastAPI service** exposing a `/predict-price` endpoint
* **JSON-structured logging** for monitoring and debugging
* a clean, modular project architecture ready for scaling and deployment

This project can be integrated within a **real estate CRM**, a mobile app, or a platform such as **Cribz**.

---

## 🎯 Technical Goals

* Build a robust ML model to estimate optimal rent prices
* Provide easy access via a REST API
* Ensure reproducibility through a clean pipeline and virtual environment
* Log all prediction requests and outputs for monitoring and auditing
* Follow professional-grade software engineering standards

---

## 🏗️ Project Structure

```
housing-pricing-recommender/
│
├── data/                  
│   └── raw/listings.csv
│
├── logs/                  # JSON-structured logs
│   └── api.log
│
├── models/                # Trained ML model
│   └── pricing_model.joblib
│
├── src/
│   ├── ml/
│   │   ├── config.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── create_fake_data.py
│   │
│   └── api/
│       ├── main.py        # FastAPI application
│       └── __init__.py
│
├── requirements.txt
└── README.md
```

---

## 🔧 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your_repo>/housing-pricing-recommender.git
cd housing-pricing-recommender
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate it

#### Windows:

```bash
venv\Scripts\activate
```

#### macOS / Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Train the Model

Generate synthetic training data:

```bash
python src/ml/create_fake_data.py
```

Train the model:

```bash
python -m src.ml.train
```

This produces:

* `models/pricing_model.joblib`
* metrics: **MAE**, **RMSE**, **R²**
* hyperparameter optimization results

---

## 🌐 Run the FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

Interactive documentation (Swagger UI):
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📬 Example API Request: `/predict-price`

### Request body

```json
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
```

### Response

```json
{
  "predicted_price": 1324.87,
  "low_range": 1244.87,
  "high_range": 1404.87
}
```

The prediction interval is based on the model RMSE.

---

## 📝 JSON-Structured Logging

Every request and prediction is logged in:

```
logs/api.log
```

Example entries:

```json
{
  "asctime": "2025-12-03 12:04:31",
  "levelname": "INFO",
  "event": "prediction_request",
  "input": { ... }
}
```

```json
{
  "asctime": "2025-12-03 12:04:31",
  "levelname": "INFO",
  "event": "prediction_result",
  "output": { ... }
}
```

This supports:

* monitoring
* analytics
* debugging
* user behavior insights
* model auditing

---

## 📚 Machine Learning Model

The final model is an **XGBoostRegressor** embedded within a sklearn Pipeline:

* one-hot encoding for categorical variables
* optimized hyperparameters with GridSearchCV
* custom training script
* model stored with joblib

Typical synthetic dataset performance (500 samples):

| Metric | Score  |
| ------ | ------ |
| MAE    | ~ 69   |
| RMSE   | ~ 80   |
| R²     | ~ 0.85 |

---

## 🚀 Next Improvements

Potential enhancements include:

* Real estate dataset ingestion (real-world data)
* SHAP-based explainability
* Dockerization & cloud deployment (Render / Railway / AWS)
* API Key authentication
* Model versioning (MLflow)
* A/B testing with multiple models
* Monitoring dashboards (ELK / Grafana)

---

## 👤 Author

**Mario Montcho**
Machine Learning • Software Engineering • Full Stack Development
Portfolio project — Housing price recommendation engine

---

## 📄 License

Free for educational and demonstration purposes.

---
