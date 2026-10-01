import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Model Registry Alias Name
MODEL_URI = "models:/house-price-predictor@champion"

FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]

# Set MLflow tracking server
mlflow.set_tracking_uri(TRACKING_URI)

# Load model from registry
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    model = None

app = FastAPI(title="House Price Predictor")


class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=20)
    age_years: int = Field(..., ge=0, le=200)  # Corrected range
    garage: int = Field(..., ge=0, le=10)       # Added missing feature
    location_score: float = Field(..., ge=1, le=10)


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_uri": MODEL_URI
    }


@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None:
        return {"error": "Model is not loaded properly"}
    
    # Convert incoming JSON payload to DataFrame using exact features order
    input_df = pd.DataFrame([features.model_dump()])[FEATURES]
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}


# Static files setup for UI
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def frontend():
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "UI index.html file not found in static folder"}