"""
Module: main.py
Description: FastAPI microservice for real-time and batch student dropout/failure risk prediction,
             serving the registered MLflow model with inference logging.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.dataset import FEATURE_COLUMNS

MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')
LOGS_DIR = os.path.join(BASE_DIR, 'data')
PREDICTION_LOG_PATH = os.path.join(LOGS_DIR, 'prediction_logs.jsonl')

# Initialize FastAPI App
app = FastAPI(
    title="Student Dropout & Failure Risk Prediction API",
    description="MLOps Microservice serving versioned ML models for early academic risk detection.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model_bundle = None


def get_model():
    """Load model bundle lazily or on startup."""
    global model_bundle
    if model_bundle is None:
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found at {BEST_MODEL_PATH}. Train models first.")
        model_bundle = joblib.load(BEST_MODEL_PATH)
    return model_bundle


# Request & Response Schemas
class StudentFeatureInput(BaseModel):
    # Demographics
    school: str = Field(default="GP", description="School ('GP' or 'MS')")
    sex: str = Field(default="F", description="Gender ('F' or 'M')")
    age: int = Field(default=17, ge=10, le=30, description="Age (10-30)")
    address: str = Field(default="U", description="Address ('U' for Urban, 'R' for Rural)")
    famsize: str = Field(default="GT3", description="Family Size ('LE3' or 'GT3')")
    pstatus: str = Field(default="T", description="Parent Status ('T' for Together, 'A' for Apart)")

    # Family & Social
    medu: int = Field(default=2, ge=0, le=4, description="Mother Education (0-4)")
    fedu: int = Field(default=2, ge=0, le=4, description="Father Education (0-4)")
    mjob: str = Field(default="other", description="Mother Job")
    fjob: str = Field(default="other", description="Father Job")
    reason: str = Field(default="course", description="Reason for choosing school")
    guardian: str = Field(default="mother", description="Guardian ('mother', 'father', 'other')")

    # Academic & Lifestyle
    traveltime: int = Field(default=1, ge=1, le=4, description="Travel Time (1-4)")
    studytime: int = Field(default=2, ge=1, le=4, description="Weekly Study Time (1-4)")
    failures: int = Field(default=0, ge=0, le=4, description="Past Failures (0-4)")
    schoolsup: str = Field(default="no", description="Extra Educational Support ('yes' or 'no')")
    famsup: str = Field(default="yes", description="Family Educational Support ('yes' or 'no')")
    paid: str = Field(default="no", description="Extra Paid Classes ('yes' or 'no')")
    activities: str = Field(default="no", description="Extracurricular Activities ('yes' or 'no')")
    nursery: str = Field(default="yes", description="Attended Nursery ('yes' or 'no')")
    higher: str = Field(default="yes", description="Wants Higher Education ('yes' or 'no')")
    internet: str = Field(default="yes", description="Internet Access ('yes' or 'no')")
    romantic: str = Field(default="no", description="In Romantic Relationship ('yes' or 'no')")
    famrel: int = Field(default=4, ge=1, le=5, description="Family Relationship (1-5)")
    freetime: int = Field(default=3, ge=1, le=5, description="Free Time (1-5)")
    goout: int = Field(default=3, ge=1, le=5, description="Going Out (1-5)")
    dalc: int = Field(default=1, ge=1, le=5, description="Workday Alcohol (1-5)")
    walc: int = Field(default=1, ge=1, le=5, description="Weekend Alcohol (1-5)")
    health: int = Field(default=3, ge=1, le=5, description="Current Health (1-5)")
    absences: int = Field(default=4, ge=0, le=93, description="Total Absences (0-93)")

    # Term Marks
    g1: int = Field(default=10, ge=0, le=20, description="Period 1 Grade (0-20)")
    g2: int = Field(default=10, ge=0, le=20, description="Period 2 Grade (0-20)")


class PredictionResponse(BaseModel):
    failure_risk_probability: float
    risk_category: str
    predicted_class: int
    recommendation: str
    top_risk_factors: List[str]
    model_version: str
    inference_latency_ms: float


class BatchPredictionRequest(BaseModel):
    students: List[StudentFeatureInput]


def log_prediction(input_data: dict, output_data: dict, latency_ms: float):
    """Asynchronously append inference requests to prediction log for drift monitoring."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "inputs": input_data,
        "outputs": output_data,
        "latency_ms": latency_ms
    }
    with open(PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.on_event("startup")
def startup_event():
    try:
        get_model()
        print("Model bundle loaded successfully on API startup.")
    except Exception as e:
        print(f"Warning: Model not loaded on startup: {e}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    bundle = get_model()
    return {
        "status": "HEALTHY",
        "timestamp": datetime.now().isoformat(),
        "model_name": bundle["model_name"],
        "target": bundle["target"],
        "trained_at": bundle["trained_at"]
    }


@app.get("/model-info")
def model_info():
    """Retrieve model metadata and training parameters."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    bundle = get_model()
    return {
        "model_name": bundle["model_name"],
        "features": bundle["features"],
        "trained_at": bundle["trained_at"]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(student: StudentFeatureInput):
    """
    Predicts dropout / failure risk for a single student profile.
    """
    start_time = time.time()
    bundle = get_model()
    clf = bundle["model"]
    preprocessor = bundle["preprocessor"]

    data_dict = student.model_dump()

    # Derived feature computations
    data_dict['internal_marks_avg'] = round((data_dict['g1'] + data_dict['g2']) / 2.0, 2)
    data_dict['estimated_attendance_pct'] = round(max(0.0, (1.0 - data_dict['absences'] / 93.0)) * 100.0, 2)
    data_dict['term1_to_term2_trend'] = round(data_dict['g2'] - data_dict['g1'], 2)

    df_in = pd.DataFrame([data_dict])[FEATURE_COLUMNS]

    # Preprocess & Predict
    X_proc = preprocessor.transform(df_in)
    proba = float(clf.predict_proba(X_proc)[0, 1]) if hasattr(clf, "predict_proba") else float(clf.predict(X_proc)[0])
    pred = int(proba >= 0.50)

    # Categorize Risk Tier
    if proba >= 0.65:
        risk_cat = "HIGH_RISK"
        rec = "High Risk Detected: Urgent intervention required. Schedule 1-on-1 tutoring and academic counseling."
    elif proba >= 0.35:
        risk_cat = "MEDIUM_RISK"
        rec = "Moderate Risk: Student requires monitoring. Recommend supplemental practice and study group participation."
    else:
        risk_cat = "LOW_RISK"
        rec = "Low Risk: Student is currently performing on track. Maintain standard curriculum schedule."

    # Identify Top Risk Contributing Factors
    risk_factors = []
    if data_dict['internal_marks_avg'] < 10:
        risk_factors.append(f"Low Internal Marks Average ({data_dict['internal_marks_avg']}/20)")
    if data_dict['absences'] > 10:
        risk_factors.append(f"High Absences ({data_dict['absences']} sessions)")
    if data_dict['failures'] > 0:
        risk_factors.append(f"Prior Academic Failures ({data_dict['failures']})")
    if data_dict['studytime'] == 1:
        risk_factors.append("Minimal Study Time (<2 hours weekly)")
    if data_dict['term1_to_term2_trend'] < 0:
        risk_factors.append(f"Declining Term Trajectory (G2 vs G1 dropped by {abs(data_dict['term1_to_term2_trend'])} pts)")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    response = {
        "failure_risk_probability": round(proba, 4),
        "risk_category": risk_cat,
        "predicted_class": pred,
        "recommendation": rec,
        "top_risk_factors": risk_factors if risk_factors else ["No significant academic risk flags detected"],
        "model_version": bundle.get("best_run_id", "v1.0")[:8],
        "inference_latency_ms": latency_ms
    }

    log_prediction(data_dict, response, latency_ms)
    return response


@app.post("/predict/batch")
def predict_batch(batch: BatchPredictionRequest):
    """Batch prediction endpoint for evaluating multiple student records."""
    results = []
    for item in batch.students:
        results.append(predict_risk(item))
    return {"total_evaluated": len(results), "predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
