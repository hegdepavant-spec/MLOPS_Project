import os
import time
import json
import pickle
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "data", "prediction_logs.jsonl")

# Initialize FastAPI App
app = FastAPI(
    title="Student Performance Risk Prediction API",
    description="Microservice to predict academic performance risk of students using Scikit-Learn / XGBoost models.",
    version="1.0.0"
)

# Enable CORS for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference
model = None

def load_model():
    """Load the serialized best model from disk."""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            model = None
    else:
        print(f"Model file not found at {MODEL_PATH}. Prediction service will run in Mock Mode.")
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()
    # Ensure data directory exists
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# Input Pydantic Schema
class PredictionRequest(BaseModel):
    study_time: int = Field(..., ge=1, le=4, description="Weekly study time tier (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    previous_failures: int = Field(..., ge=0, le=4, description="Number of past class failures")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences")

# Output Pydantic Schema
class PredictionResponse(BaseModel):
    risk: str = Field(..., description="Predicted risk classification ('High Risk' or 'Low Risk')")
    probability: float = Field(..., description="Prediction probability confidence (0.0 to 1.0)")

def log_prediction(study_time, failures, absences, risk, probability, latency_ms, error_msg=""):
    """Write prediction request and response metadata to JSONL file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "study_time": study_time,
        "previous_failures": failures,
        "absences": absences,
        "risk": risk,
        "probability": probability,
        "latency_ms": round(latency_ms, 2),
        "error": error_msg
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Failed to write prediction log: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict performance risk (High Risk / Low Risk) based on student features."""
    start_time = time.time()
    
    # Reload model if it failed during startup
    if model is None:
        load_model()
        
    try:
        # Mock mode fallback if model is not trained yet
        if model is None:
            # simple rule-based mock model for fallback
            # high absences + high failures = high risk
            if request.previous_failures > 0 or request.absences > 12:
                risk = "High Risk"
                probability = 0.75 + (0.05 * request.previous_failures)
            else:
                risk = "Low Risk"
                probability = 0.85 - (0.02 * request.absences)
                
            latency_ms = (time.time() - start_time) * 1000
            log_prediction(
                request.study_time, request.previous_failures, request.absences,
                risk, probability, latency_ms, error_msg="Mock mode active"
            )
            return {"risk": risk, "probability": round(probability, 4)}
            
        # Predict using model
        # Input features must be ordered as: ['study_time', 'previous_failures', 'absences']
        features = np.array([[request.study_time, request.previous_failures, request.absences]])
        
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        
        # 1 maps to High Risk, 0 maps to Low Risk
        risk = "High Risk" if prediction == 1 else "Low Risk"
        probability = float(probabilities[prediction])
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log success
        log_prediction(
            request.study_time, request.previous_failures, request.absences,
            risk, probability, latency_ms
        )
        
        return {"risk": risk, "probability": round(probability, 4)}
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        log_prediction(
            request.study_time, request.previous_failures, request.absences,
            "Error", 0.0, latency_ms, error_msg=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Get service metrics and prediction log history for monitoring dashboard."""
    if not os.path.exists(LOG_PATH):
        return {
            "total_requests": 0,
            "error_count": 0,
            "average_latency_ms": 0.0,
            "class_distribution": {"High Risk": 0, "Low Risk": 0},
            "recent_logs": []
        }
        
    logs = []
    error_count = 0
    latency_sum = 0.0
    class_dist = {"High Risk": 0, "Low Risk": 0}
    
    try:
        with open(LOG_PATH, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                logs.append(entry)
                
                if entry.get("error"):
                    error_count += 1
                else:
                    risk = entry.get("risk")
                    if risk in class_dist:
                        class_dist[risk] += 1
                
                latency_sum += entry.get("latency_ms", 0.0)
                
        total_requests = len(logs)
        avg_latency = (latency_sum / total_requests) if total_requests > 0 else 0.0
        
        return {
            "total_requests": total_requests,
            "error_count": error_count,
            "average_latency_ms": round(avg_latency, 2),
            "class_distribution": class_dist,
            "recent_logs": logs[-100:] # Return last 100 predictions for visual plots
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile metrics: {str(e)}")
