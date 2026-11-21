"""
Module: train.py
Description: Trains Baseline (Logistic Regression) and Advanced (Random Forest, XGBoost) models,
             tracks experiments in MLflow, evaluates performance, selects the best model,
             registers it in the MLflow Model Registry, and exports deployment artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ML and Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.dataset import get_stratified_splits, FEATURE_COLUMNS
from src.ml.preprocessing import fit_and_save_preprocessor, load_preprocessor

MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.joblib')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')
MLFLOW_DB_PATH = os.path.join(BASE_DIR, 'mlflow.db')

EXPERIMENT_NAME = "student_dropout_risk_prediction"
REGISTERED_MODEL_NAME = "student_dropout_risk_model"


def evaluate_model(model, X, y, split_name="val"):
    """Compute comprehensive classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        f"{split_name}_accuracy": float(accuracy_score(y, y_pred)),
        f"{split_name}_precision": float(precision_score(y, y_pred, zero_division=0)),
        f"{split_name}_recall": float(recall_score(y, y_pred, zero_division=0)),
        f"{split_name}_f1": float(f1_score(y, y_pred, zero_division=0)),
        f"{split_name}_roc_auc": float(roc_auc_score(y, y_proba))
    }
    return metrics, y_pred, y_proba


def train_and_evaluate_all():
    """Main training workflow with MLflow tracking and automated Model Registry."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Setup MLflow Tracking
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)

    # Automatically ensure experiment artifact_location matches current workspace if using existing SQLite DB
    if os.path.exists(MLFLOW_DB_PATH):
        try:
            import sqlite3
            mlruns_path = os.path.join(BASE_DIR, 'mlruns').replace(os.sep, '/')
            conn = sqlite3.connect(MLFLOW_DB_PATH)
            cur = conn.cursor()
            cur.execute("UPDATE experiments SET artifact_location = 'file:' || ? || '/' || experiment_id WHERE artifact_location NOT LIKE 'file:' || ? || '%'", (mlruns_path, mlruns_path))
            conn.commit()
            conn.close()
        except Exception:
            pass

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking initialized at: {tracking_uri}")

    # 2. Get Data Splits
    X_train, X_val, X_test, y_train, y_val, y_test = get_stratified_splits()

    # 3. Fit Preprocessor on Training Data
    preprocessor = fit_and_save_preprocessor(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Calculate class weight ratio for XGBoost (negative / positive)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)

    # 4. Define Candidate Model Suite
    models = {
        "LogisticRegression_Baseline": {
            "model": LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
            "params": {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "model_family": "Linear_Baseline"},
            "flavor": "sklearn"
        },
        "RandomForest_Classifier": {
            "model": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, class_weight='balanced', random_state=42),
            "params": {"n_estimators": 150, "max_depth": 6, "min_samples_split": 4, "class_weight": "balanced", "model_family": "Ensemble_Tree"},
            "flavor": "sklearn"
        },
        "XGBoost_Classifier": {
            "model": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42),
            "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "scale_pos_weight": float(scale_pos_weight), "model_family": "Gradient_Boosting"},
            "flavor": "xgboost"
        }
    }

    results = []
    best_val_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    best_run_id = None

    print("\n" + "=" * 80)
    print("STARTING MODEL TRAINING & MLFLOW EXPERIMENT BENCHMARKING")
    print("=" * 80)

    for name, config in models.items():
        clf = config["model"]
        params = config["params"]

        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id
            print(f"\n--- Training {name} (Run ID: {run_id[:8]}) ---")

            # Fit model
            clf.fit(X_train_proc, y_train)

            # Evaluate on Train, Val, Test
            train_metrics, _, _ = evaluate_model(clf, X_train_proc, y_train, "train")
            val_metrics, y_val_pred, _ = evaluate_model(clf, X_val_proc, y_val, "val")
            test_metrics, _, _ = evaluate_model(clf, X_test_proc, y_test, "test")

            all_metrics = {**train_metrics, **val_metrics, **test_metrics}

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(all_metrics)

            if config["flavor"] == "sklearn":
                mlflow.sklearn.log_model(clf, artifact_path="model", serialization_format="cloudpickle")
            else:
                mlflow.xgboost.log_model(clf, artifact_path="model")

            # Store result
            res = {
                "name": name,
                "run_id": run_id,
                "val_f1": val_metrics["val_f1"],
                "val_recall": val_metrics["val_recall"],
                "val_precision": val_metrics["val_precision"],
                "val_roc_auc": val_metrics["val_roc_auc"],
                "test_f1": test_metrics["test_f1"],
                "test_recall": test_metrics["test_recall"],
                "test_roc_auc": test_metrics["test_roc_auc"]
            }
            results.append(res)

            print(f"Validation F1: {val_metrics['val_f1']:.4f} | Recall: {val_metrics['val_recall']:.4f} | ROC-AUC: {val_metrics['val_roc_auc']:.4f}")
            print(f"Test F1:       {test_metrics['test_f1']:.4f} | Recall: {test_metrics['test_recall']:.4f} | ROC-AUC: {test_metrics['test_roc_auc']:.4f}")

            # Check if this is current champion (Primary selection: Validation F1-Score & Recall)
            if val_metrics["val_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                best_model_name = name
                best_model_obj = clf
                best_run_id = run_id

    # 5. Model Comparison Summary
    df_results = pd.DataFrame(results).sort_values(by="val_f1", ascending=False)
    print("\n" + "=" * 80)
    print("MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # 6. Save & Register Champion Model
    print("\n" + "=" * 80)
    print(f"[CHAMPION MODEL SELECTED] {best_model_name} (Validation F1: {best_val_f1:.4f})")
    print("=" * 80)

    # Save champion model artifact for standalone microservice inference
    joblib.dump({
        "model_name": best_model_name,
        "model": best_model_obj,
        "preprocessor": preprocessor,
        "features": FEATURE_COLUMNS,
        "target": "failure_risk",
        "trained_at": datetime.now().isoformat(),
        "best_run_id": best_run_id
    }, BEST_MODEL_PATH)
    print(f"Champion model bundle saved to {BEST_MODEL_PATH}")

    # Register in MLflow Model Registry
    try:
        model_uri = f"runs:/{best_run_id}/model"
        registered_model = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        print(f"Registered model '{REGISTERED_MODEL_NAME}' Version {registered_model.version} in MLflow Registry.")
    except Exception as e:
        print(f"MLflow model registration note: {e}")

    # 7. Write DVC / Git Versioning Metadata Manifest
    metadata = {
        "model_name": best_model_name,
        "mlflow_experiment": EXPERIMENT_NAME,
        "mlflow_run_id": best_run_id,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "features_count": len(FEATURE_COLUMNS),
        "target_definition": "failure_risk (G3 < 10)",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "benchmark_summary": results
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model versioning metadata manifest saved to {METADATA_PATH}")

    return df_results, best_model_name


if __name__ == "__main__":
    train_and_evaluate_all()
