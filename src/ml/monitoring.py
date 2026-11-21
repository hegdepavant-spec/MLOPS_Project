"""
Module: monitoring.py
Description: MLOps Continuous Monitoring Engine.
             Analyzes covariate data drift (KS-Test), prediction distribution shifts,
             API latency, and evaluates quantitative model retraining triggers.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.dataset import load_ml_dataset, FEATURE_COLUMNS

LOGS_DIR = os.path.join(BASE_DIR, 'data')
PREDICTION_LOG_PATH = os.path.join(LOGS_DIR, 'prediction_logs.jsonl')
MONITORING_REPORT_PATH = os.path.join(LOGS_DIR, 'monitoring_report.json')

# Retraining Policy Thresholds
KS_PVALUE_DRIFT_THRESHOLD = 0.05       # p < 0.05 indicates statistically significant distribution shift
CLASS_SHIFT_THRESHOLD_PCT = 15.0       # > 15% change in predicted failure class proportion
LATENCY_SLA_MS = 250.0                 # Max acceptable inference latency


def load_baseline_training_data():
    """Load the reference baseline training dataset."""
    train_path = os.path.join(BASE_DIR, 'data', 'ml', 'train.csv')
    if os.path.exists(train_path):
        return pd.read_csv(train_path)
    X, y, _ = load_ml_dataset()
    return X.assign(failure_risk=y)


def load_production_logs():
    """Load production inference logs from JSONL."""
    if not os.path.exists(PREDICTION_LOG_PATH):
        return pd.DataFrame()

    records = []
    with open(PREDICTION_LOG_PATH, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        return pd.DataFrame()

    # Flatten inputs, outputs, latency
    flat = []
    for r in records:
        row = {**r.get('inputs', {}), **r.get('outputs', {})}
        row['timestamp'] = r.get('timestamp')
        row['latency_ms'] = r.get('latency_ms', 0.0)
        flat.append(row)

    return pd.DataFrame(flat)


def analyze_numerical_drift(baseline_df: pd.DataFrame, prod_df: pd.DataFrame):
    """
    Performs 2-sample Kolmogorov-Smirnov (KS) test for numerical feature distributions.
    p-value < 0.05 rejects null hypothesis (indicating statistically significant drift).
    """
    numeric_cols = ['g1', 'g2', 'internal_marks_avg', 'absences', 'studytime', 'failures', 'estimated_attendance_pct']
    drift_results = {}

    for col in numeric_cols:
        if col in baseline_df.columns and col in prod_df.columns and len(prod_df[col].dropna()) >= 5:
            base_vals = baseline_df[col].dropna()
            prod_vals = prod_df[col].dropna()

            ks_stat, p_val = stats.ks_2samp(base_vals, prod_vals)
            has_drifted = bool(p_val < KS_PVALUE_DRIFT_THRESHOLD)

            drift_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 4),
                "drift_detected": has_drifted,
                "baseline_mean": round(float(base_vals.mean()), 2),
                "production_mean": round(float(prod_vals.mean()), 2)
            }
        else:
            drift_results[col] = {"status": "INSUFFICIENT_PROD_SAMPLES"}

    return drift_results


def run_mlops_monitoring():
    """
    Executes full continuous monitoring routine and outputs quantitative report.
    """
    print("=" * 80)
    print("RUNNING MLOPS DATA DRIFT & MODEL MONITORING ENGINE")
    print("=" * 80)

    baseline_df = load_baseline_training_data()
    prod_df = load_production_logs()

    total_inferences = len(prod_df)
    print(f"Loaded {len(baseline_df)} baseline samples and {total_inferences} production inference requests.")

    # 1. API Performance & Latency Metrics
    avg_latency = float(prod_df['latency_ms'].mean()) if not prod_df.empty else 0.0
    p95_latency = float(np.percentile(prod_df['latency_ms'].dropna(), 95)) if not prod_df.empty else 0.0

    # 2. Class Distribution Shift
    baseline_failure_rate = float(baseline_df['failure_risk'].mean() * 100.0) if 'failure_risk' in baseline_df.columns else 33.0
    prod_failure_rate = float(prod_df['predicted_class'].mean() * 100.0) if ('predicted_class' in prod_df.columns and total_inferences > 0) else baseline_failure_rate
    class_shift = abs(prod_failure_rate - baseline_failure_rate)

    # 3. Covariate Data Drift Analysis
    drift_report = analyze_numerical_drift(baseline_df, prod_df)
    drifted_features_count = sum(1 for v in drift_report.values() if isinstance(v, dict) and v.get("drift_detected", False))

    # 4. Retraining Criteria Decision Logic
    retrain_reasons = []
    if drifted_features_count >= 2:
        retrain_reasons.append(f"Significant covariate drift detected in {drifted_features_count} key features (p < {KS_PVALUE_DRIFT_THRESHOLD})")
    if class_shift > CLASS_SHIFT_THRESHOLD_PCT and total_inferences >= 20:
        retrain_reasons.append(f"Class distribution shifted by {class_shift:.1f}% (Threshold: {CLASS_SHIFT_THRESHOLD_PCT}%)")
    if total_inferences >= 100:
        retrain_reasons.append(f"New labeled production volume threshold reached ({total_inferences} >= 100 samples)")

    should_retrain = len(retrain_reasons) > 0

    monitoring_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_production_inferences": total_inferences,
        "api_latency_ms": {
            "mean": round(avg_latency, 2),
            "p95": round(p95_latency, 2),
            "sla_met": bool(p95_latency < LATENCY_SLA_MS)
        },
        "class_distribution": {
            "baseline_failure_rate_pct": round(baseline_failure_rate, 2),
            "production_failure_rate_pct": round(prod_failure_rate, 2),
            "absolute_shift_pct": round(class_shift, 2)
        },
        "feature_drift_analysis": drift_report,
        "drifted_features_count": drifted_features_count,
        "retraining_decision": {
            "retrain_triggered": should_retrain,
            "decision_status": "RETRAIN_RECOMMENDED" if should_retrain else "SYSTEM_HEALTHY",
            "trigger_reasons": retrain_reasons if should_retrain else ["All monitoring metrics within SLA bounds"]
        }
    }

    with open(MONITORING_REPORT_PATH, 'w') as f:
        json.dump(monitoring_summary, f, indent=4)

    print("\n--- Monitoring Summary ---")
    print(f"Total Requests: {total_inferences} | Mean Latency: {avg_latency:.2f}ms")
    print(f"Baseline Failure Rate: {baseline_failure_rate:.1f}% | Prod Predicted Failure Rate: {prod_failure_rate:.1f}%")
    print(f"Drifted Features Count: {drifted_features_count}")
    print(f"Retrain Trigger Decision: {'[RETRAIN REQUIRED]' if should_retrain else '[SYSTEM HEALTHY]'}")
    print(f"Report saved to {MONITORING_REPORT_PATH}")

    return monitoring_summary


if __name__ == "__main__":
    run_mlops_monitoring()
