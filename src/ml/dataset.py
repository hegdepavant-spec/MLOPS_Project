"""
Module: dataset.py
Description: Extracts features and target from the analytical data mart
             and performs stratified Train / Validation / Test splitting.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database import fetch_analytical_data

DATA_DIR = os.path.join(BASE_DIR, 'data')
ML_DATA_DIR = os.path.join(DATA_DIR, 'ml')

# Pre-prediction Feature Set (Strictly preventing target leakage from G3)
FEATURE_COLUMNS = [
    # Demographics
    'school', 'sex', 'age', 'address', 'famsize', 'pstatus',
    # Family & Social
    'medu', 'fedu', 'mjob', 'fjob', 'reason', 'guardian',
    # Academic Habits & Prior History
    'traveltime', 'studytime', 'failures',
    # Academic Support
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
    # Behavioral & Health
    'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health', 'absences',
    # Prior Internal Evaluations
    'g1', 'g2', 'internal_marks_avg', 'estimated_attendance_pct', 'term1_to_term2_trend'
]

TARGET_COLUMN = 'failure_risk'  # 1 = Failure (G3 < 10), 0 = Pass (G3 >= 10)


def load_ml_dataset():
    """
    Loads analytical data from PostgreSQL/SQLite, constructs binary target,
    and isolates features.
    """
    os.makedirs(ML_DATA_DIR, exist_ok=True)
    df = fetch_analytical_data()
    
    if df.empty:
        # Fallback to CSV
        csv_path = os.path.join(DATA_DIR, 'analytics', 'analytics_student_performance.csv')
        df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Construct Binary Prediction Target: Failure Risk (1 = At-Risk / Failed with G3 < 10, 0 = Pass with G3 >= 10)
    # G3 is strictly isolated and used ONLY as the target ground truth.
    df[TARGET_COLUMN] = (df['g3'] < 10).astype(int)

    # Validate all requested feature columns exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in analytical mart: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y, df


def get_stratified_splits(test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """
    Creates stratified Train (70%), Validation (15%), and Test (15%) datasets.
    Stratification ensures balanced representation of failure risk across all splits.
    """
    X, y, full_df = load_ml_dataset()

    # Step 1: Split into Train+Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 2: Split Train+Val into Train (70% total) and Val (15% total)
    # val_ratio relative to Train+Val = 0.15 / (1.0 - 0.15) = 0.1765
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
    )

    # Save splits to data/ml/ for auditability & DVC tracking
    X_train.assign(failure_risk=y_train).to_csv(os.path.join(ML_DATA_DIR, 'train.csv'), index=False)
    X_val.assign(failure_risk=y_val).to_csv(os.path.join(ML_DATA_DIR, 'val.csv'), index=False)
    X_test.assign(failure_risk=y_test).to_csv(os.path.join(ML_DATA_DIR, 'test.csv'), index=False)

    print(f"Dataset Split Summary: Train={len(X_train)} ({y_train.mean()*100:.1f}% risk), "
          f"Val={len(X_val)} ({y_val.mean()*100:.1f}% risk), "
          f"Test={len(X_test)} ({y_test.mean()*100:.1f}% risk)")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_stratified_splits()
