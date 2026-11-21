"""
Module: preprocessing.py
Description: Scikit-learn preprocessing pipeline with ColumnTransformer,
             StandardScaler, and OneHotEncoder.
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.joblib')

NUMERICAL_FEATURES = [
    'age', 'medu', 'fedu', 'traveltime', 'studytime', 'failures',
    'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health', 'absences',
    'g1', 'g2', 'internal_marks_avg', 'estimated_attendance_pct', 'term1_to_term2_trend'
]

CATEGORICAL_FEATURES = [
    'school', 'sex', 'address', 'famsize', 'pstatus',
    'mjob', 'fjob', 'reason', 'guardian',
    'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]


def build_preprocessor():
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def fit_and_save_preprocessor(X_train: pd.DataFrame):
    """Fits the preprocessor on training data and persists it to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Fitted preprocessor saved to {PREPROCESSOR_PATH}")
    return preprocessor


def load_preprocessor():
    """Loads the persisted preprocessor artifact."""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor artifact not found at {PREPROCESSOR_PATH}. Run training first.")
    return joblib.load(PREPROCESSOR_PATH)


if __name__ == "__main__":
    from src.ml.dataset import get_stratified_splits
    X_train, _, _, _, _, _ = get_stratified_splits()
    fit_and_save_preprocessor(X_train)
