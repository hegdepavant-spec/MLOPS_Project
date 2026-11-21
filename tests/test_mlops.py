"""
Test Suite: test_mlops.py
Unit and Integration tests for Part 2 MLOps Pipeline & Prediction Microservice.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.dataset import load_ml_dataset, get_stratified_splits, FEATURE_COLUMNS, TARGET_COLUMN
from src.ml.preprocessing import load_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.api.main import health_check, model_info, predict_risk, StudentFeatureInput, get_model
from src.ml.monitoring import run_mlops_monitoring, analyze_numerical_drift


class TestMLOpsPipeline(unittest.TestCase):

    def test_01_dataset_extraction_and_splitting(self):
        """Test dataset loading and stratified splitting."""
        X, y, df = load_ml_dataset()
        self.assertFalse(X.empty)
        self.assertEqual(len(X), len(y))
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertTrue(set(y.unique()).issubset({0, 1}))

        X_train, X_val, X_test, y_train, y_val, y_test = get_stratified_splits()
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(X))
        self.assertGreater(y_train.sum(), 0)
        self.assertGreater(y_val.sum(), 0)
        self.assertGreater(y_test.sum(), 0)

    def test_02_preprocessing_pipeline(self):
        """Test ColumnTransformer preprocessing artifact."""
        preprocessor = load_preprocessor()
        sample_df = pd.DataFrame([{
            'age': 17, 'medu': 2, 'fedu': 2, 'traveltime': 1, 'studytime': 2, 'failures': 0,
            'famrel': 4, 'freetime': 3, 'goout': 3, 'dalc': 1, 'walc': 1, 'health': 3, 'absences': 4,
            'g1': 12, 'g2': 11, 'internal_marks_avg': 11.5, 'estimated_attendance_pct': 95.7, 'term1_to_term2_trend': -1.0,
            'school': 'GP', 'sex': 'F', 'address': 'U', 'famsize': 'GT3', 'pstatus': 'T',
            'mjob': 'other', 'fjob': 'other', 'reason': 'course', 'guardian': 'mother',
            'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no', 'activities': 'no',
            'nursery': 'yes', 'higher': 'yes', 'internet': 'yes', 'romantic': 'no'
        }])[FEATURE_COLUMNS]

        X_proc = preprocessor.transform(sample_df)
        self.assertIsInstance(X_proc, np.ndarray)
        self.assertGreater(X_proc.shape[1], len(NUMERICAL_FEATURES))

    def test_03_model_inference_bundle(self):
        """Test champion model artifact prediction."""
        bundle = get_model()
        self.assertIn("model", bundle)
        self.assertIn("preprocessor", bundle)
        self.assertIn("model_name", bundle)

    def test_04_fastapi_endpoints(self):
        """Test FastAPI endpoint functions directly."""
        # 1. Health check
        res_health = health_check()
        self.assertEqual(res_health["status"], "HEALTHY")

        # 2. Model Info
        res_info = model_info()
        self.assertIn("model_name", res_info)

        # 3. Single Prediction
        payload = StudentFeatureInput(
            school="GP", sex="M", age=17, address="U", famsize="GT3", pstatus="T",
            medu=2, fedu=2, mjob="other", fjob="other", reason="course", guardian="mother",
            traveltime=1, studytime=1, failures=2, schoolsup="no", famsup="yes",
            paid="no", activities="no", nursery="yes", higher="no", internet="yes",
            romantic="no", famrel=3, freetime=4, goout=4, dalc=2, walc=3,
            health=3, absences=18, g1=6, g2=5
        )
        res_pred = predict_risk(payload)
        self.assertIn("failure_risk_probability", res_pred)
        self.assertIn("risk_category", res_pred)
        self.assertGreaterEqual(res_pred["failure_risk_probability"], 0.5)  # Expected high risk for failing student

    def test_05_mlops_monitoring(self):
        """Test data drift and monitoring report generation."""
        report = run_mlops_monitoring()
        self.assertIn("retraining_decision", report)
        self.assertIn("feature_drift_analysis", report)
        self.assertIn("class_distribution", report)


if __name__ == "__main__":
    unittest.main()
