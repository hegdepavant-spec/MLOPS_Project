"""
Test Suite: test_pipeline.py
Unit and Integration tests for Data Engineering Pipeline (Part 1).
"""

import os
import sys
import unittest
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ingestion import ingest_data, INGESTION_LOG_PATH, RAW_MATH_CSV
from src.validation import validate_data, validate_record
from src.transformation import transform_data, calculate_dropout_risk_score, categorize_performance
from src.database import load_database, get_db_engine


class TestDataEngineeringPipeline(unittest.TestCase):

    def test_01_ingestion(self):
        """Test that data ingestion produces raw files, logs, and staging copies."""
        metadata = ingest_data()
        self.assertEqual(metadata.get("status"), "SUCCESS")
        self.assertGreater(metadata.get("row_count", 0), 0)
        self.assertTrue(os.path.exists(RAW_MATH_CSV))
        self.assertTrue(os.path.exists(INGESTION_LOG_PATH))

    def test_02_validation_rules(self):
        """Test individual data validation logic on valid and invalid simulated rows."""
        # 1. Valid row
        valid_row = pd.Series({
            'student_id': 'STU0001',
            'school': 'GP',
            'sex': 'F',
            'age': 18,
            'address': 'U',
            'famsize': 'GT3',
            'Pstatus': 'A',
            'Medu': 4,
            'Fedu': 4,
            'Mjob': 'at_home',
            'Fjob': 'teacher',
            'reason': 'course',
            'guardian': 'mother',
            'traveltime': 2,
            'studytime': 2,
            'failures': 0,
            'schoolsup': 'yes',
            'famsup': 'no',
            'paid': 'no',
            'activities': 'no',
            'nursery': 'yes',
            'higher': 'yes',
            'internet': 'no',
            'romantic': 'no',
            'famrel': 4,
            'freetime': 3,
            'goout': 4,
            'Dalc': 1,
            'Walc': 1,
            'health': 3,
            'absences': 6,
            'G1': 15,
            'G2': 14,
            'G3': 16
        })
        errors = validate_record(valid_row)
        self.assertEqual(len(errors), 0, f"Expected no errors for valid row, got {errors}")

        # 2. Invalid row with multiple errors
        invalid_row = valid_row.copy()
        invalid_row['G3'] = 25  # Out of range (0-20)
        invalid_row['absences'] = -5  # Negative absences
        invalid_row['school'] = 'INVALID_SCHOOL'  # Invalid domain
        
        errors = validate_record(invalid_row)
        self.assertGreaterEqual(len(errors), 3)

    def test_03_validation_pipeline(self):
        """Test the validation pipeline dataset execution."""
        summary = validate_data()
        self.assertGreater(summary.get("total_records_evaluated", 0), 0)
        self.assertGreater(summary.get("valid_records_passed", 0), 0)

    def test_04_transformation_features(self):
        """Test feature engineering calculations."""
        # Test performance categorization
        self.assertEqual(categorize_performance(17.5), "Excellent")
        self.assertEqual(categorize_performance(14.2), "Good")
        self.assertEqual(categorize_performance(11.0), "Satisfactory")
        self.assertEqual(categorize_performance(7.5), "Fail / At-Risk")

        # Test risk score
        sample_student = pd.Series({
            'internal_marks_avg': 5.0, # low grades
            'absences': 20,           # high absences
            'failures': 3,            # high failures
            'studytime': 1            # low study time
        })
        score = calculate_dropout_risk_score(sample_student)
        self.assertGreater(score, 50.0, "Expected high risk score for at-risk student")

        # Test dataset transformation
        summary = transform_data()
        self.assertEqual(summary.get("status"), "SUCCESS")
        self.assertGreater(summary.get("rows_transformed", 0), 0)

    def test_05_database_loading(self):
        """Test database table creation and loading."""
        summary = load_database()
        self.assertEqual(summary.get("status"), "SUCCESS")
        self.assertIn("analytics_student_performance", summary.get("tables_created", []))
        self.assertGreater(summary.get("records_loaded", {}).get("analytics_student_performance", 0), 0)


if __name__ == "__main__":
    unittest.main()
