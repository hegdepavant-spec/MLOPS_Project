"""
Module: transformation.py
Description: Transforms cleaned student data into structured analytical datasets
             and calculates derived features (attendance %, internal marks avg,
             grade progression trends, dropout risk index, and cohort aggregates).
"""

import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLEANED_CSV = os.path.join(DATA_DIR, 'cleaned', 'student_cleaned.csv')
FALLBACK_CLEAN_CSV = os.path.join(DATA_DIR, 'processed', 'student-mat_clean.csv')
ANALYTICS_DIR = os.path.join(DATA_DIR, 'analytics')
ANALYTICS_CSV = os.path.join(ANALYTICS_DIR, 'analytics_student_performance.csv')
MART_CSV = os.path.join(DATA_DIR, 'processed', 'student_performance_mart.csv')


def categorize_performance(avg_grade: float) -> str:
    """Categorize performance on the Portuguese 0-20 scale."""
    if avg_grade >= 16.0:
        return "Excellent"
    elif avg_grade >= 14.0:
        return "Good"
    elif avg_grade >= 10.0:
        return "Satisfactory"
    else:
        return "Fail / At-Risk"


def calculate_dropout_risk_score(row: pd.Series) -> float:
    """
    Composite Dropout Risk Score (0 - 100):
    - Academic deficit (45% weight): (20 - internal_marks_avg) / 20 * 45
    - Absence rate (30% weight): min(absences, 30) / 30 * 30
    - Past academic failures (15% weight): min(failures, 3) / 3 * 15
    - Study time deficit (10% weight): ((4 - studytime) / 3) * 10
    """
    internal_avg = row.get('internal_marks_avg', row.get('G1', 10))
    absences = row.get('absences', 0)
    failures = row.get('failures', 0)
    studytime = row.get('studytime', 2)

    academic_penalty = max(0.0, ((20.0 - internal_avg) / 20.0) * 45.0)
    absence_penalty = (min(float(absences), 30.0) / 30.0) * 30.0
    failure_penalty = (min(float(failures), 3.0) / 3.0) * 15.0
    study_penalty = (max(0.0, 4.0 - float(studytime)) / 3.0) * 10.0

    total_score = round(academic_penalty + absence_penalty + failure_penalty + study_penalty, 2)
    return min(100.0, max(0.0, total_score))


def transform_data(input_path: str = None) -> dict:
    """
    Loads validated cleaned data, computes all analytical features,
    and produces the analytical data mart.
    """
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)

    source_path = input_path or CLEANED_CSV
    if not os.path.exists(source_path):
        if os.path.exists(FALLBACK_CLEAN_CSV):
            source_path = FALLBACK_CLEAN_CSV
        else:
            raise FileNotFoundError(f"Cleaned dataset not found at {source_path}. Run validation first.")

    logger.info(f"Loading cleaned dataset from {source_path}...")
    df = pd.read_csv(source_path)

    if df.empty:
        logger.warning("Cleaned dataset is empty. Aborting transformation.")
        return {"status": "EMPTY", "rows_transformed": 0}

    logger.info(f"Transforming {len(df)} records with feature engineering...")

    # 1. Subject Assignment
    df['subject'] = 'Mathematics'
    df['semester'] = 'Annual'

    # 2. Internal Marks and Overall Averages
    # G1 = Period 1 evaluation, G2 = Period 2 evaluation, G3 = Final Grade
    df['internal_marks_avg'] = ((df['G1'] + df['G2']) / 2.0).round(2)
    df['overall_academic_avg'] = ((df['G1'] + df['G2'] + df['G3']) / 3.0).round(2)
    df['final_grade'] = df['G3']

    # 3. Attendance Metrics
    # Max observed absences in dataset is ~93; calculate attendance percentage baseline
    # Estimated 93 sessions per academic year
    TOTAL_SESSIONS = 93.0
    df['estimated_attendance_pct'] = (np.maximum(0.0, (1.0 - (df['absences'] / TOTAL_SESSIONS))) * 100.0).round(2)

    # 4. Period-to-Period Progression Trends (Previous Semester / Period Performance)
    df['term1_to_term2_trend'] = (df['G2'] - df['G1']).round(2)
    df['term2_to_final_trend'] = (df['G3'] - df['G2']).round(2)
    df['overall_grade_growth'] = (df['G3'] - df['G1']).round(2)

    # 5. Categorization and Pass/Fail Status
    df['performance_category'] = df['overall_academic_avg'].apply(categorize_performance)
    df['final_performance_category'] = df['final_grade'].apply(categorize_performance)
    df['pass_status'] = df['final_grade'].apply(lambda g: 'Pass' if g >= 10 else 'Fail')

    # 6. Dropout Risk Score and Levels
    df['dropout_risk_score'] = df.apply(calculate_dropout_risk_score, axis=1)
    df['dropout_risk_level'] = df['dropout_risk_score'].apply(
        lambda s: 'High Risk' if s >= 50.0 else ('Medium Risk' if s >= 30.0 else 'Low Risk')
    )
    df['requires_intervention'] = df['dropout_risk_level'].isin(['High Risk', 'Medium Risk']) | (df['pass_status'] == 'Fail')

    # 7. Cohort-Level Aggregates (School & Sex Cohort Means)
    school_cohort_mean = df.groupby('school')['final_grade'].transform('mean').round(2)
    df['school_cohort_avg_grade'] = school_cohort_mean
    df['deviation_from_cohort_avg'] = (df['final_grade'] - df['school_cohort_avg_grade']).round(2)

    # Save to Analytics Data Mart layers
    df.to_csv(ANALYTICS_CSV, index=False)
    df.to_csv(MART_CSV, index=False)

    summary = {
        "status": "SUCCESS",
        "rows_transformed": len(df),
        "columns_count": len(df.columns),
        "analytics_output": os.path.relpath(ANALYTICS_CSV, BASE_DIR),
        "high_risk_count": int((df['dropout_risk_level'] == 'High Risk').sum()),
        "pass_rate_pct": round((df['pass_status'] == 'Pass').mean() * 100.0, 2),
        "overall_grade_avg": round(df['final_grade'].mean(), 2)
    }

    logger.info(f"Transformation complete. Data mart saved with {len(df)} rows.")
    return summary


if __name__ == "__main__":
    summary = transform_data()
    print("Transformation Summary:", json.dumps(summary, indent=2))
