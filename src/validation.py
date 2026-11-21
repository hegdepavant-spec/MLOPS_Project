"""
Module: validation.py
Description: Enforces data quality rules on staged data.
             Separates valid records into cleaned/ and invalid records into rejected/
             with documented rejection reasons and error logging.
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
STAGING_CSV = os.path.join(DATA_DIR, 'staging', 'student_staging.csv')
RAW_CSV = os.path.join(DATA_DIR, 'raw', 'student-mat.csv')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')
REJECTED_DIR = os.path.join(DATA_DIR, 'rejected')
CLEANED_CSV = os.path.join(CLEANED_DIR, 'student_cleaned.csv')
REJECTED_CSV = os.path.join(REJECTED_DIR, 'rejected_records.csv')
VALIDATION_LOG = os.path.join(REJECTED_DIR, 'validation_summary.json')

# Categorical Domains
VALID_CATEGORIES = {
    'school': {'GP', 'MS'},
    'sex': {'F', 'M'},
    'address': {'U', 'R'},
    'famsize': {'LE3', 'GT3'},
    'Pstatus': {'T', 'A'},
    'Mjob': {'teacher', 'health', 'services', 'at_home', 'other'},
    'Fjob': {'teacher', 'health', 'services', 'at_home', 'other'},
    'reason': {'home', 'reputation', 'course', 'other'},
    'guardian': {'mother', 'father', 'other'},
    'schoolsup': {'yes', 'no'},
    'famsup': {'yes', 'no'},
    'paid': {'yes', 'no'},
    'activities': {'yes', 'no'},
    'nursery': {'yes', 'no'},
    'higher': {'yes', 'no'},
    'internet': {'yes', 'no'},
    'romantic': {'yes', 'no'}
}


def validate_record(row: pd.Series) -> list:
    """Validate a single student record against data quality rules."""
    reasons = []

    # 1. Student Identifier Check
    stu_id = str(row.get('student_id', '')).strip()
    if not stu_id or pd.isna(row.get('student_id')):
        reasons.append("Missing or empty student_id")

    # 2. Grade Range & Type Validation (G1, G2, G3 in [0, 20])
    for g in ['G1', 'G2', 'G3']:
        val = row.get(g)
        if pd.isna(val):
            reasons.append(f"Missing grade value for {g}")
        else:
            try:
                numeric_g = float(val)
                if numeric_g < 0 or numeric_g > 20:
                    reasons.append(f"Out of range grade {g}={numeric_g} (expected 0-20)")
            except (ValueError, TypeError):
                reasons.append(f"Non-numeric grade {g}={val}")

    # 3. Numeric Bounds Check
    # Absences >= 0
    try:
        abs_val = int(row.get('absences', 0))
        if abs_val < 0 or abs_val > 93:
            reasons.append(f"Invalid absences count: {abs_val}")
    except (ValueError, TypeError):
        reasons.append(f"Non-integer absences: {row.get('absences')}")

    # Age (15-22 typical high school range)
    try:
        age_val = int(row.get('age', 0))
        if age_val < 10 or age_val > 30:
            reasons.append(f"Abnormal age value: {age_val}")
    except (ValueError, TypeError):
        reasons.append(f"Non-integer age: {row.get('age')}")

    # Failures (0 to 4)
    try:
        fail_val = int(row.get('failures', 0))
        if fail_val < 0 or fail_val > 4:
            reasons.append(f"Invalid failures count: {fail_val}")
    except (ValueError, TypeError):
        reasons.append(f"Non-integer failures: {row.get('failures')}")

    # Scale 1-5 checks (Medu, Fedu, traveltime, studytime, famrel, freetime, goout, Dalc, Walc, health)
    for col, min_v, max_v in [
        ('Medu', 0, 4), ('Fedu', 0, 4),
        ('traveltime', 1, 4), ('studytime', 1, 4),
        ('famrel', 1, 5), ('freetime', 1, 5), ('goout', 1, 5),
        ('Dalc', 1, 5), ('Walc', 1, 5), ('health', 1, 5)
    ]:
        try:
            val = int(row.get(col, -1))
            if val < min_v or val > max_v:
                reasons.append(f"Invalid scale value for {col}={val} (expected {min_v}-{max_v})")
        except (ValueError, TypeError):
            reasons.append(f"Non-integer value for {col}={row.get(col)}")

    # 4. Categorical Domain Checks
    for cat_col, valid_set in VALID_CATEGORIES.items():
        val = str(row.get(cat_col, '')).strip().lower()
        # Case-insensitive comparison
        if val not in {v.lower() for v in valid_set}:
            reasons.append(f"Invalid categorical value '{row.get(cat_col)}' for column '{cat_col}'")

    return reasons


def validate_data(input_path: str = None) -> dict:
    """
    Validates staged or raw student records.
    Outputs clean records to data/cleaned/student_cleaned.csv and
    rejected records to data/rejected/rejected_records.csv.
    """
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    target_source = input_path or STAGING_CSV
    if not os.path.exists(target_source):
        if os.path.exists(RAW_CSV):
            logger.info(f"Staging file not found. Falling back to raw CSV {RAW_CSV}")
            df = pd.read_csv(RAW_CSV, sep=';')
            if 'student_id' not in df.columns:
                df.insert(0, 'student_id', [f"STU{str(i).zfill(4)}" for i in range(1, len(df) + 1)])
        else:
            raise FileNotFoundError(f"Neither staging ({STAGING_CSV}) nor raw ({RAW_CSV}) exists. Run ingestion first.")
    else:
        df = pd.read_csv(target_source)

    total_rows = len(df)
    valid_rows = []
    rejected_rows = []

    logger.info(f"Starting data validation on {total_rows} records...")

    # Individual record validation
    for _, row in df.iterrows():
        errors = validate_record(row)
        if errors:
            rej = row.to_dict()
            rej['rejection_reason'] = "; ".join(errors)
            rejected_rows.append(rej)
        else:
            valid_rows.append(row.to_dict())

    # Check for Duplicate records in valid pool (excluding synthetic student_id)
    df_valid = pd.DataFrame(valid_rows)
    if not df_valid.empty:
        non_id_cols = [c for c in df_valid.columns if c != 'student_id']
        dup_mask = df_valid.duplicated(subset=non_id_cols, keep='first')
        
        if dup_mask.any():
            dups = df_valid[dup_mask]
            logger.warning(f"Detected {len(dups)} duplicate student demographic records. Moving to rejected layer.")
            for _, dup_row in dups.iterrows():
                rej_dup = dup_row.to_dict()
                rej_dup['rejection_reason'] = "Duplicate student demographic record"
                rejected_rows.append(rej_dup)
            df_valid = df_valid[~dup_mask]

    # Save cleaned data
    if not df_valid.empty:
        df_valid.to_csv(CLEANED_CSV, index=False)
        # Also maintain compatibility with processed folder if expected
        proc_dir = os.path.join(DATA_DIR, 'processed')
        os.makedirs(proc_dir, exist_ok=True)
        df_valid.to_csv(os.path.join(proc_dir, 'student-mat_clean.csv'), index=False)
    else:
        pd.DataFrame(columns=df.columns).to_csv(CLEANED_CSV, index=False)

    # Save rejected data
    df_rejected = pd.DataFrame(rejected_rows)
    if not df_rejected.empty:
        df_rejected.to_csv(REJECTED_CSV, index=False)
    else:
        pd.DataFrame(columns=list(df.columns) + ['rejection_reason']).to_csv(REJECTED_CSV, index=False)

    summary = {
        "total_records_evaluated": total_rows,
        "valid_records_passed": len(df_valid),
        "rejected_records_count": len(rejected_rows),
        "cleaned_output_path": os.path.relpath(CLEANED_CSV, BASE_DIR),
        "rejected_output_path": os.path.relpath(REJECTED_CSV, BASE_DIR),
        "validation_pass_rate_pct": round((len(df_valid) / total_rows * 100) if total_rows > 0 else 0.0, 2)
    }

    with open(VALIDATION_LOG, 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Validation complete: {summary['valid_records_passed']} valid, {summary['rejected_records_count']} rejected.")
    return summary


if __name__ == "__main__":
    summary = validate_data()
    print("Validation Summary:", json.dumps(summary, indent=2))
