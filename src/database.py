"""
Module: database.py
Description: Database access layer for PostgreSQL with automatic fallback to SQLite.
             Executes DDL schema and loads cleaned and analytical data marts.
"""

import os
import json
import sqlite3
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLEANED_CSV = os.path.join(DATA_DIR, 'cleaned', 'student_cleaned.csv')
FALLBACK_CLEAN_CSV = os.path.join(DATA_DIR, 'processed', 'student-mat_clean.csv')
ANALYTICS_CSV = os.path.join(DATA_DIR, 'analytics', 'analytics_student_performance.csv')
FALLBACK_MART_CSV = os.path.join(DATA_DIR, 'processed', 'student_performance_mart.csv')

# PostgreSQL Database credentials
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "student_performance_db")

SQLITE_PATH = os.path.join(DATA_DIR, 'analytics', 'student_performance.db')


def get_db_engine():
    """
    Attempts connection to PostgreSQL.
    If unavailable or driver missing, falls back gracefully to SQLite for local offline testing.
    """
    pg_uri = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(pg_uri, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database '{DB_NAME}' at {DB_HOST}:{DB_PORT}")
        return engine, "POSTGRESQL"
    except Exception as e:
        logger.warning(f"PostgreSQL connection unavailable ({e}). Falling back to SQLite local database.")
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        sqlite_engine = create_engine(f"sqlite:///{SQLITE_PATH}")
        return sqlite_engine, "SQLITE"


def load_database():
    """
    Creates relational and analytical tables and populates data from cleaned and analytics CSVs.
    """
    clean_path = CLEANED_CSV if os.path.exists(CLEANED_CSV) else FALLBACK_CLEAN_CSV
    analytics_path = ANALYTICS_CSV if os.path.exists(ANALYTICS_CSV) else FALLBACK_MART_CSV

    if not os.path.exists(clean_path) or not os.path.exists(analytics_path):
        raise FileNotFoundError("Missing cleaned or transformed analytics data. Run validation and transformation first.")

    clean_df = pd.read_csv(clean_path)
    analytics_df = pd.read_csv(analytics_path)

    engine, db_type = get_db_engine()

    # Standardize column casing for database compatibility
    clean_df.columns = [c.lower() for c in clean_df.columns]
    analytics_df.columns = [c.lower() for c in analytics_df.columns]

    # 1. Demographics table
    demo_cols = ['student_id', 'school', 'sex', 'age', 'address', 'famsize', 'pstatus', 
                 'medu', 'fedu', 'mjob', 'fjob', 'reason', 'guardian', 'traveltime']
    demo_df = clean_df[[c for c in demo_cols if c in clean_df.columns]]

    # 2. Academic Records table
    acad_cols = ['student_id', 'studytime', 'failures', 'schoolsup', 'famsup', 
                 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 
                 'g1', 'g2', 'g3']
    acad_df = clean_df[[c for c in acad_cols if c in clean_df.columns]].copy()
    if 'subject' not in acad_df.columns:
        acad_df['subject'] = 'Mathematics'

    # 3. Attendance & Lifestyle table
    attn_cols = ['student_id', 'absences', 'famrel', 'freetime', 'goout', 'dalc', 'walc', 'health']
    attn_df = clean_df[[c for c in attn_cols if c in clean_df.columns]].copy()
    if 'estimated_attendance_pct' in analytics_df.columns:
        attn_df['estimated_attendance_pct'] = analytics_df['estimated_attendance_pct']
    else:
        attn_df['estimated_attendance_pct'] = (1.0 - (attn_df['absences'] / 93.0)).clip(lower=0.0) * 100.0

    # 4. Analytical Data Mart table
    mart_df = analytics_df.copy()

    # Load tables using pandas to_sql
    logger.info(f"Loading data into tables ({db_type})...")
    if db_type == "SQLITE":
        with sqlite3.connect(SQLITE_PATH) as sq_conn:
            demo_df.to_sql('students', con=sq_conn, if_exists='replace', index=False)
            acad_df.to_sql('academic_records', con=sq_conn, if_exists='replace', index=False)
            attn_df.to_sql('attendance_lifestyle', con=sq_conn, if_exists='replace', index=False)
            mart_df.to_sql('analytics_student_performance', con=sq_conn, if_exists='replace', index=False)
    else:
        demo_df.to_sql('students', con=engine, if_exists='replace', index=False)
        acad_df.to_sql('academic_records', con=engine, if_exists='replace', index=False)
        attn_df.to_sql('attendance_lifestyle', con=engine, if_exists='replace', index=False)
        mart_df.to_sql('analytics_student_performance', con=engine, if_exists='replace', index=False)

    summary = {
        "status": "SUCCESS",
        "database_type": db_type,
        "tables_created": ["students", "academic_records", "attendance_lifestyle", "analytics_student_performance"],
        "records_loaded": {
            "students": len(demo_df),
            "academic_records": len(acad_df),
            "attendance_lifestyle": len(attn_df),
            "analytics_student_performance": len(mart_df)
        }
    }

    logger.info(f"Database load complete: {summary}")
    return summary


def fetch_analytical_data() -> pd.DataFrame:
    """Helper method for Dashboard to query the analytics data mart."""
    engine, _ = get_db_engine()
    query = "SELECT * FROM analytics_student_performance"
    return pd.read_sql(query, con=engine)


if __name__ == "__main__":
    result = load_database()
    print("Database Result:", json.dumps(result, indent=2))
