"""
Module: student_pipeline.py
Description: Apache Airflow DAG orchestrating the end-to-end data pipeline:
             Ingestion -> Validation -> Transformation -> PostgreSQL Database -> Analytics Mart.
"""

import os
import sys
from datetime import datetime, timedelta

# Ensure project root is in python path (resolve symlinks so Airflow DAG directory works)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Fallback for WSL environments mounting Windows workspace
WSL_WIN_PATH = "/mnt/c/Users/Pavan BG/Documents/mlops-assignment-"
if os.path.exists(WSL_WIN_PATH) and WSL_WIN_PATH not in sys.path:
    sys.path.insert(0, WSL_WIN_PATH)

from src.ingestion import ingest_data
from src.validation import validate_data
from src.transformation import transform_data
from src.database import load_database

# Default task arguments
default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    # Define the Airflow DAG
    with DAG(
        'student_performance_pipeline',
        default_args=default_args,
        description='End-to-End Student Performance Data Engineering Pipeline',
        schedule_interval='@daily',
        catchup=False,
        tags=['student_performance', 'data_engineering', 'part1']
    ) as dag:

        t1_ingest = PythonOperator(
            task_id='ingest_raw_data',
            python_callable=ingest_data,
            doc_md="""\
            #### Task 1: Ingestion
            Pulls raw UCI Student Performance dataset, stores raw copy, logs extraction metadata.
            """
        )

        t2_validate = PythonOperator(
            task_id='validate_data_quality',
            python_callable=validate_data,
            doc_md="""\
            #### Task 2: Validation
            Applies schema, domain, and duplicate checks. Separates data into Cleaned and Rejected sets.
            """
        )

        t3_transform = PythonOperator(
            task_id='transform_feature_mart',
            python_callable=transform_data,
            doc_md="""\
            #### Task 3: Transformation
            Calculates attendance %, internal marks average, grade trends, and dropout risk index.
            """
        )

        t4_load_db = PythonOperator(
            task_id='load_postgresql_database',
            python_callable=load_database,
            doc_md="""\
            #### Task 4: Database Loading
            Loads normalized dimensions and analytical data mart into PostgreSQL.
            """
        )

        # Pipeline dependencies: extract -> raw -> validate -> clean/transform -> PostgreSQL -> analytics
        t1_ingest >> t2_validate >> t3_transform >> t4_load_db

except ImportError:
    # Standalone execution mode when Airflow package is not loaded in current environment
    dag = None


def run_standalone_pipeline():
    """Standalone runner for verifying the sequential orchestration workflow."""
    print("=" * 80)
    print("STARTING STANDALONE PIPELINE EXECUTION (Airflow-equivalent workflow)")
    print("=" * 80)

    print("\n[Stage 1/4] Ingesting Raw Data...")
    ingest_res = ingest_data()
    print("Ingestion:", ingest_res)

    print("\n[Stage 2/4] Validating Data Quality...")
    valid_res = validate_data()
    print("Validation:", valid_res)

    print("\n[Stage 3/4] Transforming Analytical Features...")
    trans_res = transform_data()
    print("Transformation:", trans_res)

    print("\n[Stage 4/4] Loading Relational & Analytical Database...")
    db_res = load_database()
    print("Database Load:", db_res)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_standalone_pipeline()
