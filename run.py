"""
Master Orchestrator: Student Performance and Dropout-Risk Prediction System
End-to-end support for Part 1 (Data Engineering) and Part 2 (MLOps Pipeline).
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ingestion import ingest_data
from src.validation import validate_data
from src.transformation import transform_data
from src.database import load_database


def run_pipeline(step: str = "all") -> bool:
    print("=" * 80)
    print(f"EXECUTING PIPELINE — STEP: {step.upper()}")
    print("=" * 80)

    if step in ['ingest', 'all', 'all-pipeline']:
        print("\n[Step 1/4] Ingesting Raw Dataset...")
        ingest_res = ingest_data()
        print(f"Ingestion Status: {ingest_res.get('status')}, Rows: {ingest_res.get('row_count')}")

    if step in ['validate', 'all', 'all-pipeline']:
        print("\n[Step 2/4] Validating Quality Rules & Segregating Records...")
        valid_res = validate_data()
        print(f"Validation: Total={valid_res.get('total_records_evaluated')}, Clean={valid_res.get('valid_records_passed')}, Rejected={valid_res.get('rejected_records_count')}")

    if step in ['transform', 'all', 'all-pipeline']:
        print("\n[Step 3/4] Transforming Features & Aggregates...")
        trans_res = transform_data()
        print(f"Transformation: Status={trans_res.get('status')}, Transformed Rows={trans_res.get('rows_transformed')}")

    if step in ['database', 'load', 'all', 'all-pipeline']:
        print("\n[Step 4/4] Loading Relational Tables & Analytical Data Mart...")
        db_res = load_database()
        print(f"Database ({db_res.get('database_type')}): Tables Created={db_res.get('tables_created')}")

    if step in ['train', 'ml-train', 'all']:
        print("\n[Step 5/6] Training & Tracking ML Models in MLflow...")
        from src.ml.train import train_and_evaluate_all
        train_and_evaluate_all()

    if step in ['monitor', 'all']:
        print("\n[Step 6/6] Running MLOps Continuous Monitoring & Drift Checks...")
        from src.ml.monitoring import run_mlops_monitoring
        run_mlops_monitoring()

    if step == 'api':
        print("\n[API Server] Starting FastAPI prediction service on port 8000...")
        import uvicorn
        uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Performance Pipeline Runner (Part 1 & 2)")
    parser.add_argument(
        '--step', '-s',
        choices=['ingest', 'validate', 'transform', 'load', 'database', 'train', 'ml-train', 'monitor', 'api', 'all'],
        default='all',
        help="Specify the pipeline stage to run"
    )
    args = parser.parse_args()
    ok = run_pipeline(args.step)
    sys.exit(0 if ok else 1)
