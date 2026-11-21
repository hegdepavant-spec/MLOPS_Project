# Student Performance and Dropout-Risk Prediction System
## Complete Data Engineering (Part 1) & MLOps Pipeline (Part 2)

An end-to-end, production-grade Data Engineering and MLOps system that ingests, validates, transforms, and loads student performance data into relational storage, trains and tracks ML models with MLflow, serves real-time predictions via FastAPI & Docker, provides interactive analytics via Streamlit, and monitors data drift and retraining triggers.

---

## 1. System Overview & Architecture

```text
┌──────────────────────────────────────────────────────────┐
│             UCI Repository (student.zip)                 │
└────────────────────────────┬─────────────────────────────┘
                             │  HTTP Ingestion Engine
                             ▼
┌──────────────────────────────────────────────────────────┐
│  src/ingestion.py ──> data/raw/student-mat.csv (Raw)     │
│                   ──> data/staging/student_staging.csv   │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  src/validation.py (Data Quality Gate & Validation)      │
│    ├── Pass ──> data/cleaned/student_cleaned.csv         │
│    └── Fail ──> data/rejected/rejected_records.csv       │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  src/transformation.py (Feature & Mart Engineering)      │
│    ├── Derived Features (Attendance %, Trends, Scores)   │
│    └── Mart File ──> data/analytics/                     │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  src/database.py ──> PostgreSQL / SQLite Storage Layer   │
│    ├── Dimension: students (Demographics)                │
│    ├── Dimension: academic_records (Term evaluations)    │
│    ├── Dimension: attendance_lifestyle (Absences & Hab.) │
│    └── Data Mart: analytics_student_performance          │
└────────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Part 1: Analytics Dashboard   │ │ Part 2: MLOps Model Pipeline  │
│ dashboard/app.py (Streamlit)  │ │ src/ml/train.py (MLflow)      │
│ 5 Institutional Views         │ │ LR / RF / XGBoost Comparison  │
└───────────────────────────────┘ └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ FastAPI Inference Service     │
                                  │ src/api/main.py (Port 8000)   │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ MLOps Continuous Monitoring   │
                                  │ src/ml/monitoring.py (Drift)  │
                                  └───────────────────────────────┘
```

---

## 2. Project Directory Structure

```text
student-performance-project/
│
├── data/
│   ├── raw/                  # Immutable original dataset + ingestion audit log
│   ├── staging/              # Staged data with assigned student identifiers
│   ├── cleaned/              # Validated records that passed all quality rules
│   ├── rejected/             # Rejected records and error audit logs
│   ├── analytics/            # Transformed analytical dataset & local SQLite DB
│   ├── ml/                   # Train, validation, and test dataset splits
│   └── prediction_logs.jsonl # Real-time inference logs for drift monitoring
│
├── src/
│   ├── ingestion.py          # Part 1: Extraction and raw preservation
│   ├── validation.py         # Part 1: Schema, range, categorical checks
│   ├── transformation.py     # Part 1: Feature engineering & attendance %
│   ├── database.py           # Part 1: PostgreSQL / SQLite loader
│   ├── ml/
│   │   ├── dataset.py        # Part 2: Feature isolation & stratified splitting
│   │   ├── preprocessing.py  # Part 2: Scikit-learn ColumnTransformer pipeline
│   │   ├── train.py          # Part 2: MLflow tracking, LR/RF/XGBoost, Model Registry
│   │   └── monitoring.py     # Part 2: KS-test data drift & retraining engine
│   └── api/
│       └── main.py           # Part 2: FastAPI microservice (/predict, /health)
│
├── dags/
│   └── student_pipeline.py   # Apache Airflow DAG orchestrating ETL
│
├── sql/
│   ├── schema.sql            # PostgreSQL DDL for 3NF normalized tables
│   ├── analytics.sql         # DDL script for data mart and analytical views
│   └── sample_queries.sql    # Verification queries for testing
│
├── dashboard/
│   └── app.py                # 6-view Streamlit interactive dashboard + ML predictor
│
├── tests/
│   ├── test_pipeline.py      # Part 1: ETL unit & integration tests
│   └── test_mlops.py         # Part 2: ML, API, and monitoring tests
│
├── docs/
│   ├── architecture.md       # Architecture diagram & layer specifications
│   ├── data_dictionary.md    # Complete column definitions & derivations
│   ├── retraining_policy.md  # Formal quantitative retraining triggers & SLA
│   ├── part1_report.md       # 14-section Data Engineering technical report
│   └── part2_report.md       # Comprehensive MLOps technical report
│
├── models/
│   ├── preprocessor.joblib   # Fitted ColumnTransformer artifact
│   ├── best_model.joblib     # Production champion model bundle
│   └── model_metadata.json   # Model versioning & benchmark manifest
│
├── Dockerfile                # FastAPI container definition
├── docker-compose.yml        # PostgreSQL, FastAPI, and MLflow multi-service
├── docker-compose-airflow.yml# Dedicated Apache Airflow standalone service
├── run.py                    # Master CLI runner for all pipeline steps
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 3. Quickstart: How to Run the 3 Core Features

Before starting, ensure your Python environment is set up:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell
# or: source .venv/bin/activate  # On Linux/macOS

pip install -r requirements.txt
```

---

### Feature 1: Apache Airflow (Pipeline Orchestration)

The pipeline is orchestrated via `dags/student_pipeline.py` with 4 sequential tasks:
`ingest_raw_data` ➔ `validate_data_quality` ➔ `transform_feature_mart` ➔ `load_postgresql_database`

#### Option A: Run via Docker (Full Airflow Web UI)
1. Ensure **Docker Desktop** is running.
2. In PowerShell, launch the Airflow container:
   ```bash
   docker compose -f docker-compose-airflow.yml up -d
   ```
3. Open your browser to **[http://localhost:8080](http://localhost:8080)**:
   - **Username**: `admin`
   - **Password**: `admin`
4. Click on **`student_performance_pipeline`** and select the **Graph** tab to see all 4 tasks.
5. Click the **▶ (Trigger DAG)** button in the top right to execute the pipeline!

#### Option B: Standalone Runner (No Docker Needed)
If you do not want to run Docker, execute the exact Airflow DAG workflow directly in Python:
```bash
python dags/student_pipeline.py
```

---

### Feature 2: MLflow (Model Tracking, Registry & Drift Monitoring)

#### 1. Train Models & Register Champion
Train Logistic Regression, Random Forest, and XGBoost models with automated experiment logging and model registry:
```bash
python run.py --step train
```

#### 2. Launch MLflow Web UI
Launch the interactive tracking server:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
- Open **[http://localhost:5000](http://localhost:5000)** in your browser.
- **View Experiments**: Select `student_dropout_risk_prediction`.
- **Compare Models**: Click **"Model training"** (or **"Training runs"** in the sidebar), select the checkboxes for the models, and click the **"Compare"** button to view side-by-side metrics (F1, Recall, ROC-AUC) and parameter charts.
- **Model Registry**: View the registered champion version under **Models** (`student_dropout_risk_model`).

#### 3. Run Continuous Drift Monitoring
Evaluate two-sample Kolmogorov-Smirnov (KS) data drift and retraining triggers:
```bash
python run.py --step monitor
```
*(Results are saved to `data/monitoring_report.json`).*

---

### Feature 3: Streamlit Interactive Dashboard & ML Risk Predictor

Launch the comprehensive 6-view analytics and prediction portal:
```bash
streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to explore:
- **Tabs 1–5 (Institutional Analytics)**:
  - Overall academic distributions and pass rates.
  - Attendance and lifestyle impact correlation.
  - Term-over-term trajectory trends (G1 to G2 to G3).
  - Early-warning intervention roster with one-click counselor export.
  - Individual student drilldown profiles.
- **Tab 6 (Real-Time ML Risk Predictor)**:
  - Live interactive form to input student attributes.
  - Predicts failure risk percentage with gauge chart visualization.
  - Outputs immediate risk tier (`HIGH_RISK`, `MEDIUM_RISK`, `LOW_RISK`) and customized academic recommendations.

---

## 4. Serving & Automation Helpers

### 4.1 Master Pipeline Runner (`run.py`)
Run any stage or the entire system end-to-end with the CLI runner:

```bash
# Run everything from ingestion to model training and drift checks:
python run.py --step all

# Or run individual steps:
python run.py --step ingest      # 1. Ingest raw dataset
python run.py --step validate    # 2. Validate data quality
python run.py --step transform   # 3. Transform features
python run.py --step database    # 4. Load database & data mart
python run.py --step train       # 5. Train & evaluate ML models
python run.py --step monitor     # 6. Run KS drift & retraining check
python run.py --step api         # 7. Start FastAPI prediction server
```

### 4.2 FastAPI Prediction Microservice
Start the production-ready REST API:
```bash
python run.py --step api
```
- Interactive Swagger UI: **[http://localhost:8000/docs](http://localhost:8000/docs)**
- Health check: `GET http://localhost:8000/health`
- Single prediction: `POST http://localhost:8000/predict`
- Batch prediction: `POST http://localhost:8000/predict/batch`

---

## 5. Automated Test Suites

Run the full automated test suite to verify data engineering and MLOps integrity:

```bash
# Part 1 Tests (ETL, Schema Validation, SQLite/PostgreSQL Database):
python -m unittest tests/test_pipeline.py

# Part 2 Tests (Stratified Splits, Preprocessing, API Inference, Drift Monitoring):
python -m unittest tests/test_mlops.py
```

---

## 6. Model Retraining Governance Triggers

The monitoring engine (`src/ml/monitoring.py`) evaluates 4 concrete criteria to trigger automated retraining:
1. **Covariate Data Drift**: Two-sample Kolmogorov-Smirnov (KS) test ($p < 0.05$ on $\ge 2$ features).
2. **Class Shift**: Absolute deviation $> 15.0\%$ between baseline failure rate and production predictions.
3. **Performance Degradation**: Validation F1-score drops $> 10\%$.
4. **Data Volume Threshold**: Accumulation of $\ge 100$ new labeled student records.
