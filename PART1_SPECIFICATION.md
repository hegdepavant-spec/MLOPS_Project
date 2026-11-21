# Student Performance Project — Part 1 MVP Specification & Execution Plan

## Overview
Demonstrate a complete pipeline from ingestion through dashboard output, focused strictly on data engineering (Part 1).

---

## Phase 1: Data
### Step 1 — Choose & Download the Dataset
- **Dataset**: UCI Student Performance Dataset (Portuguese / Math performance).
- **Directory**: `data/raw/` (Keep original unmodified).
- **Metadata to Record**:
  - Dataset name
  - Source
  - Download/access instructions
  - File name
  - Number of rows
  - Number of columns
  - Column descriptions

### Step 2 — Project Setup & Structure
Target folder structure:
```text
student-performance-project/
│
├── data/
│   ├── raw/
│   ├── staging/
│   ├── cleaned/
│   └── rejected/
│
├── src/
│   ├── ingestion.py
│   ├── validation.py
│   ├── transformation.py
│   └── database.py
│
├── dags/
│   └── student_pipeline.py
│
├── sql/
│   ├── schema.sql
│   └── analytics.sql
│
├── dashboard/
│   └── app.py
│
├── tests/
│
├── docs/
│   ├── architecture.png
│   └── data_dictionary.xlsx
│
├── requirements.txt
└── README.md
```

### Step 3 — Build Python Ingestion (`src/ingestion.py`)
- Ingestion flow: `Dataset -> Python/Pandas -> Raw data`
- Key functions:
  - Check file existence
  - Record extraction date, source, status, row count
  - Maintain raw copy in `data/raw/`
  - Error logging
- Verification: `python src/ingestion.py`

### Step 4 — Exploratory Data Inspection
- Inspect via Pandas:
  - `df.shape`, `df.columns`, `df.dtypes`
  - `df.isnull().sum()`, `df.duplicated().sum()`, `df.describe()`
  - Unique values, min/max marks.
  - Rely only on actual columns present in dataset (no invented columns).

### Step 5 — Validation Rules (`src/validation.py`)
- Validation checks:
  - Duplicate student records
  - Invalid marks / out-of-bound scores
  - Missing required fields
  - Invalid identifiers / incorrect datatypes / invalid categorical values
- Data separation:
  - Valid records -> `data/cleaned/` (or staging)
  - Invalid records -> `data/rejected/` (with error log)

### Step 6 — Cleaning & Transformation (`src/transformation.py`)
- Pipeline:
  - Standardize columns
  - Handle missing values / remove duplicates
  - Validate marks
  - Calculated features & aggregates (e.g. attendance percentage/absences, score averages, past performance, student/subject/semester aggregates where supported)
  - Output clean analytical dataset

---

## Phase 2: Engineering & Database
### Step 7 — PostgreSQL Database Design (`src/database.py`, `sql/schema.sql`)
- Database: `student_performance_db`
- Schema reflecting actual data (e.g. `students`, `subjects`, `student_performance`, etc.)
- Load cleaned data into PostgreSQL tables.

### Step 8 — Analytical Table / Data Mart (`sql/analytics.sql`)
- Create analytical mart table (e.g., `analytics_student_performance`):
  - `student_id`, `subject`, `average_score`, attendance/absences metrics, `performance_category`, etc.
  - Feeds directly into downstream analytics & dashboard.

### Step 9 — Build Apache Airflow DAG (`dags/student_pipeline.py`)
- DAG Workflow:
  `START -> INGEST DATA -> VALIDATE DATA -> TRANSFORM DATA -> LOAD POSTGRESQL -> CREATE ANALYTICS TABLE -> END`

---

## Phase 3: Application & Dashboard
### Step 10 — Streamlit Dashboard (`dashboard/app.py`)
- Connect Streamlit to PostgreSQL database / analytics table.
- Implement at least 5 meaningful views/indicators:
  1. **Overall Performance**: Average score, student count, pass rate.
  2. **Performance Distribution**: Score and performance categories.
  3. **Subject Performance**: Comparative metrics across subjects.
  4. **Attendance/Absence vs Performance**: Impact analysis.
  5. **Student Analysis**: Individual drill-down and score cards.

---

## Phase 4: Submission & Deliverables
### Step 11 — Architecture Diagram
- Visualizing Source -> Python/Pandas Ingestion -> Raw Layer -> Validation + Transformation -> PostgreSQL Cleaned Data -> Analytics Mart -> Streamlit Dashboard, orchestrated by Airflow.

### Step 12 — Data Dictionary & Validation Documentation
- Tabular documentation (`Column`, `Type`, `Description`, `Source`, `Transformation`, Validation Rules).

### Step 13 — End-to-End Pipeline Testing & Evidence
- Complete execution verification and capture of screenshots across all pipeline stages.

### Step 14 — README Documentation
- Setup, prerequisites, PostgreSQL configuration, step-by-step execution guide for ingestion, Airflow DAG, and Streamlit.

### Step 15 — 8–10 Page Project Report
- Structure covering Sections 1 through 15 with screenshots/evidence.

### Step 16 — Final Deliverables Checklist
1. Source code + ETL
2. Airflow DAG
3. PostgreSQL schema + tables
4. Dataset source/access documentation
5. Architecture diagram
6. Data dictionary + validation rules
7. Streamlit dashboard
8. 8–10 page report
9. Screenshots / execution evidence
10. README.md
