# Student Performance and Dropout-Risk Prediction System
## Comprehensive Technical Report — Part 1: Data Engineering & Analytics Infrastructure

**Author**: Individual Academic Project  
**Module**: Data Engineering & MLOps Pipeline  
**Version**: 1.0.0 (Part 1 Deliverable)  
**Date**: Academic Year 2025/2026  

---

## Executive Summary

This report documents the architectural design, engineering implementation, and verification of **Part 1** of the *Student Performance and Dropout-Risk Prediction System*. The primary objective of Part 1 is the construction of a production-grade data engineering pipeline that ingests historical student benchmark data from the UCI Machine Learning Repository, enforces stringent multi-tiered data quality validations, computes domain-specific analytical features, loads normalized relational structures and an analytical data mart into PostgreSQL, and presents interactive insights via an Apache Airflow-orchestrated Streamlit dashboard.

All data transformations and schemas are grounded in real, empirical data attributes without fabricating unsupported columns. The pipeline guarantees raw data immutability, automated audit logging, rejected record isolation, and deterministic execution.

---

## 1. Introduction & Problem Understanding

Higher educational institutions face significant challenges in identifying students at risk of academic failure or course dropout before term-end evaluations. Early detection allows academic counselors and department heads to stage targeted interventions, such as supplemental tutoring, attendance counseling, and personalized study plans.

However, academic data is frequently siloed, heterogeneous, and prone to quality anomalies (such as transcription errors, missing values, duplicate entries, and out-of-bound marks). Without a reliable, automated data engineering foundation:
1. Historical data cannot be consistently audited or reproduced.
2. Downstream analytical dashboards produce skewed or misleading institutional metrics.
3. Subsequent predictive machine learning models (Part 2) risk learning from contaminated or unvalidated features.

Thus, Part 1 focuses strictly on building a robust, auditable, and automated data engineering pipeline.

---

## 2. Problem Statement & Key Objectives

### 2.1 Problem Statement
Design, implement, test, and orchestrate an end-to-end data pipeline capable of:
- Ingesting raw student performance datasets from a public repository without manual intervention.
- Isolating and logging data quality defects into a dedicated rejected layer.
- Transforming validated records into clean analytical features (e.g., attendance percentage, internal marks average, multi-term grade trajectories, and composite dropout risk scores).
- Persisting structured dimensions and analytical marts into a relational database (PostgreSQL).
- Serving multi-dimensional institutional indicators via an interactive dashboard.

### 2.2 Core Technical Objectives
- **Data Ingestion**: Reproducible Python extraction script maintaining raw data immutability and capturing extraction metadata (timestamp, source URI, status, row count).
- **Data Quality & Isolation**: Comprehensive validation suite separating records into `cleaned/` and `rejected/` layers with exact error rationale.
- **Feature Engineering**: Derivation of domain metrics (Internal Marks Average = $(G1+G2)/2$, Attendance Rate %, Grade Progression $= G2 - G1$, and a 0–100 Composite Dropout Risk Score).
- **Database Schema**: 3NF normalized tables (`students`, `academic_records`, `attendance_lifestyle`) and a flattened data mart (`analytics_student_performance`) with foreign keys and check constraints.
- **Orchestration**: Apache Airflow DAG managing task dependencies from extraction to analytical mart creation.
- **Data Visualization**: Streamlit dashboard presenting at least five distinct analytical views.

---

## 3. Dataset Description & Provenance

### 3.1 Dataset Selection
- **Dataset Name**: Student Performance Data Set (Mathematics Subject).
- **Origin**: UCI Machine Learning Repository (Cortez and Silva, 2008).
- **Primary Domain**: Secondary education performance in Portuguese schools (Gabriel Pereira `GP` and Mousinho da Silveira `MS`).
- **File Format**: Semicolon-delimited CSV (`student-mat.csv`).
- **Raw Observations**: 395 student instances across 33 initial attributes.

### 3.2 Key Attributes & Semantics

| Column Category | Specific Columns | Description & Domain Range |
| :--- | :--- | :--- |
| **Demographics** | `school`, `sex`, `age`, `address`, `famsize`, `Pstatus` | Student school (`GP`, `MS`), gender (`F`, `M`), age (15–22), residential address (`U`, `R`), family size (`LE3`, `GT3`), parental status (`T`, `A`). |
| **Social & Family** | `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `guardian` | Parental education (0=none to 4=higher education), occupations, school choice reasons, primary guardian. |
| **Academic Habits** | `traveltime`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities` | Travel time (1–4), weekly study time (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h), past class failures (0–4), educational support flags (`yes`/`no`). |
| **Attendance & Health** | `absences`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health` | Number of school absences (0–93), family relationship quality (1–5), alcohol consumption (1–5), health rating (1–5). |
| **Evaluations / Targets**| `G1`, `G2`, `G3` | First period grade (0–20), Second period grade (0–20), Final course grade (0–20). |

---

## 4. System Architecture

The pipeline follows a modern multi-layer Data Lakehouse & Warehousing architectural design:

```
[UCI Machine Learning Repository]
               │
               ▼ (HTTP Ingestion Engine - src/ingestion.py)
[Raw Layer: data/raw/student-mat.csv (Immutable)] ──> [Audit Log: ingestion_log.json]
               │
               ▼ (Staging Layer - data/staging/student_staging.csv)
[Quality Validation Engine - src/validation.py]
         ├── (Failed Rules) ──> [Rejected Layer: data/rejected/rejected_records.csv]
         └── (Passed Rules) ──> [Cleaned Layer: data/cleaned/student_cleaned.csv]
                                       │
                                       ▼ (Transformation Engine - src/transformation.py)
                                [Analytical Feature Engineering & Mart Generation]
                                       │
                                       ▼ (Database Loader - src/database.py)
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
          [PostgreSQL Relational DB]           [Analytics Data Mart]
          - students (Demographics)            - analytics_student_performance
          - academic_records (Grades)          - Analytical SQL Views
          - attendance_lifestyle (Absences)
                     │
                     ▼ (Direct SQL Query Layer)
          [Streamlit Interactive Dashboard - dashboard/app.py]
```

### Layer Responsibilities:
1. **Source / Ingestion Layer**: Fetches raw data over HTTPS, validates payload integrity, and maintains unmodified raw archives.
2. **Raw & Staging Layer**: Stores the immutable baseline and generates synthetic primary keys (`STU0001` .. `STU0395`).
3. **Quality & Validation Layer**: Applies strict predicate checks (ranges, domains, duplicate profiles).
4. **Transformation Layer**: Computes higher-order aggregations and risk metrics.
5. **Storage & Data Mart Layer**: Normalized PostgreSQL tables enforcing relational integrity, alongside an indexed data mart table.
6. **Presentation Layer**: Streamlit web dashboard providing real-time data exploration and early-warning intervention lists.
7. **Orchestration**: Apache Airflow DAG (`student_performance_pipeline`) ensuring strict linear dependency execution.

---

## 5. Ingestion Engine Implementation

The ingestion process is encapsulated within `src/ingestion.py`.

### 5.1 Ingestion Workflow
1. Verifies if local raw files exist; if absent, streams the archive from the UCI remote server.
2. Unpacks the archive into `data/raw/` without altering delimiters, headers, or record values.
3. Assigns standardized surrogate keys (`student_id`) in `data/staging/student_staging.csv`.
4. Emits a structured JSON audit file (`data/raw/ingestion_log.json`) recording:
   - `dataset_name`
   - `source`
   - `extraction_date` (ISO-8601 timestamp)
   - `status` (`SUCCESS` or `FAILED`)
   - `row_count` (395 records verified)
   - `column_count` (33 features)
   - `error_message` (if encountered)

---

## 6. Data Quality Validation & Error Isolation

The validation engine in `src/validation.py` operates as an automated quality gate between staging and cleaned storage.

### 6.1 Validation Rules Matrix

| Check Identifier | Target Columns | Rule Specification | Failure Action |
| :--- | :--- | :--- | :--- |
| **VR-01: Identifier** | `student_id` | Must be non-null, non-empty string. | Reject record |
| **VR-02: Grade Bounds** | `G1`, `G2`, `G3` | Must be numeric and $0.0 \le G \le 20.0$. | Reject record |
| **VR-03: Absences** | `absences` | Must be non-negative integer ($0 \le \text{absences} \le 93$). | Reject record |
| **VR-04: Age Bounds** | `age` | Must be integer between 10 and 30. | Reject record |
| **VR-05: Likert Scales** | `studytime`, `health`, etc. | Must conform to 1–4 or 1–5 discrete bounds. | Reject record |
| **VR-06: Categorical Domains** | `school`, `sex`, `address`, `Pstatus`, etc. | Must exist within defined enumeration sets. | Reject record |
| **VR-07: Uniqueness** | Demographics tuple | Duplicate demographic vectors are flagged. | Move copy to rejected |

### 6.2 Rejected Record Layer
Any record violating one or more validation rules is appended to `data/rejected/rejected_records.csv` alongside a concatenated `rejection_reason` column (e.g., `"Invalid grade G3=25; Negative absences=-5"`). A summary report is saved to `data/rejected/validation_summary.json`.

---

## 7. Data Transformation & Feature Engineering

The transformation module in `src/transformation.py` prepares the analytical dataset by computing derived metrics:

### 7.1 Mathematical Formulations

1. **Average Internal Marks**:
   $$\text{internal\_marks\_avg} = \frac{G1 + G2}{2}$$
2. **Overall Academic Average**:
   $$\text{overall\_academic\_avg} = \frac{G1 + G2 + G3}{3}$$
3. **Estimated Attendance Percentage**:
   Based on the maximum observed school sessions ($N = 93$ days):
   $$\text{estimated\_attendance\_pct} = \max\left(0.0, \left(1.0 - \frac{\text{absences}}{93}\right) \times 100\right)$$
4. **Period Grade Progression (Trend Analysis)**:
   $$\text{term1\_to\_term2\_trend} = G2 - G1$$
   $$\text{term2\_to\_final\_trend} = G3 - G2$$
   $$\text{overall\_grade\_growth} = G3 - G1$$
5. **Academic Categorization**:
   - **Excellent**: $\text{overall\_academic\_avg} \ge 16.0$
   - **Good**: $14.0 \le \text{overall\_academic\_avg} < 16.0$
   - **Satisfactory**: $10.0 \le \text{overall\_academic\_avg} < 14.0$
   - **Fail / At-Risk**: $\text{overall\_academic\_avg} < 10.0$
6. **Composite Dropout Risk Score ($0 - 100$)**:
   $$\text{Risk Score} = \underbrace{\left(\frac{20 - \text{internal\_marks\_avg}}{20} \times 45\right)}_{\text{Academic Deficit (45\%)}} + \underbrace{\left(\frac{\min(\text{absences}, 30)}{30} \times 30\right)}_{\text{Absence Factor (30\%)}} + \underbrace{\left(\frac{\min(\text{failures}, 3)}{3} \times 15\right)}_{\text{Failure History (15\%)}} + \underbrace{\left(\frac{4 - \text{studytime}}{3} \times 10\right)}_{\text{Study Deficit (10\%)}}$$

---

## 8. Relational Database Design (PostgreSQL)

The database schema (`sql/schema.sql`) implements a normalized structure adhering to Third Normal Form (3NF) to eliminate redundancy, while maintaining an optimized data mart table for analytical querying.

### 8.1 Entity-Relationship Overview
- **`students` Table (Demographic Dimension)**: Primary Key `student_id`. Stores invariant personal and parental attributes.
- **`academic_records` Table (Academic Dimension)**: Primary Key `student_id` (Foreign Key referencing `students`). Stores term evaluations ($G1, G2, G3$), study habits, and support structures.
- **`attendance_lifestyle` Table (Behavioral Dimension)**: Primary Key `student_id` (Foreign Key referencing `students`). Stores absences, family relations, alcohol intake, and attendance percentages.
- **`analytics_student_performance` Table (Data Mart)**: Denormalized analytical table populated with pre-joined demographic, academic, and feature-engineered risk columns for low-latency dashboard querying.

### 8.2 Database Access Layer & Offline Resilience
The module `src/database.py` utilizes SQLAlchemy. If a remote PostgreSQL instance is unreachable (e.g. during local offline evaluation), the database engine **gracefully falls back to an embedded SQLite instance** (`data/analytics/student_performance.db`), guaranteeing zero crashes while preserving exact SQL semantics.

---

## 9. Analytical Data Mart & Views

The script `sql/analytics.sql` defines optimized views to power institutional reporting:
1. **`view_school_performance_summary`**: Aggregates total enrollment, average final grade, pass rates, and high-risk student counts by school cohort.
2. **`view_dropout_risk_breakdown`**: Summarizes average grades, absences, and study hours partitioned by risk tiers.
3. **`view_students_requiring_intervention`**: Returns an ordered list of students flagged for immediate counselor intervention based on composite risk indices.

---

## 10. Streamlit Interactive Dashboard

The dashboard (`dashboard/app.py`) interfaces directly with the PostgreSQL / SQLite data mart, offering five interactive analytical views:

1. **View 1: Overall Performance & Cohort KPIs**
   - High-level metric cards: Total Students ($395$), Average Final Grade ($10.42/20$), Pass Rate ($67.1\%$), High-Risk Students ($47$), Average Attendance ($93.9\%$).
   - Performance tier distribution donut chart and grade histogram by school.
2. **View 2: Attendance & Lifestyle Impact**
   - Scatter plot correlating absences with final course grades ($G3$).
   - Box plots analyzing the effect of weekly study tiers and weekend alcohol consumption on grades.
3. **View 3: Period Grade Trajectory**
   - Multi-term trajectory line graph tracking cohort progression across Period 1 ($G1$), Period 2 ($G2$), and Final ($G3$).
   - Progression distribution highlighting positive and negative score differentials.
4. **View 4: Early Warning & Intervention Roster**
   - Filterable, sortable student roster highlighting high-risk profiles in real-time.
   - One-click CSV export for academic counselors.
5. **View 5: Individual Student Drill-Down Profile**
   - Detailed profile card with demographic breakdown, term grade trajectory chart, and automated intervention recommendations.

---

## 11. Pipeline Orchestration (Apache Airflow)

![Apache Airflow DAG Graph View](docs/airflow_dag_success.png)

The workflow is orchestrated via `dags/student_pipeline.py`.

```
[ingest_raw_data] ──> [validate_data_quality] ──> [transform_feature_mart] ──> [load_postgresql_database]
```

Each task is defined using `PythonOperator` with clear documentation metadata and retry logic ($1$ retry after $2$ minutes). A standalone execution harness is built-in to facilitate verification in non-Airflow environments.

---

## 12. Verification, Testing & Empirical Evidence

### 12.1 Automated Test Suite Results
The test suite (`tests/test_pipeline.py`) covers unit-level logic and integration pipelines:
- **`test_01_ingestion`**: Passed. Verified raw file presence and metadata logging.
- **`test_02_validation_rules`**: Passed. Verified correct handling of valid rows and rejection of out-of-bounds inputs ($G3=25$, $\text{absences}=-5$).
- **`test_03_validation_pipeline`**: Passed. Verified 395 records evaluated.
- **`test_04_transformation_features`**: Passed. Verified feature calculation and categorization logic.
- **`test_05_database_loading`**: Passed. Verified relational table creation and data mart population.

**Execution Output**: `Ran 5 tests in 5.341s — OK (100% Pass Rate)`.

### 12.2 Empirical Pipeline Output Summary

| Stage | Input Records | Output Records | Key Metric / Output File |
| :--- | :--- | :--- | :--- |
| **Ingestion** | Remote ZIP | 395 rows | `data/raw/student-mat.csv` |
| **Validation** | 395 raw rows | 395 clean, 0 rejected | `data/cleaned/student_cleaned.csv` |
| **Transformation**| 395 clean rows | 395 mart rows | `data/analytics/analytics_student_performance.csv` |
| **Database Load** | 395 mart rows | 4 tables populated | `students`, `academic_records`, `attendance_lifestyle`, `analytics_student_performance` |

---

## 13. Limitations & Constraints

1. **Dataset Granularity**: The UCI dataset represents an annual cohort with three discrete evaluation periods ($G1, G2, G3$). Continuous daily assignment submission records are not part of the empirical benchmark; attendance is represented by aggregate absences.
2. **Surrogate Identifiers**: The original dataset anonymizes student identity without unique student keys. Surrogate identifiers (`STU0001` .. `STU0395`) were deterministically generated during ingestion.
3. **Database Driver Fallback**: For standalone evaluation environments where a local PostgreSQL daemon is offline, the pipeline automatically provides SQLite fallback to ensure uninterrupted review.

---

## 14. Conclusion & Transition to Part 2

Part 1 has successfully implemented and validated an end-to-end Data Engineering pipeline for the Student Performance and Dropout-Risk Prediction System. All data layers (raw, staging, cleaned, rejected, analytics) are strictly maintained, automated validations prevent data contamination, relational database schemas enforce structural integrity, and the Streamlit dashboard provides actionable institutional insights.

This solid data engineering foundation will directly feed **Part 2**, where machine learning models (XGBoost, Random Forest, Logistic Regression), experiment tracking with MLflow, FastAPI prediction endpoints, Docker containerization, and model monitoring will be introduced.
