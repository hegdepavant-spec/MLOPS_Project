# System Architecture — Student Performance & Dropout-Risk Data Pipeline

![System Architecture Diagram](docs/architecture.png)

## 1. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph S1["1. DATA SOURCE LAYER"]
        A["UCI Machine Learning Repository<br/>(Student Performance Dataset)"]
    end

    subgraph S2["2. INGESTION LAYER"]
        B["Python Ingestion Engine<br/>(src/ingestion.py)"]
        B_log["Metadata & Audit Logger<br/>(extraction date, source, status, count)"]
    end

    subgraph S3["3. STORAGE & STAGING LAYER"]
        C["Raw Layer (Immutable)<br/>data/raw/student-mat.csv"]
        D["Staging Layer<br/>data/staging/student_staging.csv"]
    end

    subgraph S4["4. QUALITY & TRANSFORMATION LAYER"]
        E["Validation Engine<br/>(src/validation.py)"]
        E_rej["Rejected Layer<br/>data/rejected/rejected_records.csv"]
        E_clean["Cleaned Layer<br/>data/cleaned/student_cleaned.csv"]
        F["Transformation Engine<br/>(src/transformation.py)"]
    end

    subgraph S5["5. POSTGRESQL DATABASE & DATA MART"]
        G1["Demographics<br/>(students)"]
        G2["Academics<br/>(academic_records)"]
        G3["Attendance<br/>(attendance_lifestyle)"]
        H["Analytical Data Mart<br/>(analytics_student_performance)"]
    end

    subgraph S6["6. PRESENTATION & APPLICATION LAYER"]
        I["Streamlit Interactive Dashboard<br/>(dashboard/app.py)"]
    end

    subgraph S7["7. ORCHESTRATION LAYER"]
        DAG["Apache Airflow Workflow<br/>(dags/student_pipeline.py)"]
    end

    A -->|HTTP / TLS Download| B
    B --> B_log
    B -->|Persist Raw Copy| C
    B -->|Assign IDs & Stage| D
    D -->|Audit Schema & Domain| E
    E -->|Failed Quality Rules| E_rej
    E -->|Passed Validations| E_clean
    E_clean -->|Feature Engineering & Aggregation| F
    F -->|Relational ETL Load| G1
    F -->|Relational ETL Load| G2
    F -->|Relational ETL Load| G3
    F -->|Populate Data Mart| H
    H -->|Direct SQL Query| I

    DAG -.->|Orchestrates| B
    DAG -.->|Orchestrates| E
    DAG -.->|Orchestrates| F
    DAG -.->|Orchestrates| H
```

---

## 2. Layer-by-Layer Technical Specification

| Layer | Component | Implementation | Responsibility |
| :--- | :--- | :--- | :--- |
| **Source** | UCI Repository | Remote HTTP Endpoint | Original benchmark student data |
| **Ingestion** | `src/ingestion.py` | Python 3 + Pandas + Urllib | Download archive, unpack, record audit metadata without altering raw file |
| **Raw & Staging** | `data/raw/`, `data/staging/` | CSV + JSON | Immutable preservation of historical baseline + staged data with unique student keys |
| **Validation** | `src/validation.py` | Data Quality Rules Engine | Check duplicate students, numeric bounds (0-20 grades), category domains, and log errors |
| **Transformation** | `src/transformation.py` | Analytical Feature Generator | Derive attendance %, internal marks average, period progression trends, and dropout risk score |
| **Storage** | PostgreSQL / SQLite | Normalized Relational Schema | 3 normalized tables (`students`, `academic_records`, `attendance_lifestyle`) with strict constraints |
| **Analytics Mart** | `analytics_student_performance` | SQL Table + Views | Pre-aggregated flat data mart feeding dashboard views |
| **Dashboard** | `dashboard/app.py` | Streamlit + Plotly | 5 interactive views for cohort KPIs, risk drill-downs, and student profiles |
| **Orchestration** | `dags/student_pipeline.py` | Apache Airflow DAG | Automated sequential execution and dependency management |
