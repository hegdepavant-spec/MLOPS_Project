-- =============================================================================
-- SQL Analytics: Analytical Data Mart Creation & Aggregations
-- Database: student_performance_db
-- =============================================================================

DROP TABLE IF EXISTS analytics_student_performance CASCADE;

-- Analytical Data Mart Table
CREATE TABLE analytics_student_performance (
    student_id VARCHAR(10) PRIMARY KEY,
    school VARCHAR(5) NOT NULL,
    sex VARCHAR(1) NOT NULL,
    age INT NOT NULL,
    address VARCHAR(1) NOT NULL,
    famsize VARCHAR(5) NOT NULL,
    pstatus VARCHAR(1) NOT NULL,
    medu INT NOT NULL,
    fedu INT NOT NULL,
    mjob VARCHAR(30),
    fjob VARCHAR(30),
    reason VARCHAR(30),
    guardian VARCHAR(30),
    traveltime INT,
    studytime INT NOT NULL,
    failures INT NOT NULL,
    schoolsup VARCHAR(5),
    famsup VARCHAR(5),
    paid VARCHAR(5),
    activities VARCHAR(5),
    nursery VARCHAR(5),
    higher VARCHAR(5),
    internet VARCHAR(5),
    romantic VARCHAR(5),
    absences INT NOT NULL,
    famrel INT,
    freetime INT,
    goout INT,
    dalc INT,
    walc INT,
    health INT,
    g1 INT NOT NULL,
    g2 INT NOT NULL,
    g3 INT NOT NULL,
    subject VARCHAR(30) DEFAULT 'Mathematics',
    semester VARCHAR(20) DEFAULT 'Annual',
    internal_marks_avg NUMERIC(5,2) NOT NULL,
    overall_academic_avg NUMERIC(5,2) NOT NULL,
    final_grade INT NOT NULL,
    estimated_attendance_pct NUMERIC(5,2) NOT NULL,
    term1_to_term2_trend NUMERIC(5,2),
    term2_to_final_trend NUMERIC(5,2),
    overall_grade_growth NUMERIC(5,2),
    performance_category VARCHAR(30) NOT NULL,
    final_performance_category VARCHAR(30) NOT NULL,
    pass_status VARCHAR(10) NOT NULL,
    dropout_risk_score NUMERIC(5,2) NOT NULL,
    dropout_risk_level VARCHAR(20) NOT NULL,
    requires_intervention BOOLEAN NOT NULL,
    school_cohort_avg_grade NUMERIC(5,2),
    deviation_from_cohort_avg NUMERIC(5,2),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analytical View 1: School & Subject Summary KPIs
CREATE OR REPLACE VIEW view_school_performance_summary AS
SELECT 
    school,
    subject,
    COUNT(student_id) AS total_students,
    ROUND(AVG(final_grade), 2) AS avg_final_grade,
    ROUND(AVG(internal_marks_avg), 2) AS avg_internal_grade,
    ROUND(AVG(estimated_attendance_pct), 2) AS avg_attendance_pct,
    ROUND(SUM(CASE WHEN pass_status = 'Pass' THEN 1 ELSE 0 END)::NUMERIC / COUNT(student_id) * 100, 2) AS pass_rate_pct,
    SUM(CASE WHEN dropout_risk_level = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_students_count,
    SUM(CASE WHEN requires_intervention THEN 1 ELSE 0 END) AS intervention_required_count
FROM analytics_student_performance
GROUP BY school, subject;

-- Analytical View 2: Dropout Risk & Attendance Correlation Mart
CREATE OR REPLACE VIEW view_dropout_risk_breakdown AS
SELECT 
    dropout_risk_level,
    performance_category,
    COUNT(student_id) AS student_count,
    ROUND(AVG(g3), 2) AS avg_final_score,
    ROUND(AVG(absences), 2) AS avg_absences,
    ROUND(AVG(studytime), 2) AS avg_study_time,
    ROUND(AVG(failures), 2) AS avg_prior_failures
FROM analytics_student_performance
GROUP BY dropout_risk_level, performance_category
ORDER BY avg_final_score ASC;

-- Analytical View 3: High-Risk Intervention List
CREATE OR REPLACE VIEW view_students_requiring_intervention AS
SELECT 
    student_id,
    school,
    sex,
    age,
    g1,
    g2,
    g3 AS final_grade,
    internal_marks_avg,
    absences,
    estimated_attendance_pct,
    failures,
    dropout_risk_score,
    dropout_risk_level,
    performance_category,
    pass_status
FROM analytics_student_performance
WHERE requires_intervention = TRUE
ORDER BY dropout_risk_score DESC, final_grade ASC;
