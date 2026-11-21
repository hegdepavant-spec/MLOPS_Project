-- =============================================================================
-- SQL Schema: Student Performance Data Pipeline (PostgreSQL)
-- Database: student_performance_db
-- =============================================================================

-- Drop tables if they exist to allow clean replay
DROP TABLE IF EXISTS analytics_student_performance CASCADE;
DROP TABLE IF EXISTS attendance_lifestyle CASCADE;
DROP TABLE IF EXISTS academic_records CASCADE;
DROP TABLE IF EXISTS students CASCADE;

-- 1. Student Demographics Dimension
CREATE TABLE students (
    student_id VARCHAR(10) PRIMARY KEY,
    school VARCHAR(5) NOT NULL CHECK (school IN ('GP', 'MS')),
    sex VARCHAR(1) NOT NULL CHECK (sex IN ('F', 'M')),
    age INT NOT NULL CHECK (age BETWEEN 10 AND 30),
    address VARCHAR(1) NOT NULL CHECK (address IN ('U', 'R')),
    famsize VARCHAR(5) NOT NULL CHECK (famsize IN ('LE3', 'GT3')),
    pstatus VARCHAR(1) NOT NULL CHECK (pstatus IN ('T', 'A')),
    medu INT NOT NULL CHECK (medu BETWEEN 0 AND 4),
    fedu INT NOT NULL CHECK (fedu BETWEEN 0 AND 4),
    mjob VARCHAR(30) NOT NULL,
    fjob VARCHAR(30) NOT NULL,
    reason VARCHAR(30) NOT NULL,
    guardian VARCHAR(30) NOT NULL,
    traveltime INT NOT NULL CHECK (traveltime BETWEEN 1 AND 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Academic Performance and Support Records
CREATE TABLE academic_records (
    student_id VARCHAR(10) PRIMARY KEY REFERENCES students(student_id) ON DELETE CASCADE,
    subject VARCHAR(30) NOT NULL DEFAULT 'Mathematics',
    studytime INT NOT NULL CHECK (studytime BETWEEN 1 AND 4),
    failures INT NOT NULL CHECK (failures BETWEEN 0 AND 4),
    schoolsup VARCHAR(5) NOT NULL CHECK (schoolsup IN ('yes', 'no')),
    famsup VARCHAR(5) NOT NULL CHECK (famsup IN ('yes', 'no')),
    paid VARCHAR(5) NOT NULL CHECK (paid IN ('yes', 'no')),
    activities VARCHAR(5) NOT NULL CHECK (activities IN ('yes', 'no')),
    nursery VARCHAR(5) NOT NULL CHECK (nursery IN ('yes', 'no')),
    higher VARCHAR(5) NOT NULL CHECK (higher IN ('yes', 'no')),
    internet VARCHAR(5) NOT NULL CHECK (internet IN ('yes', 'no')),
    romantic VARCHAR(5) NOT NULL CHECK (romantic IN ('yes', 'no')),
    g1 INT NOT NULL CHECK (g1 BETWEEN 0 AND 20),
    g2 INT NOT NULL CHECK (g2 BETWEEN 0 AND 20),
    g3 INT NOT NULL CHECK (g3 BETWEEN 0 AND 20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Attendance and Student Lifestyle Behavior
CREATE TABLE attendance_lifestyle (
    student_id VARCHAR(10) PRIMARY KEY REFERENCES students(student_id) ON DELETE CASCADE,
    absences INT NOT NULL CHECK (absences >= 0),
    famrel INT NOT NULL CHECK (famrel BETWEEN 1 AND 5),
    freetime INT NOT NULL CHECK (freetime BETWEEN 1 AND 5),
    goout INT NOT NULL CHECK (goout BETWEEN 1 AND 5),
    dalc INT NOT NULL CHECK (dalc BETWEEN 1 AND 5),
    walc INT NOT NULL CHECK (walc BETWEEN 1 AND 5),
    health INT NOT NULL CHECK (health BETWEEN 1 AND 5),
    estimated_attendance_pct NUMERIC(5,2) NOT NULL CHECK (estimated_attendance_pct BETWEEN 0 AND 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for query performance
CREATE INDEX idx_students_school ON students(school);
CREATE INDEX idx_academic_g3 ON academic_records(g3);
CREATE INDEX idx_attendance_absences ON attendance_lifestyle(absences);
