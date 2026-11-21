-- =============================================================================
-- SQL Sample Verification Queries
-- Database: student_performance_db
-- =============================================================================

-- 1. Row counts across all core and analytical tables
SELECT 'students' AS table_name, COUNT(*) AS total_records FROM students
UNION ALL
SELECT 'academic_records', COUNT(*) FROM academic_records
UNION ALL
SELECT 'attendance_lifestyle', COUNT(*) FROM attendance_lifestyle
UNION ALL
SELECT 'analytics_student_performance', COUNT(*) FROM analytics_student_performance;

-- 2. Verify Grade distribution and Pass rates
SELECT 
    school,
    COUNT(*) AS student_count,
    ROUND(AVG(g1), 2) AS avg_period1,
    ROUND(AVG(g2), 2) AS avg_period2,
    ROUND(AVG(g3), 2) AS avg_final,
    MIN(g3) AS min_grade,
    MAX(g3) AS max_grade,
    ROUND(COUNT(CASE WHEN g3 >= 10 THEN 1 END) * 100.0 / COUNT(*), 2) AS pass_percentage
FROM academic_records ar
JOIN students s ON ar.student_id = s.student_id
GROUP BY school;

-- 3. Top 10 High-Risk Students requiring academic intervention
SELECT 
    student_id,
    school,
    final_grade,
    internal_marks_avg,
    absences,
    estimated_attendance_pct,
    dropout_risk_score,
    dropout_risk_level,
    performance_category
FROM analytics_student_performance
WHERE dropout_risk_level = 'High Risk'
ORDER BY dropout_risk_score DESC, final_grade ASC
LIMIT 10;

-- 4. Impact of Absences & Study Time on Final Performance
SELECT 
    studytime,
    CASE 
        WHEN absences = 0 THEN '0 (Perfect)'
        WHEN absences BETWEEN 1 AND 5 THEN '1 - 5'
        WHEN absences BETWEEN 6 AND 15 THEN '6 - 15'
        ELSE '16+'
    END AS absence_tier,
    COUNT(*) AS student_count,
    ROUND(AVG(g3), 2) AS avg_final_grade,
    ROUND(AVG(dropout_risk_score), 2) AS avg_risk_score
FROM analytics_student_performance
GROUP BY studytime, absence_tier
ORDER BY studytime ASC, avg_final_grade DESC;
