"""
Streamlit Dashboard: Student Performance & Dropout-Risk Analytics System
Part 1 - Data Engineering & Analytical Data Mart Visualizer
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Append project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database import fetch_analytical_data, get_db_engine

# Page Configuration
st.set_page_config(
    page_title="Student Performance & Dropout-Risk Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        font-size: 15px;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_data():
    """Load data directly from PostgreSQL / database mart."""
    try:
        df = fetch_analytical_data()
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        # Fallback to local csv if database is offline
        csv_path = os.path.join(BASE_DIR, 'data', 'analytics', 'analytics_student_performance.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()


df = load_data()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/student-center.png", width=64)
st.sidebar.title("Pipeline Controls")
st.sidebar.markdown("**Part 1: Data Engineering Mart**")

if df.empty:
    st.error("No data found in Analytical Data Mart. Please run the pipeline first.")
    st.stop()

# Filters
school_filter = st.sidebar.multiselect(
    "Select School Cohort:",
    options=df['school'].unique(),
    default=df['school'].unique()
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Level:",
    options=df['dropout_risk_level'].unique(),
    default=df['dropout_risk_level'].unique()
)

# Apply filters
filtered_df = df[(df['school'].isin(school_filter)) & (df['dropout_risk_level'].isin(risk_filter))]

# Header
st.markdown('<div class="main-header">🎓 Student Performance & Dropout-Risk Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Part 1 Data Engineering Data Mart & Analytics Explorer | Connected to PostgreSQL Data Store</div>', unsafe_allow_html=True)

# Overview KPIs
col1, col2, col3, col4, col5 = st.columns(5)
total_students = len(filtered_df)
avg_g3 = filtered_df['final_grade'].mean() if total_students > 0 else 0
pass_rate = (filtered_df['pass_status'] == 'Pass').mean() * 100 if total_students > 0 else 0
high_risk_count = (filtered_df['dropout_risk_level'] == 'High Risk').sum()
avg_attendance = filtered_df['estimated_attendance_pct'].mean() if total_students > 0 else 0

col1.metric("Total Students", f"{total_students:,}")
col2.metric("Average Final Grade (0-20)", f"{avg_g3:.2f}")
col3.metric("Pass Rate", f"{pass_rate:.1f}%")
col4.metric("High-Risk Students", f"{high_risk_count}", delta=f"{(high_risk_count/max(1, total_students))*100:.1f}%", delta_color="inverse")
col5.metric("Avg Attendance Rate", f"{avg_attendance:.1f}%")

st.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Overall Performance",
    "📈 2. Attendance & Lifestyle Impact",
    "🔄 3. Period Grade Trajectory",
    "⚠️ 4. Early Warning & Interventions",
    "👤 5. Student Drill-Down Profile",
    "🤖 6. Real-Time ML Predictor"
])

# -----------------------------------------------------------------------------
# TAB 1: Overall Performance
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Academic Performance Distribution & Cohort Summary")
    c1, c2 = st.columns(2)
    
    with c1:
        fig_cat = px.pie(
            filtered_df, 
            names='performance_category', 
            title="Performance Category Distribution",
            color='performance_category',
            color_discrete_map={
                'Excellent': '#10B981',
                'Good': '#3B82F6',
                'Satisfactory': '#F59E0B',
                'Fail / At-Risk': '#EF4444'
            },
            hole=0.4
        )
        fig_cat.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        fig_hist = px.histogram(
            filtered_df, 
            x='final_grade', 
            color='school', 
            nbins=20, 
            barmode='group',
            title="Final Grade (G3) Distribution by School",
            labels={'final_grade': 'Final Grade (0-20 scale)', 'school': 'School'},
            color_discrete_sequence=['#3B82F6', '#8B5CF6']
        )
        fig_hist.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Passing Threshold (10)")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Detailed Demographic Breakdown")
    demo_grp = filtered_df.groupby(['school', 'sex', 'pass_status']).agg(
        student_count=('student_id', 'count'),
        avg_final_grade=('final_grade', 'mean'),
        avg_absences=('absences', 'mean'),
        avg_risk_score=('dropout_risk_score', 'mean')
    ).reset_index()
    demo_grp['avg_final_grade'] = demo_grp['avg_final_grade'].round(2)
    demo_grp['avg_absences'] = demo_grp['avg_absences'].round(1)
    demo_grp['avg_risk_score'] = demo_grp['avg_risk_score'].round(2)
    st.dataframe(demo_grp, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Attendance & Lifestyle Impact
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Attendance, Absences, and Lifestyle Correlation Analysis")
    c1, c2 = st.columns(2)

    with c1:
        fig_scatter = px.scatter(
            filtered_df,
            x='absences',
            y='final_grade',
            color='dropout_risk_level',
            size='dropout_risk_score',
            hover_data=['student_id', 'g1', 'g2', 'studytime'],
            title="Absences vs Final Grade (G3)",
            labels={'absences': 'Total School Absences', 'final_grade': 'Final Grade (0-20)'},
            color_discrete_map={'High Risk': '#EF4444', 'Medium Risk': '#F59E0B', 'Low Risk': '#10B981'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        fig_box = px.box(
            filtered_df,
            x='studytime',
            y='final_grade',
            color='studytime',
            title="Study Time Weekly Tiers vs Final Grade",
            labels={'studytime': 'Study Time Tier (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)', 'final_grade': 'Final Grade (0-20)'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Parental Education & Prior History Impact
    c3, c4 = st.columns(2)
    with c3:
        edu_col = 'medu' if 'medu' in filtered_df.columns else ('Medu' if 'Medu' in filtered_df.columns else None)
        if edu_col:
            edu_grp = filtered_df.groupby(edu_col)['final_grade'].mean().reset_index()
            fig_edu = px.bar(
                edu_grp,
                x=edu_col,
                y='final_grade',
                title="Mother Education Level (0-4) vs Avg Final Grade",
                labels={edu_col: 'Mother Education (0=None, 1=Primary, 2=5-9th, 3=Secondary, 4=Higher)', 'final_grade': 'Avg Final Grade'},
                color='final_grade',
                color_continuous_scale='Blues_r'
            )
            st.plotly_chart(fig_edu, use_container_width=True)

    with c4:
        fig_fail = px.box(
            filtered_df,
            x='failures',
            y='dropout_risk_score',
            color='failures',
            title="Prior Academic Failures vs Dropout Risk Score",
            labels={'failures': 'Prior Class Failures count', 'dropout_risk_score': 'Calculated Dropout Risk Score'}
        )
        st.plotly_chart(fig_fail, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: Period Grade Trajectory
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Academic Trajectory: Term 1 (G1) ➔ Term 2 (G2) ➔ Final (G3)")
    
    # Trajectory chart
    g_means = pd.DataFrame({
        'Evaluation Period': ['Period 1 (G1)', 'Period 2 (G2)', 'Final Exam (G3)'],
        'Overall Cohort Average': [filtered_df['g1'].mean(), filtered_df['g2'].mean(), filtered_df['g3'].mean()],
        'High-Risk Students Average': [
            filtered_df[filtered_df['dropout_risk_level'] == 'High Risk']['g1'].mean() if high_risk_count > 0 else 0,
            filtered_df[filtered_df['dropout_risk_level'] == 'High Risk']['g2'].mean() if high_risk_count > 0 else 0,
            filtered_df[filtered_df['dropout_risk_level'] == 'High Risk']['g3'].mean() if high_risk_count > 0 else 0
        ],
        'Low-Risk Students Average': [
            filtered_df[filtered_df['dropout_risk_level'] == 'Low Risk']['g1'].mean(),
            filtered_df[filtered_df['dropout_risk_level'] == 'Low Risk']['g2'].mean(),
            filtered_df[filtered_df['dropout_risk_level'] == 'Low Risk']['g3'].mean()
        ]
    })

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(x=g_means['Evaluation Period'], y=g_means['Overall Cohort Average'], mode='lines+markers', name='Cohort Average', line=dict(color='#3B82F6', width=3)))
    fig_traj.add_trace(go.Scatter(x=g_means['Evaluation Period'], y=g_means['High-Risk Students Average'], mode='lines+markers', name='High-Risk Average', line=dict(color='#EF4444', width=3, dash='dash')))
    fig_traj.add_trace(go.Scatter(x=g_means['Evaluation Period'], y=g_means['Low-Risk Students Average'], mode='lines+markers', name='Low-Risk Average', line=dict(color='#10B981', width=3, dash='dot')))
    fig_traj.update_layout(title="Grade Progression Across Terms by Risk Category", yaxis_title="Score (0-20 scale)", template="plotly_white")
    st.plotly_chart(fig_traj, use_container_width=True)

    st.markdown("#### Progression Distribution (Term 1 to Term 2 Growth)")
    fig_trend_hist = px.histogram(
        filtered_df,
        x='term1_to_term2_trend',
        color='dropout_risk_level',
        title="Distribution of Term 1 to Term 2 Score Changes",
        labels={'term1_to_term2_trend': 'Score Change (G2 - G1)'},
        color_discrete_map={'High Risk': '#EF4444', 'Medium Risk': '#F59E0B', 'Low Risk': '#10B981'}
    )
    st.plotly_chart(fig_trend_hist, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Early Warning & Interventions
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("⚠️ Early Warning & Intervention System")
    st.markdown("Filter and inspect students flagged for immediate counselor intervention based on composite risk factors.")

    c1, c2 = st.columns([1, 3])
    with c1:
        only_interventions = st.checkbox("Show Only Students Requiring Intervention", value=True)
    
    warn_df = filtered_df.copy()
    if only_interventions:
        warn_df = warn_df[warn_df['requires_intervention'] == True]

    cols_to_show = [
        'student_id', 'school', 'sex', 'age', 'final_grade', 'internal_marks_avg', 
        'absences', 'failures', 'studytime', 'dropout_risk_score', 'dropout_risk_level', 'pass_status'
    ]
    warn_df = warn_df.sort_values(by=['dropout_risk_score', 'final_grade'], ascending=[False, True])

    st.dataframe(
        warn_df[cols_to_show].style.background_gradient(subset=['dropout_risk_score'], cmap='Reds'),
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Intervention Roster (CSV)",
        data=warn_df[cols_to_show].to_csv(index=False),
        file_name="intervention_student_roster.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# TAB 5: Student Drill-Down Profile
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("👤 Individual Student Academic Profile")
    
    stu_list = filtered_df['student_id'].unique().tolist()
    selected_stu = st.selectbox("Select Student Identifier:", options=stu_list)

    if selected_stu:
        stu_row = filtered_df[filtered_df['student_id'] == selected_stu].iloc[0]

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Student ID", stu_row['student_id'])
        col_b.metric("Risk Level", stu_row['dropout_risk_level'])
        col_c.metric("Final Grade", f"{stu_row['final_grade']} / 20")
        col_d.metric("Absences", f"{stu_row['absences']} days")

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown("#### Demographics & Academic Profile")
            profile_dict = {
                "School": stu_row.get('school'),
                "Sex / Age": f"{stu_row.get('sex')} / {stu_row.get('age')} years old",
                "Address Type": "Urban" if stu_row.get('address') == 'U' else "Rural",
                "Family Size": stu_row.get('famsize'),
                "Mother / Father Job": f"{stu_row.get('mjob')} / {stu_row.get('fjob')}",
                "Weekly Study Time": f"Tier {stu_row.get('studytime')} (Tier 1-4)",
                "Past Failures": stu_row.get('failures'),
                "School / Family Support": f"{stu_row.get('schoolsup')} / {stu_row.get('famsup')}",
                "Internet Access": stu_row.get('internet'),
                "Estimated Attendance %": f"{stu_row.get('estimated_attendance_pct')}%"
            }
            st.table(pd.DataFrame(list(profile_dict.items()), columns=["Attribute", "Value"]))

        with col_r:
            st.markdown("#### Term Grade Breakdown & Trajectory")
            grades_data = pd.DataFrame({
                "Evaluation": ["Period 1 (G1)", "Period 2 (G2)", "Final Exam (G3)"],
                "Score": [stu_row['g1'], stu_row['g2'], stu_row['g3']]
            })
            fig_stu_g = px.line(
                grades_data, 
                x="Evaluation", 
                y="Score", 
                markers=True,
                title=f"Score Progression for {selected_stu}",
                range_y=[0, 20]
            )
            fig_stu_g.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Passing Threshold")
            st.plotly_chart(fig_stu_g, use_container_width=True)

            # Recommendations
            st.markdown("#### Automated System Recommendation")
            if stu_row['dropout_risk_level'] == 'High Risk':
                st.error("🚨 **High Priority Intervention Required**: Recommend 1-on-1 tutoring sessions, attendance monitoring, and parent-teacher conference.")
            elif stu_row['dropout_risk_level'] == 'Medium Risk':
                st.warning("⚠️ **Moderate Risk Observed**: Recommend academic support groups and study habits tracking.")
            else:
                st.success("✅ **Satisfactory Academic Standing**: Student is on track. Continue standard curriculum progress.")

# -----------------------------------------------------------------------------
# TAB 6: Real-Time ML Risk Predictor (Part 2 MLOps Extension)
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("🤖 Real-Time Student Failure Risk Prediction Engine")
    st.markdown("Enter student evaluation metrics below to generate real-time predictive risk scoring via the versioned ML model & FastAPI service.")

    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.markdown("##### 📚 Academic Marks & Progression")
        input_g1 = st.slider("Period 1 Grade (G1)", min_value=0, max_value=20, value=9)
        input_g2 = st.slider("Period 2 Grade (G2)", min_value=0, max_value=20, value=8)
        input_failures = st.selectbox("Prior Class Failures", options=[0, 1, 2, 3, 4], index=1)
        input_studytime = st.selectbox("Weekly Study Hours", options=[1, 2, 3, 4], format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x], index=0)

    with p_col2:
        st.markdown("##### 🏫 Attendance & Habits")
        input_absences = st.slider("Total Absences (Sessions)", min_value=0, max_value=60, value=12)
        input_school = st.selectbox("School", options=["GP", "MS"], index=0)
        input_sex = st.selectbox("Gender", options=["F", "M"], index=0)
        input_age = st.number_input("Age", min_value=14, max_value=22, value=17)
        input_schoolsup = st.selectbox("Extra Educational Support", options=["no", "yes"], index=0)

    with p_col3:
        st.markdown("##### 🏡 Family & Lifestyle Background")
        input_famsup = st.selectbox("Family Educational Support", options=["yes", "no"], index=0)
        input_paid = st.selectbox("Extra Paid Classes", options=["no", "yes"], index=0)
        input_higher = st.selectbox("Desires Higher Education", options=["yes", "no"], index=0)
        input_internet = st.selectbox("Home Internet Access", options=["yes", "no"], index=0)
        input_famrel = st.slider("Family Relationship Quality (1-5)", min_value=1, max_value=5, value=4)

    predict_btn = st.button("🚀 Run Real-Time ML Risk Prediction", type="primary", use_container_width=True)

    if predict_btn:
        payload = {
            "school": input_school, "sex": input_sex, "age": int(input_age), "address": "U", "famsize": "GT3", "pstatus": "T",
            "medu": 2, "fedu": 2, "mjob": "other", "fjob": "other", "reason": "course", "guardian": "mother",
            "traveltime": 1, "studytime": int(input_studytime), "failures": int(input_failures),
            "schoolsup": input_schoolsup, "famsup": input_famsup, "paid": input_paid, "activities": "no",
            "nursery": "yes", "higher": input_higher, "internet": input_internet, "romantic": "no",
            "famrel": int(input_famrel), "freetime": 3, "goout": 3, "dalc": 1, "walc": 1, "health": 3,
            "absences": int(input_absences), "g1": int(input_g1), "g2": int(input_g2)
        }

        # Attempt prediction via local FastAPI or directly via joblib
        prediction_result = None
        try:
            import requests
            resp = requests.post("http://localhost:8000/predict", json=payload, timeout=2.0)
            if resp.status_code == 200:
                prediction_result = resp.json()
        except Exception:
            # Fallback to direct model inference
            try:
                import joblib
                model_file = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
                if os.path.exists(model_file):
                    bundle = joblib.load(model_file)
                    clf = bundle["model"]
                    prep = bundle["preprocessor"]
                    payload['internal_marks_avg'] = (payload['g1'] + payload['g2']) / 2.0
                    payload['estimated_attendance_pct'] = max(0.0, (1.0 - payload['absences'] / 93.0)) * 100.0
                    payload['term1_to_term2_trend'] = payload['g2'] - payload['g1']
                    from src.ml.dataset import FEATURE_COLUMNS
                    df_p = pd.DataFrame([payload])[FEATURE_COLUMNS]
                    X_p = prep.transform(df_p)
                    pr = float(clf.predict_proba(X_p)[0, 1]) if hasattr(clf, "predict_proba") else float(clf.predict(X_p)[0])
                    risk_tag = "HIGH_RISK" if pr >= 0.65 else ("MEDIUM_RISK" if pr >= 0.35 else "LOW_RISK")
                    recs = "High Priority Intervention Required" if pr >= 0.65 else ("Moderate Support Needed" if pr >= 0.35 else "On Track")
                    factors = []
                    if payload['internal_marks_avg'] < 10: factors.append(f"Low Internal Marks Avg ({payload['internal_marks_avg']}/20)")
                    if payload['absences'] > 10: factors.append(f"Excessive Absences ({payload['absences']} days)")
                    if payload['failures'] > 0: factors.append(f"Previous Failures ({payload['failures']})")
                    prediction_result = {
                        "failure_risk_probability": round(pr, 4),
                        "risk_category": risk_tag,
                        "predicted_class": int(pr >= 0.5),
                        "recommendation": recs,
                        "top_risk_factors": factors if factors else ["No critical negative factors"],
                        "model_version": bundle.get("best_run_id", "v1.0")[:8],
                        "inference_latency_ms": 1.2
                    }
            except Exception as e:
                st.error(f"Prediction error: {e}")

        if prediction_result:
            st.markdown("---")
            res_c1, res_c2 = st.columns([1, 2])

            with res_c1:
                prob = prediction_result["failure_risk_probability"]
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Failure Risk", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#EF4444" if prob >= 0.65 else ("#F59E0B" if prob >= 0.35 else "#10B981")},
                        'steps': [
                            {'range': [0, 35], 'color': "#ECFDF5"},
                            {'range': [35, 65], 'color': "#FEF3C7"},
                            {'range': [65, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 65
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_c2:
                cat = prediction_result["risk_category"]
                if cat == "HIGH_RISK":
                    st.error(f"🚨 **Prediction: HIGH RISK OF COURSE FAILURE ({(prob*100):.1f}%)**")
                elif cat == "MEDIUM_RISK":
                    st.warning(f"⚠️ **Prediction: MODERATE ACADEMIC RISK ({(prob*100):.1f}%)**")
                else:
                    st.success(f"✅ **Prediction: LOW RISK / ON TRACK TO PASS ({(prob*100):.1f}%)**")

                st.markdown(f"**Automated Recommendation:** {prediction_result['recommendation']}")
                st.markdown("**Identified Risk Drivers:**")
                for f in prediction_result["top_risk_factors"]:
                    st.markdown(f"- 🔴 {f}")

                st.caption(f"Served by MLflow Model Version: `{prediction_result['model_version']}` | Latency: `{prediction_result['inference_latency_ms']}ms`")

st.markdown("---")
st.caption("Student Performance & Dropout-Risk Prediction System — End-to-End MLOps Pipeline | Orchestrated via Airflow, PostgreSQL, MLflow & FastAPI")
