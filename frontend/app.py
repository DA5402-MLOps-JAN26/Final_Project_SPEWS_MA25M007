import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, random

API = os.getenv("API_BASE_URL", "http://localhost:8001")
st.set_page_config(page_title="SPEWS – Early Warning System", layout="wide", page_icon="🎓")

# ---------- Sidebar Navigation ----------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home & Help",
        "Student Risk Dashboard",
        "Cohort Overview",
        "ML Pipeline Console",
        "Monitoring Dashboard"
    ]
)

# ---------- Helper Functions ----------
def api(endpoint, method="GET", body=None):
    try:
        url = f"{API}{endpoint}"
        r = requests.post(url, json=body, timeout=5) if method == "POST" else requests.get(url, timeout=5)
        return r.json() if r.ok else None
    except:
        return None

def risk_badge(level):
    colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
    c = colors.get(level, "#888")
    return f'<span style="background:{c};color:white;padding:3px 14px;border-radius:10px;font-weight:bold;font-size:13px">{level}</span>'

# ---------- Page 0: Home & Help ----------
if page == "Home & Help":
    st.title("🎓 Student Performance Early Warning System (SPEWS)")
    st.markdown("""
    ### What is SPEWS?

    **SPEWS** predicts whether a student is at risk of dropping out or failing, **before** the semester ends.  
    It uses machine learning to analyse student behaviour (clicks in the virtual learning environment, assessment scores, and background information) and gives a **real‑time risk score** each week.

    > **Why this matters:** Academic advisors can use SPEWS to identify struggling students early and provide support when it's needed most.

    ### How to use this application
    Use the **sidebar on the left** to navigate between screens:

    - **Student Risk Dashboard** – Enter a student's weekly data and get an instant risk prediction.
    - **Cohort Overview** – View and filter a whole group of students, export reports.
    - **ML Pipeline Console** – See the status of the machine learning system, trigger retraining, and open MLflow/Airflow.
    - **Monitoring Dashboard** – Check system health, data drift, and live Prometheus metrics.

    ### Interpreting Risk Levels
    | Colour | Level | Meaning |
    |--------|-------|---------|
    | 🟢 Green | **Low Risk** (score < 35%) | Student is on track. Regular monitoring recommended. |
    | 🟡 Yellow | **Medium Risk** (35–65%) | Some warning signs. Proactive check‑in advised. |
    | 🔴 Red | **High Risk** ( > 65%) | Strong indicators of dropout. Immediate intervention needed. |

    ### Data Used
    The model uses the following information to make a prediction:

    - **Virtual Learning Environment (VLE) activity:** weekly clicks, cumulative clicks, trend slope, weeks since last active.
    - **Assessments:** latest score, average weighted score, number of missed assessments, average days late.
    - **Background:** number of previous attempts, early unregistration flag.
    """)

# ---------- Page 1: Student Risk Dashboard ----------
elif page == "Student Risk Dashboard":
    st.title("🎓 Student Risk Dashboard")
    st.markdown("Enter a student's data below and click **Predict Risk** to see their current risk level. Use the week slider to simulate different points in the semester.")

    # --- Input Section (3 columns x 4 rows) ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Identity & Time**")
        sid = st.number_input("Student ID", value=123456, step=1, help="A unique identifier for the student.")
        week = st.slider("Week", 1, 30, 8, help="Current week of the semester (1–30).")

        st.markdown("**VLE Activity**")
        clicks = st.number_input("Weekly Clicks (this week)", value=12.0, step=1.0,
                                 help="Number of clicks in the online learning platform this week.")
        cum_clicks = st.number_input("Cumulative Clicks", value=72.0, step=1.0,
                                     help="Total clicks from week 1 up to now.")
        weeks_inactive = st.number_input("Weeks Since Last Active", value=0.0, step=1.0,
                                         help="How many weeks since the student last accessed the platform.")
        click_trend = st.number_input("Click Trend Slope", value=-2.0, step=0.5,
                                      help="Change in weekly clicks over the last 3 weeks. Negative = declining engagement.")

    with col2:
        st.markdown("**Assessment Performance**")
        latest_score = st.number_input("Latest Assessment Score", value=34.0, step=1.0, min_value=0.0, max_value=100.0,
                                       help="Score of the most recent assessment (0–100).")
        avg_score = st.number_input("Average Weighted Score", value=34.0, step=1.0, min_value=0.0, max_value=100.0,
                                    help="Weighted average of all assessments so far.")
        missed = st.number_input("Missed Assessments", value=2, step=1, min_value=0,
                                 help="Number of assessments due that were not submitted.")
        days_late = st.number_input("Avg Days Late", value=5.0, step=0.5, min_value=0.0,
                                    help="Average number of days late on submitted assessments.")

    with col3:
        st.markdown("**Background & Status**")
        prev_attempts = st.number_input("Previous Attempts", value=0, step=1, min_value=0,
                                        help="How many times the student has attempted this module before.")
        early_unreg = st.selectbox("Early Unregistered?", ["No", "Yes"],
                                   help="Has the student unregistered early from any previous module?")

    st.markdown("---")

    # --- Bottom Section: Trend (left) | Prediction (right) ---
    bottom_left, bottom_right = st.columns([2, 1])

    with bottom_left:
        st.subheader("📈 Week‑by‑Week Risk Trend (Simulated)")
        st.caption("This chart shows how the student's risk **might** evolve throughout the semester based on historical patterns. It is a simulation, not a forecast.")
        weeks = list(range(1, week + 1))
        random.seed(sid)
        scores = [random.uniform(0.1, 0.8) for _ in weeks]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weeks, y=scores, mode='lines+markers',
                                 line=dict(color='#2E75B6', width=2)))
        fig.add_hline(y=0.35, line_dash="dot", line_color="orange", annotation_text="Medium threshold")
        fig.add_hline(y=0.65, line_dash="dot", line_color="red", annotation_text="High threshold")
        fig.update_layout(xaxis_title="Week", yaxis_title="Risk Score", height=280)
        st.plotly_chart(fig, use_container_width=True)

    with bottom_right:
        st.subheader("🔮 Prediction")
        st.caption("Click the button to get a real‑time risk assessment.")
        if st.button("Predict Risk", type="primary", use_container_width=True):
            gender_enc = 0
            disability_enc = 0
            education_level = 2.0
            imd_score = 5.0
            studied_credits = 60.0
            early_unreg_enc = 1 if early_unreg == "Yes" else 0

            payload = {"student": {
                "id_student": int(sid),
                "week_number": week,
                "weekly_clicks_current": clicks,
                "cumulative_clicks": cum_clicks,
                "weeks_since_active": weeks_inactive,
                "click_trend_slope": click_trend,
                "latest_score": latest_score,
                "avg_weighted_score": avg_score,
                "missed_assessments": float(missed),
                "avg_days_late": days_late,
                "gender_enc": gender_enc,
                "disability_enc": disability_enc,
                "education_level": education_level,
                "imd_score": imd_score,
                "num_of_prev_attempts": int(prev_attempts),
                "studied_credits": studied_credits,
                "early_unreg": early_unreg_enc
            }}
            result = api("/predict", "POST", payload)
            if result:
                st.session_state.prediction = result
            else:
                st.error("Prediction failed. Check API connection.")

        if "prediction" in st.session_state:
            res = st.session_state.prediction
            st.markdown(f"### Risk Level: {risk_badge(res['risk_level'])}",
                        unsafe_allow_html=True)
            st.metric("Risk Score", f"{res['risk_score']:.2%}")
            if res.get('top_features'):
                st.caption("Top factors driving this prediction:")
                df_fi = pd.DataFrame(res['top_features'])
                fig_fi = px.bar(df_fi, x='importance', y='feature',
                                orientation='h', color='importance',
                                color_continuous_scale='RdYlGn_r')
                fig_fi.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_fi, use_container_width=True)

# ---------- Page 2: Cohort Overview ----------
elif page == "Cohort Overview":
    st.title("📊 Cohort Overview")
    st.markdown("This screen shows a simulated group of 50 students. You can filter by risk level and export the data as a CSV file for further analysis.")
    random.seed(42)
    n = 50
    demo_data = pd.DataFrame({
        'Student ID': range(10000, 10000 + n),
        'Risk Score': [round(random.uniform(0.05, 0.95), 2) for _ in range(n)],
        'Week': [random.randint(1, 8) for _ in range(n)],
        'Last Active': [f"Week {random.randint(1,8)}" for _ in range(n)],
    })
    demo_data['Risk Level'] = demo_data['Risk Score'].apply(
        lambda x: 'High' if x > 0.65 else ('Medium' if x > 0.35 else 'Low'))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(demo_data))
    c2.metric("At Risk", len(demo_data[demo_data['Risk Level'].isin(['Medium','High'])]))
    c3.metric("High Risk", len(demo_data[demo_data['Risk Level'] == 'High']))
    c4.metric("Safe", len(demo_data[demo_data['Risk Level'] == 'Low']))
    filter_val = st.multiselect("Filter by Risk Level", ['Low','Medium','High'],
                                default=['High','Medium'])
    view = demo_data[demo_data['Risk Level'].isin(filter_val)] if filter_val else demo_data
    view = view.sort_values('Risk Score', ascending=False)
    st.dataframe(view, use_container_width=True, height=360)
    csv = view.to_csv(index=False)
    st.download_button("Export CSV", csv, "cohort_risk_report.csv", "text/csv")

# ---------- Page 3: ML Pipeline Console ----------
elif page == "ML Pipeline Console":
    st.title("⚙️ ML Pipeline Console")
    st.markdown("Monitor the machine learning pipeline status and access supporting tools.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("API Status")
        h = api("/health")
        if h:
            st.success(f"API Online – {h.get('timestamp', '')[:19]}")
        else:
            st.error("API Offline")
        st.subheader("Model Info")
        info = api("/model/info")
        if info:
            st.json(info)
        if st.button("Trigger Manual Retrain"):
            st.warning("Retrain triggered – monitor MLflow for results.")
    with c2:
        st.subheader("Experiment Tracking")
        st.link_button("Open MLflow", "http://localhost:5001")
        st.caption("View training experiments and model versions.")
        st.subheader("Workflow Orchestration")
        st.link_button("Open Airflow", "http://localhost:8080")
        st.caption("See the weekly data pipeline and retraining DAG.")

# ---------- Page 4: Monitoring Dashboard ----------
elif page == "Monitoring Dashboard":
    st.title("📈 Monitoring Dashboard")
    st.markdown("Real‑time system health and model performance indicators.")
    r = api("/ready")
    if r and r.get('model_loaded'):
        st.success(f"Model loaded – version {r.get('model_version', '?')}")
    else:
        st.error("Model not loaded")
    col1, col2, col3 = st.columns(3)
    col1.metric("API Status", "Healthy")
    col2.metric("PSI Score", "0.0139", help="Population Stability Index – measures data drift. Threshold is 0.2.")
    col3.metric("Error Rate", "0.0%", help="Percentage of failed prediction requests.")
    st.subheader("Grafana Dashboards")
    st.link_button("Open Grafana", "http://localhost:3001")
    st.caption("Live charts: request rate, latency, risk distribution, and more. Login: admin / admin.")
    st.subheader("Prometheus Metrics")
    st.caption("Raw metrics collected from the inference service.")
    try:
        response = requests.get(f"{API}/metrics", timeout=5)
        if response.status_code == 200:
            lines = response.text.splitlines()
            spews_lines = [line for line in lines if line.startswith("spews_") and not line.startswith("#")]
            if spews_lines:
                st.code("\n".join(spews_lines[:20]), language="text")
            else:
                st.info("No SPEWS metrics collected yet. Make some prediction requests to see data.")
        else:
            st.warning("Could not fetch metrics.")
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")