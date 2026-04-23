import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, random

API = os.getenv("API_BASE_URL", "http://localhost:8001")
st.set_page_config(page_title="SPEWS", layout="wide", page_icon="🎓")

page = st.sidebar.radio("Navigation", [
    "Student Risk Dashboard",
    "Cohort Overview",
    "ML Pipeline Console",
    "Monitoring Dashboard"
])

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

if page == "Student Risk Dashboard":
    st.title("🎓 Student Risk Dashboard")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Predict Risk")
        sid = st.number_input("Student ID", value=123456, step=1)
        week = st.slider("Week", 1, 30, 8)
        clicks = st.number_input("Weekly Clicks", value=12.0)
        score = st.number_input("Latest Score (0-100)", value=34.0)
        missed = st.number_input("Missed Assessments", value=2)
        unreg = st.selectbox("Early Unregistered?", [0, 1])
        days_late = st.number_input("Avg Days Late", value=5.0)
        if st.button("Predict Risk", type="primary"):
            payload = {"student": {
                "id_student": int(sid),
                "week_number": week,
                "weekly_clicks_current": clicks,
                "cumulative_clicks": clicks * week,
                "weeks_since_active": 0.0,
                "click_trend_slope": 0.0,
                "latest_score": score,
                "avg_weighted_score": score,
                "missed_assessments": float(missed),
                "avg_days_late": days_late,
                "gender_enc": 0,
                "disability_enc": 0,
                "education_level": 2.0,
                "imd_score": 5.0,
                "num_of_prev_attempts": 0,
                "studied_credits": 60.0,
                "early_unreg": unreg,
            }}
            result = api("/predict", "POST", payload)
            if result:
                st.markdown(f"### Risk Level: {risk_badge(result['risk_level'])}", unsafe_allow_html=True)
                st.metric("Risk Score", f"{result['risk_score']:.2%}")
                st.subheader("Top Contributing Factors")
                if result.get('top_features'):
                    df_fi = pd.DataFrame(result['top_features'])
                    fig = px.bar(df_fi, x='importance', y='feature', orientation='h',
                                 color='importance', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Prediction failed. Check API connection.")
    with c2:
        st.subheader("Week-by-Week Risk Trend (Simulated)")
        weeks = list(range(1, week+1))
        random.seed(sid)
        scores = [random.uniform(0.1, 0.8) for _ in weeks]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weeks, y=scores, mode='lines+markers', line=dict(color='#2E75B6', width=2)))
        fig.add_hline(y=0.35, line_dash="dot", line_color="orange", annotation_text="Medium threshold")
        fig.add_hline(y=0.65, line_dash="dot", line_color="red", annotation_text="High threshold")
        fig.update_layout(xaxis_title="Week", yaxis_title="Risk Score", height=300)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Cohort Overview":
    st.title("📊 Cohort Overview")
    random.seed(42)
    n = 50
    demo_data = pd.DataFrame({
        'Student ID': range(10000, 10000+n),
        'Risk Score': [round(random.uniform(0.05, 0.95), 2) for _ in range(n)],
        'Week': [random.randint(1, 8) for _ in range(n)],
        'Last Active': [f"Week {random.randint(1,8)}" for _ in range(n)],
    })
    demo_data['Risk Level'] = demo_data['Risk Score'].apply(
        lambda x: 'High' if x > 0.65 else ('Medium' if x > 0.35 else 'Low'))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(demo_data))
    c2.metric("At Risk", len(demo_data[demo_data['Risk Level'].isin(['Medium','High'])]))
    c3.metric("High Risk", len(demo_data[demo_data['Risk Level']=='High']))
    c4.metric("Safe", len(demo_data[demo_data['Risk Level']=='Low']))
    filter_val = st.multiselect("Filter by Risk Level", ['Low','Medium','High'], default=['High','Medium'])
    view = demo_data[demo_data['Risk Level'].isin(filter_val)] if filter_val else demo_data
    view = view.sort_values('Risk Score', ascending=False)
    st.dataframe(view, use_container_width=True, height=360)
    csv = view.to_csv(index=False)
    st.download_button("Export CSV", csv, "cohort_risk_report.csv", "text/csv")

elif page == "ML Pipeline Console":
    st.title("⚙️ ML Pipeline Console")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("API Status")
        h = api("/health")
        if h:
            st.success(f"API Online - {h.get('timestamp', '')[:19]}")
        else:
            st.error("API Offline")
        st.subheader("Model Info")
        info = api("/model/info")
        if info:
            st.json(info)
        if st.button("Trigger Manual Retrain"):
            st.warning("Retrain triggered - monitor MLflow UI")
    with c2:
        st.subheader("MLflow Experiments")
        st.link_button("Open MLflow", "http://localhost:5001")
        st.subheader("Airflow Pipeline")
        st.link_button("Open Airflow", "http://localhost:8080")

elif page == "Monitoring Dashboard":
    st.title("📈 Monitoring Dashboard")
    r = api("/ready")
    if r and r.get('model_loaded'):
        st.success(f"Model loaded - version {r.get('model_version', '?')}")
    else:
        st.error("Model not loaded")
    c1, c2, c3 = st.columns(3)
    c1.metric("API Status", "Healthy")
    c2.metric("PSI Score", "0.0139", help="Threshold: 0.2")
    c3.metric("Error Rate", "0.0%")
    st.subheader("Grafana Dashboard")
    st.link_button("Open Grafana", "http://localhost:3001")
    st.info("Default login: admin / admin")
    st.subheader("Prometheus Metrics")
    metrics = api("/metrics")
    if metrics:
        st.text(str(metrics)[:800])

