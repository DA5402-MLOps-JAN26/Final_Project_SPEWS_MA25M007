# SPEWS User Manual

## Introduction
The Student Performance Early Warning System (SPEWS) predicts student dropout risk on a weekly basis. This tool helps academic advisors identify at‑risk students early and provide timely interventions.

## Accessing the Application
Open your web browser and go to:  
- **Local Development:** `http://localhost:8501`  
- **Docker Deployment:** `http://localhost:3002`

## Navigating the Dashboard
Use the sidebar on the left to switch between five screens.

---

### 1. Home & Help
**Purpose:** Understand the SPEWS system and how to use it.

**What you see:**
- A brief description of the project and its goals.
- A risk‑level legend explaining the meaning of **Low** (green), **Medium** (amber), and **High** (red) risk scores.
- A summary of the data the model uses for predictions.
- Links to other resources.

---

### 2. Student Risk Dashboard
**Purpose:** Get a real‑time risk prediction for an individual student.

**How to use:**
1. Enter the **Student ID** (any number is fine for demo).
2. Adjust the **Week** slider (1–30) to the current semester week.
3. Fill in the student's activity data:
   - **Weekly Clicks:** Number of VLE clicks in the current week.
   - **Latest Score:** Most recent assessment score (0–100).
   - **Missed Assessments:** How many assessments the student has not submitted.
   - **Early Unregistered?** Select `1` if the student has already unregistered early, otherwise `0`.
   - **Avg Days Late:** Average days late on submitted assessments.
4. Click the **Predict Risk** button.

**Understanding the results:**
- **Risk Badge:**  
  - **Green (Low):** Risk score < 35% – student is likely on track.  
  - **Amber (Medium):** Risk score 35%–65% – proactive check‑in recommended.  
  - **Red (High):** Risk score > 65% – immediate intervention needed.
- **Risk Score:** The model's confidence (0%–100%).
- **Top Contributing Factors:** The three features most influencing this prediction.
- **Trend Chart:** A simulated week‑by‑week risk trajectory (for illustration).

---

### 3. Cohort Overview
**Purpose:** View and filter a list of students with their predicted risk levels.

**How to use:**
- The table shows a simulated cohort of 50 students with their risk scores and levels.
- Use the **Filter by Risk Level** dropdown to show only High, Medium, or Low risk students.
- Click **Export CSV** to download the filtered list as a spreadsheet for further analysis.

---

### 4. ML Pipeline Console
**Purpose:** Monitor the health of the ML pipeline and access supporting tools.

**What you see:**
- **API Status:** Shows whether the prediction service is online.
- **Model Info:** Displays the current production model version.
- **Trigger Manual Retrain:** (Simulated) Initiates a model retraining cycle.
- **Open MLflow:** Opens the MLflow UI where you can view past experiments and model versions.
- **Open Airflow:** Opens the Airflow UI to see the weekly data pipeline status.

---

### 5. Monitoring Dashboard
**Purpose:** View real‑time system metrics and alerts.

**What you see:**
- **Model Loaded:** Confirms the prediction model is active.
- **API Status / PSI Score / Error Rate:** Key health indicators.
  - **PSI Score:** Measures data drift. Alert threshold is 0.2.
  - **Error Rate:** Should remain near 0%.
- **Open Grafana:** Launches a detailed monitoring dashboard with live charts.
- **Prometheus Metrics:** Raw metrics for advanced users.

## Troubleshooting
| Issue | Solution |
|-------|----------|
| "Model not loaded" message | Ensure the FastAPI service and MLflow are running. |
| Prediction button does nothing | Check that you have filled all input fields. Refresh the page if needed. |
| Dashboard looks empty | Wait a few seconds for data to load, or try a different time range in the top‑right corner. |
| Cannot connect to MLflow/Airflow | Make sure Docker services are up (`docker-compose ps`). |

---
