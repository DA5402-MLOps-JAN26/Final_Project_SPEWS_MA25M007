Final_Project_SPEWS_MA25M007
Student Performance Early Warning System (SPEWS)
https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007  
https://www.python.org/  
https://www.docker.com/  
https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007/actions  
https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007

A complete end‑to‑end MLOps pipeline that predicts student dropout risk weekly using the Open University Learning Analytics Dataset (OULAD).

📋 Table of Contents
Overview

Key Features

Technology Stack

Project Structure

Getting Started

Prerequisites

Local Development Setup

Docker Deployment

Usage

Streamlit Dashboard

API Endpoints

Airflow Pipeline

Monitoring

Testing

Documentation

Contributors

License

Overview
The Student Performance Early Warning System (SPEWS) identifies at‑risk students early in the semester so that academic advisors can intervene before dropout occurs. The system ingests demographic, assessment, and clickstream data, trains a machine learning model (XGBoost), and serves real‑time predictions via a REST API and an interactive web dashboard.

Key Features
Automated weekly pipeline with Airflow

✅ MLflow experiment tracking & model registry

✅ FastAPI real‑time inference

✅ Streamlit interactive dashboard

✅ Prometheus & Grafana monitoring

✅ Reproducible environments with Conda & Docker

✅ 22 unit tests with 88% coverage

Technology Stack
Category	Tools
Data Versioning	DVC, Git LFS
Workflow	Apache Airflow 2.8
Experiment Tracking	MLflow 3.11
Model Training	XGBoost, scikit‑learn
Model Serving	FastAPI, Uvicorn
Frontend	Streamlit, Plotly
Monitoring	Prometheus, Grafana, Pushgateway
Containerisation	Docker, Docker Compose
Testing	Pytest, pytest‑cov
Languages	Python 3.10, Bash


Project Structure
plaintext
Final_Project_SPEWS_MA25M007/
├── MLproject
├── README.md
├── conda.yaml
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml / dvc.lock
├── params.yaml
│
├── api/
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   └── __init__.py
│
├── dags/
│   └── student_pipeline.py
│
├── data/
│   ├── features.py
│   └── features/
│
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   ├── test_plan.md
│   ├── test_report.md
│   └── user_manual.md
│
├── frontend/
│   ├── app.py
│   └── Dockerfile
│
├── models/
│   ├── trainer.py
│   ├── retrain.py
│   └── production_model.pkl
│
├── monitoring/
│   ├── exporter.py
│   ├── prometheus.yml
│   └── grafana_dashboard.json
│
├── notebooks/
│   ├── eda_part1.py
│   └── eda_part2.py
│
├── scripts/
│   ├── build_features.py
│   ├── run_training.py
│   ├── register_best_model.py
│   ├── retrain_robust.py
│   └── final_check.py
│
└── tests/
    ├── test_api.py
    ├── test_features.py
    ├── test_monitoring.py
    └── test_predictor.py
Getting Started
Prerequisites
Conda (or Python 3.10 with virtualenv)

Docker & Docker Compose

Git LFS

Kaggle API credentials

Local Development Setup
bash
# Clone the repository
git clone https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007.git
cd Final_Project_SPEWS_MA25M007

# Create and activate Conda environment
conda create -n spews python=3.10 -y
conda activate spews
pip install -r requirements.txt

# Download OULAD dataset
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/
cd data/raw
kaggle datasets download -d anlgrbz/student-demographics-online-education-dataoulad --unzip
cd ../..

# Pull DVC-tracked files
dvc pull

# Start MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5000

# Start FastAPI
uvicorn api.main:app --reload --port 8001

# Start Streamlit
export API_BASE_URL=http://localhost:8001
cd frontend && streamlit run app.py --server.port 8501
Docker Deployment
bash
docker-compose up --build -d
Usage
Streamlit Dashboard
Screen	Description
Home & Help	Intro, risk legend, guidance on interpreting predictions.
Student Risk Dashboard	Input student metrics → real‑time risk prediction.
Cohort Overview	Simulated cohort view, filters, CSV export.
ML Pipeline Console	API status, model version, manual retraining.
Monitoring Dashboard	Live metrics (latency, PSI drift, error rate) with Grafana panels.


API Endpoints
Method	Endpoint	Description
GET	/health	Health check
GET	/ready	Readiness probe (model loaded)
GET	/model/info	Metadata about production model
POST	/predict	Single student risk prediction
POST	/predict/batch	Batch predictions
GET	/metrics	Prometheus metrics


Example request:

bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
        "student": {
          "id_student": 12345,
          "week_number": 6,
          "clicks": 120,
          "score": 65,
          "missed_assessments": 1
        }
      }'
Airflow Pipeline
Weekly DAG student_weekly_pipeline

Steps: ingest VLE data → validate → clean outliers → compute PSI drift → retrain if PSI > 0.2 → log metrics to Pushgateway

Manual trigger config:

json
{"week_num": 8}
Monitoring
Prometheus scrapes /metrics and Pushgateway

Grafana dashboard shows:

API request rate & latency

Prediction risk distribution

Model F1 score

Pipeline success status

Data drift PSI score

API error rate

Testing
bash
pytest tests/ -v --cov=. --cov-report=term-missing
Results: 22 tests passed, 88% coverage

Documentation
Document	Description
docs/HLD.md	High‑Level

