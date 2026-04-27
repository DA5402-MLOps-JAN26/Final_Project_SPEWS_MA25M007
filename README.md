# Final_Project_SPEWS_MA25M007
# Student Performance Early Warning System (SPEWS)

[![MLOps](https://img.shields.io/badge/MLOps-Complete-blue)](https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-22%20passed-success)](https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007/actions)
[![Coverage](https://img.shields.io/badge/Coverage-88%25-brightgreen)](https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007)

A complete end-to-end MLOps pipeline that predicts student dropout risk weekly using the Open University Learning Analytics Dataset (OULAD).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
  - [Docker Deployment](#docker-deployment)
- [Usage](#usage)
  - [Streamlit Dashboard](#streamlit-dashboard)
  - [API Endpoints](#api-endpoints)
  - [Airflow Pipeline](#airflow-pipeline)
  - [Monitoring](#monitoring)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

The **Student Performance Early Warning System (SPEWS)** identifies at‑risk students early in the semester so that academic advisors can intervene before dropout occurs. The system ingests demographic, assessment, and clickstream data, trains a machine learning model (XGBoost), and serves real‑time predictions via a REST API and an interactive web dashboard.

The entire lifecycle is automated and reproducible:
- **Data versioning** with DVC
- **Workflow orchestration** with Apache Airflow (drift detection & retraining)
- **Experiment tracking & model registry** with MLflow
- **Model serving** with FastAPI
- **Monitoring** with Prometheus and Grafana
- **Containerisation** with Docker Compose

This project was developed as part of the **DA5402 – Machine Learning Operations Lab** course and follows industry‑standard MLOps practices.

---

## Key Features

-  **Automated weekly pipeline** – Airflow DAG ingests new data, validates schema, detects drift, and triggers retraining.
- **Experiment tracking** – MLflow logs all hyperparameters, metrics, and artifacts. The best model is registered and aliased `@production`.
- **Real‑time inference** – FastAPI serves predictions via REST endpoints with health/readiness probes.
- **Interactive dashboard** – Streamlit provides five screens for individual risk assessment, cohort overview, pipeline console, and monitoring.
- **Comprehensive monitoring** – Prometheus scrapes metrics (request count, latency, risk distribution, PSI score). Grafana displays live dashboards.
- **Reproducible environments** – Conda environment and Docker images guarantee consistent behaviour across development and deployment.
- **Extensive testing** – 22 unit tests with 88% code coverage.

---

## Technology Stack

| Category              | Tools                                                                 |
|-----------------------|-----------------------------------------------------------------------|
| **Data Versioning**   | DVC, Git LFS                                                          |
| **Workflow**          | Apache Airflow 2.8                                                    |
| **Experiment Tracking**| MLflow 3.11                                                           |
| **Model Training**    | XGBoost, scikit‑learn                                                 |
| **Model Serving**     | FastAPI, Uvicorn                                                      |
| **Frontend**          | Streamlit, Plotly                                                     |
| **Monitoring**        | Prometheus, Grafana, Pushgateway                                      |
| **Containerisation**  | Docker, Docker Compose                                                |
| **Testing**           | Pytest, pytest‑cov                                                    |
| **Languages**         | Python 3.10, Bash                                                     |

---

## Project Structure

Final_Project_SPEWS_MA25M007/
├── MLproject              # MLflow project definition
├── README.md              # Project documentation
├── conda.yaml             # Conda environment specification
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Multi-service orchestration
├── Dockerfile             # Backend API image
├── dvc.yaml / dvc.lock    # DVC pipeline stages & lock file
├── params.yaml            # DVC parameters
│
├── api/                   # FastAPI application
│   ├── main.py            # Endpoint definitions
│   ├── predictor.py       # Model loading & inference
│   ├── schemas.py         # Pydantic models
│   └── __init__.py
│
├── dags/                  # Airflow DAGs
│   └── student_pipeline.py
│
├── data/                  # Data & feature engineering
│   ├── features.py
│   └── features/          # DVC-tracked feature matrices
│
├── docs/                  # Documentation
│   ├── HLD.md             # High-Level Design
│   ├── LLD.md             # Low-Level Design
│   ├── test_plan.md
│   ├── test_report.md
│   └── user_manual.md
│
├── frontend/              # Streamlit UI
│   ├── app.py
│   └── Dockerfile
│
├── models/                # Model training & retraining
│   ├── trainer.py
│   ├── retrain.py
│   └── production_model.pkl
│
├── monitoring/            # Prometheus & Grafana
│   ├── exporter.py
│   ├── prometheus.yml
│   └── grafana_dashboard.json
│
├── notebooks/             # Exploratory Data Analysis
│   ├── eda_part1.py
│   └── eda_part2.py
│
├── scripts/               # Utility scripts
│   ├── build_features.py
│   ├── run_training.py
│   ├── register_best_model.py
│   ├── retrain_robust.py
│   └── final_check.py
│
└── tests/                 # Unit tests
    ├── test_api.py
    ├── test_features.py
    ├── test_monitoring.py
    └── test_predictor.py

---

## Getting Started

### Prerequisites

- **Conda** (or Python 3.10 with virtualenv)
- **Docker** and **Docker Compose**
- **Git LFS**
- **Kaggle API credentials** (for dataset download)

### Local Development Setup

# Clone the repository
git clone https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007.git
cd Final_Project_SPEWS_MA25M007

# Create and activate Conda environment
conda create -n spews python=3.10 -y
conda activate spews
pip install -r requirements.txt

# Download the OULAD dataset
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/
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
The entire stack can be started with a single command:

bash
docker-compose up --build -d

After all services are healthy.
Access services:

Streamlit UI → http://localhost:3002

FastAPI → http://localhost:8002/health

MLflow → http://localhost:5001

Airflow → http://localhost:8080 (admin/admin)

Grafana → http://localhost:3001 (admin/admin)

Prometheus → http://localhost:9090

Usage

Streamlit Dashboard

| Screen | Description |
| --- | --- |
| **Home & Help** | Provides an introduction to SPEWS, explains the purpose of the system, shows the risk level legend (low, medium, high), and offers guidance on how to interpret predictions. |
| **Student Risk Dashboard** | Interactive form where advisors can input student metrics (weekly clicks, assessment scores, missed submissions). The system returns a real‑time dropout risk prediction with confidence scores. |
| **Cohort Overview** | Displays a simulated cohort of students with sortable and filterable risk levels. Advisors can drill down into individual profiles and export cohort risk reports as CSV for offline analysis. |
| **ML Pipeline Console** | Shows the current status of the API and ML pipeline. Displays the active model version, retraining history, and allows manual retraining triggers directly from the UI. |
| **Monitoring Dashboard** | Provides live system metrics such as API request rate, latency, error rate, and PSI drift score. Includes embedded Grafana panels for deeper monitoring insights. |

API Endpoints
| Method | Endpoint | Description |
| --- | --- | --- |
| **GET** | ``/health`` | Basic health check to confirm the API service is running. |
| **GET** | ``/ready`` | Readiness probe that verifies if the production model is loaded and ready. |
| **GET** | ``/model/info`` | Returns metadata about the current production model (version, training date, metrics). |
| **POST** | ``/predict`` | Accepts a single student’s data in JSON format and returns dropout risk prediction. |
| **POST** | ``/predict/batch`` | Accepts multiple student records in JSON format and returns batch predictions. |
| **GET** | ``/metrics`` | Exposes Prometheus metrics (request count, latency, risk distribution, PSI score) for monitoring. |

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
The DAG student_weekly_pipeline runs weekly. It:

Ingests and validates VLE data.

Cleans outliers.

Computes PSI drift score against baseline statistics.

Branches to retraining if PSI > 0.2.

Logs metrics to Pushgateway.

Manually trigger with config:

json
{"week_num": 8}
Monitoring
Prometheus scrapes /metrics from the API and Pushgateway.

Grafana dashboard (SPEWS Monitoring) displays:

API request rate & latency percentiles

Prediction risk distribution

Model F1 score

Pipeline success status

Data drift PSI score

API error rate

Testing
Run the test suite with coverage:

bash
pytest tests/ -v --cov=. --cov-report=term-missing
Current results: 22 tests passed, 88% overall coverage.

Documentation
Document	Description
docs/HLD.md	High‑Level Design – architecture, technology choices
docs/LLD.md	Low‑Level Design – API endpoint specifications and I/O schemas
docs/test_plan.md	Test strategy, cases, and acceptance criteria
docs/test_report.md	Test execution summary and coverage report
docs/user_manual.md	Step‑by‑step guide for non‑technical users
Contributors
Aswini VJ (MA25M007) – DA5402 MLOps Lab, IIT Madras

License
This project is submitted as part of academic coursework and is not intended for production use without further security and scalability hardening.

Git Tags: v0.1-foundation → v0.2-ml-pipeline → v0.3-serving → v1.0-submission
Final Submission: April 2026

text
