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

- ✅ **Automated weekly pipeline** – Airflow DAG ingests new data, validates schema, detects drift, and triggers retraining.
- ✅ **Experiment tracking** – MLflow logs all hyperparameters, metrics, and artifacts. The best model is registered and aliased `@production`.
- ✅ **Real‑time inference** – FastAPI serves predictions via REST endpoints with health/readiness probes.
- ✅ **Interactive dashboard** – Streamlit provides five screens for individual risk assessment, cohort overview, pipeline console, and monitoring.
- ✅ **Comprehensive monitoring** – Prometheus scrapes metrics (request count, latency, risk distribution, PSI score). Grafana displays live dashboards.
- ✅ **Reproducible environments** – Conda environment and Docker images guarantee consistent behaviour across development and deployment.
- ✅ **Extensive testing** – 22 unit tests with 88% code coverage.

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
├── MLproject # MLflow project definition
├── README.md # This file
├── conda.yaml # Conda environment specification
├── docker-compose.yml # Multi‑service Docker orchestration
├── dvc.lock # DVC pipeline lock file
├── dvc.yaml # DVC pipeline stages
├── params.yaml # DVC parameters
├── requirements.txt # Python dependencies
├── Dockerfile # Backend API Docker image
├── api/ # FastAPI application
│ ├── init.py
│ ├── main.py # Endpoint definitions
│ ├── predictor.py # Model loading & inference
│ └── schemas.py # Pydantic models
├── dags/ # Airflow DAGs
│ └── student_pipeline.py # Weekly drift/retrain pipeline
├── data/ # Data and feature engineering
│ ├── features.py # Feature engineering functions
│ └── features/ # Feature matrices (tracked by DVC)
├── docs/ # Documentation
│ ├── HLD.md # High‑Level Design
│ ├── LLD.md # Low‑Level Design
│ ├── test_plan.md # Test plan
│ ├── test_report.md # Test execution report
│ └── user_manual.md # End‑user guide
├── frontend/ # Streamlit UI
│ ├── Dockerfile
│ └── app.py # Main Streamlit application
├── models/ # Model training & retraining
│ ├── production_model.pkl # Bundled production model (fallback)
│ ├── retrain.py # Incremental retraining script
│ └── trainer.py # Model training class
├── monitoring/ # Prometheus & Grafana
│ ├── exporter.py # Prometheus metric definitions
│ ├── grafana_dashboard.json # Pre‑configured dashboard
│ └── prometheus.yml # Prometheus scrape config
├── notebooks/ # EDA scripts
│ ├── eda_part1.py
│ └── eda_part2.py
├── scripts/ # Utility scripts
│ ├── build_features.py
│ ├── final_check.py
│ ├── register_best_model.py
│ ├── retrain_robust.py
│ └── run_training.py
└── tests/ # Unit tests
├── test_api.py
├── test_features.py
├── test_monitoring.py
└── test_predictor.py

text

---

## Getting Started

### Prerequisites

- **Conda** (or Python 3.10 with virtualenv)
- **Docker** and **Docker Compose**
- **Git LFS**
- **Kaggle API credentials** (for dataset download)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/DA5402-MLOps-JAN26/Final_Project_SPEWS_MA25M007.git
   cd Final_Project_SPEWS_MA25M007
Create and activate the Conda environment

bash
conda create -n <name> python=3.10 -y
conda activate <name>
pip install -r requirements.txt
Download the OULAD dataset

bash
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/
cd data/raw
kaggle datasets download -d anlgrbz/student-demographics-online-education-dataoulad --unzip
cd ../..
Pull DVC‑tracked files

bash
dvc pull
Start MLflow UI

bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5000
Start FastAPI

bash
uvicorn api.main:app --reload --port 8001
Start Streamlit

bash
export API_BASE_URL=http://localhost:8001
cd frontend && streamlit run app.py --server.port 8501
Docker Deployment
The entire stack can be started with a single command:

bash
docker-compose up --build -d
After all services are healthy, access:

Streamlit UI: http://localhost:3002

FastAPI: http://localhost:8002/health

MLflow: http://localhost:5001

Airflow: http://localhost:8080 (admin/admin)

Grafana: http://localhost:3001 (admin/admin)

Prometheus: http://localhost:9090

Usage
Streamlit Dashboard
Screen                     | Description
--------------------------|-------------
Home & Help               | Introduction to SPEWS, risk level legend, and data explanation
Student Risk Dashboard    | Enter student metrics (clicks, scores, missed assessments) and get a real‑time risk prediction.
Cohort Overview           | View and filter a simulated cohort. Export risk reports as CSV.
ML Pipeline Console       | Monitor API status, view current model version, and trigger manual retraining.
Monitoring Dashboard      | Live system metrics (error rate, PSI score) and links to Grafana.

### API Endpoints

| Method | Endpoint           | Description                                      |
|--------|--------------------|--------------------------------------------------|
| GET    | `/health`          | Health check                                     |
| GET    | `/ready`           | Readiness probe (model loaded status)            |
| GET    | `/model/info`      | Metadata about the production model              |
| POST   | `/predict`         | Single student risk prediction                   |
| POST   | `/predict/batch`   | Batch prediction for multiple students           |
| GET    | `/metrics`         | Prometheus metrics                               |

Example prediction request:

```bash
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
