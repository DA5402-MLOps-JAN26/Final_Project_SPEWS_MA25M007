# High‑Level Design – Student Performance Early Warning System

## Technology Choices

| Component          | Choice                | Rationale                                                              |
|--------------------|-----------------------|------------------------------------------------------------------------|
| Data Pipeline      | Apache Airflow        | Scheduled weekly batch ingestion with retry logic                       |
| Versioning         | DVC + Git LFS         | Reproducibility – every model tied to specific data version             |
| Experiment Tracking| MLflow                | Unified experiment tracking + model registry in one tool                |
| Model Serving      | FastAPI               | Native Pydantic validation, async, auto OpenAPI docs                    |
| Monitoring         | Prometheus + Grafana  | Industry‑standard NRT monitoring + alerting                             |
| Containerization   | Docker Compose        | Independent frontend/backend services (rubric requirement)              |

## Architecture Layers

1. **Data Layer**: Airflow DAG ingests weekly, DVC tracks versions.
2. **ML Layer**: MLflow tracks experiments, best model promoted to registry.
3. **Serving Layer**: FastAPI loads Production model from MLflow registry.
4. **Presentation Layer**: Streamlit dashboard – REST API only (loose coupling).
5. **Monitoring Layer**: Prometheus scrapes `/metrics`, Grafana visualizes.

## Loose Coupling

Frontend and backend are completely independent Docker containers. They communicate ONLY via REST API calls – no shared Python imports.