# Test Plan – SPEWS

## Scope
This test plan covers unit tests for feature engineering functions and API contract tests for all FastAPI endpoints.

## Acceptance Criteria
- All unit tests pass.
- API endpoints return correct HTTP status codes and response schemas.
- Code coverage ≥ 80% for core modules (`data/features.py`, `api/predictor.py`, `api/main.py`).
- Model predictions vary sensibly with input changes (manual verification).
## Test Cases

| ID   | Type | Module | Test Description | Expected Result |
|------|------|--------|------------------|-----------------|
| TC01 | Unit | `data.features` | `compute_weekly_vle_features` returns DataFrame with required columns | Columns present |
| TC02 | Unit | `data.features` | Cumulative clicks never negative | All values ≥ 0 |
| TC03 | Unit | `data.features` | Demographic gender encoding is binary | Values ∈ {0,1} |
| TC04 | Unit | `data.features` | IMD score within 1-10 range | All values between 1 and 10 |
| TC05 | Unit | `data.features` | Early unregistration flag set correctly | Student 2 has `early_unreg=1` |
| TC06 | Unit | `data.features` | `compute_assessment_features` calculates missed assessments correctly | Correct counts |
| TC07 | Unit | `data.features` | `compute_assessment_features` calculates weighted average correctly | Correct weighted average |
| TC08 | API  | `api.main` | `GET /health` returns 200 and status ok | 200, `{"status":"ok"}` |
| TC09 | API  | `api.main` | `GET /ready` returns 200 and model_loaded true | 200, `model_loaded: true` |
| TC10 | API  | `api.main` | `GET /model/info` returns 200 and model metadata | 200, fields present |
| TC11 | API  | `api.main` | `POST /predict` with valid payload returns 200 and risk prediction | 200, risk_score ∈ [0,1] |
| TC12 | API  | `api.main` | `POST /predict/batch` returns 200 and batch summary | 200, total matches input count |
| TC13 | API  | `api.main` | `GET /metrics` returns 200 and Prometheus text | 200, content contains metric names |
| TC14 | API  | `api.main` | `GET /metrics` returns SPEWS‑specific metrics | Contains `spews_api_requests_total` |
| TC15 | API  | `api.main` | Lifespan metric seeding does not crash | API starts successfully, health check passes |
| TC16 | Unit | `api.predictor` | Fallback model creation | `model_version == "fallback"` |
| TC17 | Unit | `api.predictor` | Prediction with valid features returns correct risk score | Score matches expected value |
| TC18 | Unit | `api.predictor` | Risk level thresholds are correct | Low/Medium/High boundaries respected |
| TC19 | Unit | `api.predictor` | Top features fallback with no model has equal importances | `importance == 1.0` for all |
| TC20 | Unit | `api.predictor` | Batch prediction returns correct number of results | `len(results) == len(input)` |
| TC21 | Unit | `monitoring.exporter` | Drift metrics update without error | No exception raised |
| TC22 | Unit | `monitoring.exporter` | Model F1 metric update sets gauge | No exception raised |

## Test Execution
Tests are run using `pytest` with coverage measurement.
