# Low‑Level Design – SPEWS API

## Base URL
- Local development: `http://localhost:8001`
- Docker deployment: `http://localhost:8002`

## Endpoints

### GET `/health`
**Description:** Health check endpoint for orchestration.  
**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-04-21T12:00:00.000000"
}
```

### GET `/ready`
**Description:** Readiness probe indicating model loaded status.  
**Response:**
```json
{
"status": "ready",
"model_version": "12",
"model_loaded": true
}
```


### GET `/model/info`
**Description:** Returns metadata about the currently loaded production model.  
**Response:**
```json
{
"model_name": "student_dropout_best",
"version": "12",
"alias": "production",
"run_id": "39ce0acb6903461b8965567377cbd6de",
"f1_score": null
}
```

### POST `/predict`
**Description:** Predict dropout risk for a single student.  
**Request Body:**
```json
{
"student": {
"id_student": 12345,
"week_number": 6,
"weekly_clicks_current": 12.0,
"cumulative_clicks": 72.0,
"weeks_since_active": 0.0,
"click_trend_slope": 0.0,
"latest_score": 34.0,
"avg_weighted_score": 34.0,
"missed_assessments": 2.0,
"avg_days_late": 5.0,
"gender_enc": 0,
"disability_enc": 0,
"education_level": 2.0,
"imd_score": 5.0,
"num_of_prev_attempts": 0,
"studied_credits": 60.0,
"early_unreg": 0
}
}
```

**Response:**
```json
{
"id_student": 12345,
"week_number": 6,
"risk_score": 0.9209,
"risk_level": "High",
"top_features": [
{"feature": "missed_assessments", "importance": 0.1855, "value": 2.0},
{"feature": "num_of_prev_attempts", "importance": 0.1959, "value": 0.0},
{"feature": "latest_score", "importance": 0.1398, "value": 34.0}
],
"model_version": "12",
"prediction_id": "a9b3b579-d73a-4ae9-aacd-2636a786776b"
}

```

### POST `/predict/batch`
**Description:** Predict dropout risk for multiple students.  
**Request Body:**
```json
{
"students": [ ... ]
}
```
**Response:**
```json
{
"predictions": [ ... ],
"total": 2,
"high_risk_count": 1,
"model_version": "12"
}
```

### GET `/metrics`
**Description:** Exposes Prometheus metrics in text format.  
**Response:** Prometheus exposition format.

## Loose Coupling
The frontend (Streamlit) communicates with the backend (FastAPI) exclusively via these REST endpoints. The API base URL is injected via the `API_BASE_URL` environment variable. No Python modules are shared between the two containers.