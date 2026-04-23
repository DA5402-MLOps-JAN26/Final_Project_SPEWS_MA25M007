import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, PropertyMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session", autouse=True)
def mock_predictor_basics():
    with patch('api.predictor.Predictor.load_model'), \
         patch('api.predictor.Predictor.is_loaded', new_callable=PropertyMock) as mock_loaded:
        mock_loaded.return_value = True
        yield

from api.main import app

client = TestClient(app)

SAMPLE_STUDENT = {
    "id_student": 1, "week_number": 4, "weekly_clicks_current": 10.0,
    "cumulative_clicks": 40.0, "weeks_since_active": 0.0, "click_trend_slope": 0.0,
    "latest_score": 70.0, "avg_weighted_score": 70.0, "missed_assessments": 1.0,
    "avg_days_late": 2.0, "gender_enc": 0, "disability_enc": 0,
    "education_level": 2.0, "imd_score": 5.0, "num_of_prev_attempts": 0,
    "studied_credits": 60.0, "early_unreg": 0
}

MOCK_PREDICTION = {
    "risk_score": 0.45,
    "risk_level": "Medium",
    "top_features": [{"feature": "latest_score", "importance": 0.2, "value": 70.0}],
    "model_version": "test"
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ready():
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "model_name" in data

@patch('api.main.predictor.predict_single', return_value=MOCK_PREDICTION)
def test_predict(mock_predict):
    r = client.post("/predict", json={"student": SAMPLE_STUDENT})
    assert r.status_code == 200
    data = r.json()
    assert "risk_score" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]

@patch('api.main.predictor.predict_single', return_value=MOCK_PREDICTION)
def test_predict_batch(mock_predict):
    r = client.post("/predict/batch", json={"students": [SAMPLE_STUDENT, SAMPLE_STUDENT]})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert "predictions" in data

def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200

def test_metrics_content():
    r = client.get("/metrics")
    assert "spews_api_requests_total" in r.text or "python_info" in r.text
