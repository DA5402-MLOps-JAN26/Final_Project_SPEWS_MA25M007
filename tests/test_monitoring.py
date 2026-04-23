import pytest
from monitoring.exporter import (
    update_drift_metrics,
    update_model_metrics,
    PSI_SCORE,
    MODEL_F1
)

def test_update_drift_metrics():
    update_drift_metrics({"sum_click": 0.15})
    # PSI_SCORE is a Gauge; we just verify it doesn't crash
    assert True

def test_update_model_metrics():
    update_model_metrics(0.865)
    assert True
