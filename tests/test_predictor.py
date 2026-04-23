import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictor import Predictor, FEATURE_COLS

def test_fallback_model_creation():
    p = Predictor()
    p._load_fallback_model()
    assert p.model is not None
    assert p.model_version == "fallback"

def test_predict_single_with_valid_features():
    p = Predictor()
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])
    p.model = DummyModel()
    p.model_version = "dummy"
    features = {col: 0.0 for col in FEATURE_COLS}
    result = p.predict_single(features)
    assert result['risk_score'] == 0.7
    assert result['risk_level'] == "High"
    assert result['model_version'] == "dummy"
    assert 'top_features' in result

def test_predict_single_risk_level_thresholds():
    p = Predictor()
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[1 - prob, prob]])
    p.model = DummyModel()
    with patch.object(p.model, 'predict_proba', return_value=np.array([[0.8, 0.2]])):
        result = p.predict_single({col: 0.0 for col in FEATURE_COLS})
        assert result['risk_level'] == "Low"
    with patch.object(p.model, 'predict_proba', return_value=np.array([[0.5, 0.5]])):
        result = p.predict_single({col: 0.0 for col in FEATURE_COLS})
        assert result['risk_level'] == "Medium"

def test_top_features_fallback_importance():
    p = Predictor()
    class NoImportanceModel:
        pass
    p.model = NoImportanceModel()
    X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLS}])
    top = p._top_features(X)
    assert len(top) == 3
    for item in top:
        assert item['importance'] == 1.0

def test_predict_batch():
    p = Predictor()
    p.model = MagicMock()
    p.model.predict_proba.return_value = np.array([[0.4, 0.6]])
    p.model_version = "test"
    def mock_top_features(X):
        return [{"feature": "f1", "importance": 0.1, "value": 0.5}]
    p._top_features = mock_top_features
    features_list = [{col: 0.1 for col in FEATURE_COLS}]
    results = p.predict_batch(features_list)
    assert len(results) == 1
    assert results[0]['risk_score'] == 0.6
