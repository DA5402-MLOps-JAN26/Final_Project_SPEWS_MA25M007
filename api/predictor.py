import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os
import joblib
from monitoring.exporter import MODEL_F1

FEATURE_COLS = [
    'weekly_clicks_current', 'cumulative_clicks', 'weeks_since_active', 'click_trend_slope',
    'latest_score', 'avg_weighted_score', 'missed_assessments', 'avg_days_late',
    'gender_enc', 'disability_enc', 'education_level', 'imd_score',
    'num_of_prev_attempts', 'studied_credits', 'early_unreg'
]

MODEL_NAME = os.getenv("MODEL_NAME", "student_dropout_best")
ALIAS = os.getenv("MODEL_ALIAS", "production")

class Predictor:
    def __init__(self):
        self.model = None
        self.model_version = "unknown"
        self.model_run_id = "unknown"

    def load_model(self, tracking_uri: str = "http://127.0.0.1:5000",
                   model_name: str = MODEL_NAME,
                   alias: str = ALIAS):
        # First try to load from the bundled pickle file (Docker-friendly)
        pickle_path = '/app/models/production_model.pkl'
        if os.path.exists(pickle_path):
            self.model = joblib.load(pickle_path)
            self.model_version = "docker-bundled"
            self.model_run_id = "bundled"
            print("Loaded bundled model from pickle")
            self._update_model_f1_metric(tracking_uri, model_name, alias)
            return

        # Fallback to MLflow
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            model_uri = f"models:/{model_name}@{alias}"
            self.model = mlflow.sklearn.load_model(model_uri)
            version = client.get_model_version_by_alias(model_name, alias)
            self.model_version = version.version
            self.model_run_id = version.run_id
            print(f"Loaded {model_name}@{alias} v{self.model_version}")
            self._update_model_f1_metric(tracking_uri, model_name, alias)
        except Exception as e:
            print(f"MLflow load failed: {e}")
            self._load_fallback_model()

    def _load_fallback_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=15, random_state=42)
        self.model = RandomForestClassifier(n_estimators=30, random_state=42)
        self.model.fit(X, y)
        self.model_version = "fallback"
        print("Fallback model active")
        MODEL_F1.set(0.0)  # Fallback model F1 unknown

    def _update_model_f1_metric(self, tracking_uri, model_name, alias):
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            version = client.get_model_version_by_alias(model_name, alias)
            run = client.get_run(version.run_id)
            f1 = run.data.metrics.get('f1_at_risk', 0.0)
            MODEL_F1.set(f1)
            print(f"Updated model F1 metric: {f1:.4f}")
        except Exception as e:
            print(f"Could not update model F1 metric: {e}")

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        X = pd.DataFrame([features])
        Xf = X.reindex(columns=FEATURE_COLS, fill_value=0)
        prob = float(self.model.predict_proba(Xf)[0][1])
        if prob < 0.35:
            level = "Low"
        elif prob < 0.65:
            level = "Medium"
        else:
            level = "High"
        return {
            'risk_score': round(prob, 4),
            'risk_level': level,
            'top_features': self._top_features(Xf),
            'model_version': str(self.model_version),
        }

    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        return [self.predict_single(f) for f in features_list]

    def _top_features(self, X: pd.DataFrame) -> List[Dict]:
        model = self.model
        if hasattr(model, 'named_steps'):
            clf = list(model.named_steps.values())[-1]
        else:
            clf = model
        importances = getattr(clf, 'feature_importances_', np.ones(len(FEATURE_COLS)))
        if len(importances) != len(FEATURE_COLS):
            importances = np.ones(len(FEATURE_COLS))
        pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:3]
        return [{'feature': f, 'importance': round(float(i), 4),
                 'value': round(float(X[f].values[0]), 4)} for f, i in pairs]

    @property
    def is_loaded(self):
        return self.model is not None