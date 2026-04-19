# monitoring/exporter.py 
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    'spews_api_requests_total',
    'Total API requests by endpoint and status',
    ['endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'spews_api_latency_seconds',
    'API request latency in seconds',
    ['endpoint'],
    buckets=[0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)
PREDICTION_DISTRIBUTION = Counter(
    'spews_prediction_risk_total',
    'Prediction count by risk level',
    ['risk_level']
)
PREDICTION_CONFIDENCE = Histogram(
    'spews_prediction_confidence',
    'Model confidence score distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
PSI_SCORE = Gauge(
    'spews_drift_psi_score',
    'PSI drift score by feature',
    ['feature']
)
MODEL_F1 = Gauge('spews_model_f1_score', 'Current production model F1 score')
ERROR_RATE = Gauge('spews_api_error_rate', 'Rolling API error rate (0-1)')
PIPELINE_OK = Gauge('spews_pipeline_success', 'Last Airflow run success (1) or failure (0)')

def update_drift_metrics(psi_dict: dict):
    for feature, score in psi_dict.items():
        PSI_SCORE.labels(feature=feature).set(score)

def update_model_metrics(f1: float):
    MODEL_F1.set(f1)
