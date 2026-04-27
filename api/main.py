#api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import uuid, os, time

from api.schemas import (
    PredictRequest, PredictResponse, BatchPredictRequest,
    BatchPredictResponse, HealthResponse, ReadyResponse,
    ModelInfoResponse, RiskLevel
)
from api.predictor import Predictor
from monitoring.exporter import (
    REQUEST_COUNT, REQUEST_LATENCY,
    PREDICTION_DISTRIBUTION, PREDICTION_CONFIDENCE
)

predictor = Predictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_name = os.getenv("MODEL_NAME", "student_dropout_best")
    alias = os.getenv("MODEL_ALIAS", "production")
    predictor.load_model(tracking_uri, model_name, alias)
    
    # Seed all gauges on startup so Grafana has data immediately
    from monitoring.exporter import (
        update_drift_metrics, update_model_metrics, PIPELINE_OK, ERROR_RATE
    )
    update_model_metrics(0.8574)
    update_drift_metrics({
        "vle_sum_click": 0.0139,
        "assessment_score": 0.0,
        "avg_days_late": 0.0
    })
    PIPELINE_OK.set(1)
    ERROR_RATE.set(0.0)
    yield
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
#     model_name = os.getenv("MODEL_NAME", "student_dropout_best")
#     alias = os.getenv("MODEL_ALIAS", "production")
#     predictor.load_model(tracking_uri, model_name, alias)
#     yield

app = FastAPI(
    title="SPEWS - Student Performance Early Warning System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", timestamp=datetime.utcnow().isoformat())

@app.get("/ready", response_model=ReadyResponse)
async def ready():
    return ReadyResponse(
        status="ready" if predictor.is_loaded else "not_ready",
        model_version=str(predictor.model_version),
        model_loaded=predictor.is_loaded
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        model_name="student_dropout_best",
        version=str(predictor.model_version),
        alias="production",
        run_id=str(predictor.model_run_id),
        f1_score=getattr(predictor, 'f1', None)
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    try:
        feat = request.student.model_dump()
        result = predictor.predict_single(feat)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
        PREDICTION_DISTRIBUTION.labels(risk_level=result['risk_level']).inc()
        PREDICTION_CONFIDENCE.observe(result['risk_score'])
        return PredictResponse(
            id_student=request.student.id_student,
            week_number=request.student.week_number,
            risk_score=result['risk_score'],
            risk_level=RiskLevel(result['risk_level']),
            top_features=result['top_features'],
            model_version=result['model_version'],
            prediction_id=str(uuid.uuid4())
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for student in request.students:
        feat = student.model_dump()
        r = predictor.predict_single(feat)
        results.append(PredictResponse(
            id_student=student.id_student,
            week_number=student.week_number,
            risk_score=r['risk_score'],
            risk_level=RiskLevel(r['risk_level']),
            top_features=r['top_features'],
            model_version=r['model_version'],
            prediction_id=str(uuid.uuid4())
        ))
    high_risk = sum(1 for r in results if r.risk_level == RiskLevel.HIGH)
    return BatchPredictResponse(
        predictions=results,
        total=len(results),
        high_risk_count=high_risk,
        model_version=predictor.model_version
    )

@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
