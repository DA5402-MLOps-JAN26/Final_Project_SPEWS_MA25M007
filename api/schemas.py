# api/schemas.py 
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class StudentFeatures(BaseModel):
    id_student: int
    week_number: int = Field(..., ge=1, le=40)
    weekly_clicks_current: float = 0.0
    cumulative_clicks: float = 0.0
    weeks_since_active: float = 0.0
    click_trend_slope: float = 0.0
    latest_score: float = 0.0
    avg_weighted_score: float = 0.0
    missed_assessments: float = 0.0
    avg_days_late: float = 0.0
    gender_enc: int = 0
    disability_enc: int = 0
    education_level: float = 2.0
    imd_score: float = 5.0
    num_of_prev_attempts: int = 0
    studied_credits: float = 60.0
    early_unreg: int = 0

class PredictRequest(BaseModel):
    student: StudentFeatures

class PredictResponse(BaseModel):
    id_student: int
    week_number: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    top_features: List[Dict[str, Any]]
    model_version: str
    prediction_id: str

class BatchPredictRequest(BaseModel):
    students: List[StudentFeatures]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    total: int
    high_risk_count: int
    model_version: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ReadyResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    alias: str
    run_id: str
    f1_score: Optional[float] = None
