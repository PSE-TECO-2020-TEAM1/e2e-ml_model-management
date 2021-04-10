from app.models.schemas.training_config import TrainingConfigInResponse
from typing import Dict, List
from pydantic import BaseModel
from app.models.schemas.mongo_model import OID

class SingleMetric(BaseModel):
    name: str
    score: float

class PerformanceMetrics(BaseModel):
    label: str
    metrics: List[SingleMetric]

class MlModelInResponse(BaseModel):
    config: TrainingConfigInResponse
    labelPerformanceMetrics: List[PerformanceMetrics]

class MlModelMetadataInResponse(BaseModel):
    id: OID
    name: str