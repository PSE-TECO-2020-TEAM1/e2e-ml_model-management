from app.models.schemas.training_config import TrainingConfigInResponse
from typing import List
from app.models.schemas.mongo_model import MongoModel, OID

class SingleMetricInResponse(MongoModel):
    name: str
    score: float

class PerformanceMetricsInResponse(MongoModel):
    label: str
    metrics: List[SingleMetricInResponse]

class MlModelInResponse(MongoModel):
    config: TrainingConfigInResponse
    labelPerformanceMetrics: List[PerformanceMetricsInResponse]

class MlModelMetadataInResponse(MongoModel):
    id: OID
    name: str