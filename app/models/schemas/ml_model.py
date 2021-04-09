from typing import Dict, List
from pydantic import BaseModel
from app.models.schemas.mongo_model import OID
from app.ml.objects.classification import Classifier

class SingleMetric(BaseModel):
    name: str
    score: float

class PerformanceMetrics(BaseModel):
    label: str
    metrics: List[SingleMetric]

class MlModelInResponse(BaseModel):
    # TODO IMPORTANT TRAINING CONFIG HERE
    classifier: Classifier
    hyperparameters: List[Dict]
    labelPerformanceMetrics: List[PerformanceMetrics]

class MlModelMetadataInResponse(BaseModel):
    id: OID
    name: str