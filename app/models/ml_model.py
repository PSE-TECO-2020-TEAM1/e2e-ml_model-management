from typing import Dict, Optional

from app.util.performance_metrics import PerformanceMetric
from app.models.mongo_model import OID, MongoModel

class PerformanceMetrics(MongoModel):
    metrics: Dict[PerformanceMetric, int]

class PerformanceMetricsPerLabel(MongoModel):
    labels: Dict[str, PerformanceMetric]

class ML_Model(MongoModel):
    _id: Optional[OID] = None
    name: str
    window_size: int
    sliding_step: int
