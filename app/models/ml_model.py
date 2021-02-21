from typing import Any, Dict, Optional

from bson.binary import Binary

from app.models.mongo_model import OID, MongoModel
from app.util.performance_metrics import PerformanceMetric


class PerformanceMetrics(MongoModel):
    metrics: Dict[PerformanceMetric, float]


class PerformanceMetricsPerLabel(MongoModel):
    metrics_of_labels: Dict[str, PerformanceMetrics]


class ML_Model(MongoModel):
    id: Optional[OID] = None
    name: str
    workspace_id: OID
    window_size: int
    sliding_step: int
    label_performance_metrics: PerformanceMetricsPerLabel
    imputer_object: Binary
    normalizer_object: Binary
    classifier_object: Binary
    hyperparameters: Dict[str, Any]
