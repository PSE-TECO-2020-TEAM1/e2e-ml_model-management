from typing import Dict, Optional

from app.models.mongo_model import OID, MongoModel
from app.util.ml_objects import IClassifier, INormalizer
from app.util.performance_metrics import PerformanceMetric


class PerformanceMetrics(MongoModel):
    metrics: Dict[PerformanceMetric, int]


class PerformanceMetricsPerLabel(MongoModel):
    labels: Dict[str, PerformanceMetrics]


class ML_Model(MongoModel):
    id: Optional[OID] = None
    name: str
    window_size: int
    sliding_step: int
    performance_metrics: PerformanceMetricsPerLabel
    # imputer_object: IImputer
    normalizer_object: INormalizer
    classifier_object: IClassifier
