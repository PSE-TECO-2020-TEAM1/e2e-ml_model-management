from app.util.training_parameters import Classifier, Feature, Imputation, Normalization
from typing import Any, Dict, List, Optional

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
    windowSize: int
    slidingStep: int
    features: List[Feature]
    imputation: Imputation
    imputer_object: OID
    normalization: Normalization
    normalizer_object: OID
    classifier: Classifier
    classifier_object: OID
    hyperparameters: Dict[str, Any]
    labelPerformanceMetrics: PerformanceMetricsPerLabel
