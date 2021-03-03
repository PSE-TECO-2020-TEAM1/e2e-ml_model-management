from app.models.workspace import Sensor
from pydantic.fields import Field
from app.util.training_parameters import Classifier, Feature, Imputation, Normalization
from typing import Any, Dict, List

from app.models.mongo_model import OID, MongoModel
from app.util.performance_metrics import PerformanceMetric


class PerformanceMetrics(MongoModel):
    metrics: Dict[PerformanceMetric, float]


class PerformanceMetricsPerLabel(MongoModel):
    metrics_of_labels: Dict[str, PerformanceMetrics]


class MlModel(MongoModel):
    id: OID = Field(None, alias="_id")
    name: str
    workspaceId: OID
    windowSize: int
    slidingStep: int
    sortedFeatures: List[Feature]
    imputation: Imputation
    imputerObject: OID
    normalization: Normalization
    normalizerObject: OID
    classifier: Classifier
    classifierObject: OID
    hyperparameters: Dict[str, Any]
    labelPerformanceMetrics: PerformanceMetricsPerLabel
    columnOrder: List[str]
    sensors: List[Sensor]
    labelCodeToLabel: Dict[str, str]
