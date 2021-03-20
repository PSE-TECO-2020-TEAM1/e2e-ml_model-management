from pydantic.fields import Field
from app.models.workspace import Sensor
from app.models.mongo_model import MongoModel, OID
from app.models.ml_model import PerformanceMetrics
from typing import Any, Dict, List, Tuple

from app.util.training_parameters import Classifier, Feature, Imputation, Normalization


class ClassifierSelection(MongoModel):
    classifier: Classifier
    hyperparameters: Dict[str, Dict[str, Any]]
    conditions: List[str]


class GetParametersRes(MongoModel):
    features: List[Feature]
    imputers: List[Imputation]
    normalizers: List[Normalization]
    windowSize: Tuple[int, int]
    slidingStep: Tuple[int, int]
    classifierSelections: List[ClassifierSelection]


class OneModelInGetModelsRes(MongoModel):
    id: OID
    name: str


class GetModelsRes(MongoModel):
    models: List[OneModelInGetModelsRes] = Field([])


class GetModelRes(MongoModel):
    id: OID
    name: str
    windowSize: int
    slidingStep: int
    sortedFeatures: List[Feature]
    imputation: Imputation
    normalization: Normalization
    classifier: Classifier
    hyperparameters: List[Dict]
    labelPerformanceMetrics: List[PerformanceMetrics]


class GetPredictionConfigRes(MongoModel):
    sensors: List[Sensor]


class GetTrainingProgressRes(MongoModel):
    progress: int


class GetPredictionIdRes(MongoModel):
    predictionId: str

class GetPredictionResultsRes(MongoModel):
    labels: List[str]
    start: int
    end: int
