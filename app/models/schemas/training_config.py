from app.models.schemas.mongo_model import MongoModel
from typing import Any, Dict, List
from pydantic.main import BaseModel
from app.ml.objects.classification import Classifier
from app.ml.objects.normalization import Normalization
from app.ml.objects.imputation import Imputation
from app.ml.objects.feature import Feature

class PerComponentConfigInTrain(MongoModel):
    sensor: str
    component: str
    features: List[Feature]
    imputation: Imputation
    normalizer: Normalization

class TrainingConfigInTrain(MongoModel):
    modelName: str
    windowSize: int
    slidingStep: int
    perComponentConfigs: List[PerComponentConfigInTrain]
    classifier: Classifier
    hyperparameters: Dict[str, Any]

class HyperparameterInResponse(MongoModel):
    name: str
    value: Any

class TrainingConfigInResponse(BaseModel):
    modelName: str
    windowSize: int
    slidingStep: int
    perComponentConfigs: List[PerComponentConfigInTrain]
    classifier: Classifier
    hyperparameters: List[HyperparameterInResponse]