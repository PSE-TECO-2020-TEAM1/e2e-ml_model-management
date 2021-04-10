from typing import Any, Dict, List
from pydantic.main import BaseModel
from app.ml.objects.classification import Classifier
from app.ml.objects.normalization import Normalization
from app.ml.objects.imputation import Imputation
from app.ml.objects.feature import Feature

class PerComponentConfigInTrain(BaseModel):
    sensor: str
    component: str
    features: List[Feature]
    imputation: Imputation
    normalization: Normalization

class TrainingConfigInTrain(BaseModel):
    modelName: str
    windowSize: int
    slidingStep: int
    perComponentConfigs: List[PerComponentConfigInTrain]
    classifier: Classifier
    hyperparameters: Dict[str, Any]

class HyperparameterInResponse(BaseModel):
    name: str
    value: Any

class TrainingConfigInResponse(BaseModel):
    modelName: str
    windowSize: int
    slidingStep: int
    perComponentConfigs: List[PerComponentConfigInTrain]
    classifier: Classifier
    hyperparameters: List[HyperparameterInResponse]