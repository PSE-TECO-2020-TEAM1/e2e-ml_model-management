from pydantic import BaseModel
from app.models.mongo_model import MongoModel, OID
from app.models.ml_model import ML_Model
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel

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
    classifier_selections: List[ClassifierSelection]


class GetModelsRes(MongoModel):
    models: List[Tuple[OID, str]]

class GetModelRes(MongoModel):
    model: ML_Model

class GetPredictionConfigRes(MongoModel):
    pass # TODO

class GetTrainingProgressRes(MongoModel):
    progress: int

class GetPredictionIdRes(MongoModel):
    pass # TODO