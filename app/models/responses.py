from pydantic import BaseModel
from app.models.mongo_model import OID
from app.models.ml_model import ML_Model
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel

from app.util.training_parameters import Classifier, Feature, Imputation, Normalization


class ClassifierSelection(BaseModel):
    classifier: Classifier
    hyperparameters: Dict[str, Dict[str, Any]]
    conditions: List[str]


class GetParametersRes(BaseModel):
    # TODO sliding window min max constraints here ?
    features: List[Feature]
    imputers: List[Imputation]
    normalizers: List[Normalization]
    # windowSize: Tuple[int, int]
    # slidingStep: Tuple[int, int]
    classifier_selections: List[ClassifierSelection]


class GetModelsRes(BaseModel):
    models: List[Tuple[OID, str]]

class GetModelRes(BaseModel):
    model: ML_Model

class GetPredictionConfigRes(BaseModel):
    pass # TODO

class GetTrainingProgressRes(BaseModel):
    pass # TODO

class GetPredictionIdRes(BaseModel):
    pass # TODO