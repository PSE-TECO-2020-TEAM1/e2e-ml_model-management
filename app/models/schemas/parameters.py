from typing import Any, Dict, List
from pydantic import BaseModel
from app.ml.objects.classification import Classifier
from app.ml.objects.feature import Feature
from app.ml.objects.imputation import Imputation
from app.ml.objects.normalization import Normalization

class ClassifierSelection(BaseModel):
    classifier: Classifier
    hyperparameters: Dict[str, Dict[str, Any]]
    conditions: List[str]


class ParametersInResponse(BaseModel):
    features: List[Feature]
    imputations: List[Imputation]
    normalizations: List[Normalization]
    classifierSelections: List[ClassifierSelection]