from typing import List
from app.ml.objects.classification import Classifier
from app.ml.objects.feature import Feature
from app.ml.objects.imputation import Imputation
from app.ml.objects.normalization import Normalization

def get_features() -> List[Feature]:
    return [feature for feature in Feature]

def get_imputations() -> List[Imputation]:
    return [imputation for imputation in Imputation]

def get_normalizations() -> List[Normalization]:
    return [normalization for normalization in Normalization]

def get_classifiers() -> List[Classifier]:
    return [classifier for classifier in Classifier]