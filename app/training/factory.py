from typing import Any, Callable, Dict

import numpy
from app.util.ml_objects import IClassifier, IImputer, INormalizer
from app.util.training_parameters import Classifier, Imputation, Normalization
from pyts.preprocessing import InterpolationImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVC

ImputerDict: Dict[Imputation, Callable[[], IImputer]] = {
    Imputation.MEAN_IMPUTATION: lambda: SimpleImputer(missing_values=numpy.nan, strategy="mean"),
    Imputation.ZERO_INTERPOLATION: lambda: InterpolationImputer(missing_values=numpy.nan, strategy="zero"),
    Imputation.LINEAR_INTERPOLATION: lambda: InterpolationImputer(missing_values=numpy.nan, strategy="linear"),
    Imputation.QUADRATIC_INTERPOLATION: lambda: InterpolationImputer(missing_values=numpy.nan, strategy="quadratic"),
    Imputation.CUBIC_INTERPOLATION: lambda: InterpolationImputer(missing_values=numpy.nan, strategy="cubic")
    # TODO add these imputations
    # Imputation.MOVING_AVERAGE_IMPUTATION: ,
    # Imputation.LAST_OBSERVATION_CARRIED_FORWARD_IMPUTATION:
}


def get_imputer(imputation: Imputation) -> IImputer:
    return ImputerDict[imputation]()


NormalizerDict: Dict[Normalization, Callable[[], INormalizer]] = {
    Normalization.MIN_MAX_SCALER: lambda: MinMaxScaler(),
    Normalization.NORMALIZER: lambda: Normalizer(),
    Normalization.QUANTILE_TRANSFORMER: lambda: QuantileTransformer(),
    Normalization.ROBUST_SCALER: lambda: RobustScaler(),
    Normalization.STANDARD_SCALER: lambda: StandardScaler()
}


def get_normalizer(normalizer: Normalization) -> INormalizer:
    return NormalizerDict[normalizer]()


ClassifierDict: Dict[Classifier, Callable[[], IClassifier]] = {
    Classifier.MLP_CLASSIFIER: lambda hyperparameters: MLPClassifier(**hyperparameters),
    Classifier.SVC_CLASSIFIER: lambda hyperparameters: SVC(**hyperparameters),
    Classifier.RANDOM_FOREST_CLASSIFIER: lambda hyperparameters: RandomForestClassifier(**hyperparameters),
    Classifier.KNEIGHBORS_CLASSIFIER: lambda hyperparameters: KNeighborsClassifier(**hyperparameters)
}

def get_classifier(classifier: Classifier, hyperparameters: Dict[str, Any]) -> IClassifier:
    return ClassifierDict[classifier](hyperparameters)
