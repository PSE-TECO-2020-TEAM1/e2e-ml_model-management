from typing import Any, Dict
from app.ml.training.parameters.imputations import Imputation
from app.ml.training.parameters.normalizations import Normalization
from app.ml.training.parameters.classifiers import Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from pyts.preprocessing import InterpolationImputer
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)

ImputerDict = {
    Imputation.MEAN_IMPUTATION: lambda: SimpleImputer(strategy="mean"),
    Imputation.ZERO_INTERPOLATION: lambda: InterpolationImputer(strategy="zero"),
    Imputation.LINEAR_INTERPOLATION: lambda: InterpolationImputer(strategy="linear"),
    Imputation.QUADRATIC_INTERPOLATION: lambda: InterpolationImputer(strategy="quadratic"),
    Imputation.CUBIC_INTERPOLATION: lambda: InterpolationImputer(strategy="cubic")
    # TODO add these imputations
    # Imputation.MOVING_AVERAGE_IMPUTATION: ,
    # Imputation.LAST_OBSERVATION_CARRIED_FORWARD_IMPUTATION:
}


def get_imputer(imputation: Imputation):
    return ImputerDict[imputation]()


NormalizerDict = {
    Normalization.MIN_MAX_SCALER: lambda: MinMaxScaler(),
    Normalization.NORMALIZER: lambda: Normalizer(),
    Normalization.QUANTILE_TRANSFORMER: lambda: QuantileTransformer(),
    Normalization.ROBUST_SCALER: lambda: RobustScaler(),
    Normalization.STANDARD_SCALER: lambda: StandardScaler()
}


def get_normalizer(normalization: Normalization):
    return NormalizerDict[normalization]()


ClassifierDict = {
    Classifier.MLP_CLASSIFIER: lambda hyperparameters: MLPClassifier(**hyperparameters),
    Classifier.SVC_CLASSIFIER: lambda hyperparameters: SVC(**hyperparameters),
    Classifier.RANDOM_FOREST_CLASSIFIER: lambda hyperparameters: RandomForestClassifier(**hyperparameters),
    Classifier.KNEIGHBORS_CLASSIFIER: lambda hyperparameters: KNeighborsClassifier(**hyperparameters)
}


def get_classifier(classifier: Classifier, hyperparameters: Dict[str, Any]):
    return ClassifierDict[classifier](hyperparameters)
