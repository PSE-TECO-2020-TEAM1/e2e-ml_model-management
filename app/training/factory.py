from app.util.parameters import Imputation, Normalization, Classifier, Feature
from app.util.ml_objects import IImputer, INormalizer, IClassifier

from typing import Callable, Dict
from pandas import np

from sklearn.impute import SimpleImputer
from pyts.preprocessing import InterpolationImputer

from sklearn.preprocessing import MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


ImputerDict: Dict[Imputation, Callable[[], IImputer]] = {
    Imputation.MEAN_IMPUTATION: lambda: SimpleImputer(missing_values=np.nan, strategy="mean"),
    Imputation.ZERO_INTERPOLATION: lambda: InterpolationImputer(missing_values=np.nan, strategy="zero"),
    Imputation.LINEAR_INTERPOLATION: lambda: InterpolationImputer(missing_values=np.nan, strategy="linear"),
    Imputation.QUADRATIC_INTERPOLATION: lambda: InterpolationImputer(missing_values=np.nan, strategy="quadratic"),
    Imputation.CUBIC_INTERPOLATION: lambda: InterpolationImputer(missing_values=np.nan, strategy="cubic")
    #TODO
    # Imputation.MOVING_AVERAGE_IMPUTATION: ,
    # Imputation.LAST_OBSERVATION_CARRIED_FORWARD_IMPUTATION:
}


def getImputer(imputation: Imputation) -> IImputer:
    return ImputerDict[imputation]()


NormalizerDict: Dict[Normalization, Callable[[], INormalizer]] = {
    Normalization.MIN_MAX_SCALER: lambda: MinMaxScaler(),
    Normalization.NORMALIZER: lambda: Normalizer(),
    Normalization.QUANTILE_TRANSFORMER: lambda: QuantileTransformer(),
    Normalization.ROBUST_SCALER: lambda: RobustScaler(),
    Normalization.STANDARD_SCALER: lambda: StandardScaler()
}


def getNormalizer(normalization: Normalization) -> INormalizer:
    return NormalizerDict[normalization]()


ClassifierDict: Dict[Classifier, Callable[[], IClassifier]] = {
    Classifier.MLP_CLASSIFIER: lambda: MLPClassifier(),
    Classifier.SV_CLASSIFIER: lambda: SVC(),
    Classifier.RANDOM_FOREST_CLASSIFIER: lambda: RandomForestClassifier(),
    Classifier.KNEIGHBORS_CLASSIFIER: lambda: KNeighborsClassifier()
}

#TODO hyperparameters here ??
def getClassifier(classifier: Classifier, hyperparameters: ConfigurationSpace) -> IClassifier:
    return ClassifierDict[classifier]()