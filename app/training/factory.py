from typing import Callable, Dict

from sklearn.impute import SimpleImputer
from pyts.preprocessing import InterpolationImputer

from sklearn.preprocessing import MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ConfigSpace import ConfigurationSpace

from app.util.training_parameters import Imputation, Normalizer, Classifier
from app.util.ml_objects import IImputer, INormalizer, IClassifier


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


def get_imputer(imputation: Imputation) -> IImputer:
    return ImputerDict[imputation]()


NormalizerDict: Dict[Normalizer, Callable[[], INormalizer]] = {
    Normalizer.MIN_MAX_SCALER: lambda: MinMaxScaler(),
    Normalizer.NORMALIZER: lambda: Normalizer(),
    Normalizer.QUANTILE_TRANSFORMER: lambda: QuantileTransformer(),
    Normalizer.ROBUST_SCALER: lambda: RobustScaler(),
    Normalizer.STANDARD_SCALER: lambda: StandardScaler()
}


def get_normalizer(normalization: Normalizer) -> INormalizer:
    return NormalizerDict[normalization]()


ClassifierDict: Dict[Classifier, Callable[[], IClassifier]] = {
    Classifier.MLP_CLASSIFIER: lambda: MLPClassifier(),
    Classifier.SVC_CLASSIFIER: lambda: SVC(),
    Classifier.RANDOM_FOREST_CLASSIFIER: lambda: RandomForestClassifier(),
    Classifier.KNEIGHBORS_CLASSIFIER: lambda: KNeighborsClassifier()
}

#TODO hyperparameters here ??
def get_classifier(classifier: Classifier, hyperparameters: ConfigurationSpace) -> IClassifier:
    return ClassifierDict[classifier]()