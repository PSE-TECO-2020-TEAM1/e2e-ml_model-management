from enum import Enum

#TODO lowercase all ???
class Imputation(str, Enum):
    MEAN_IMPUTATION = "mean_imputation"
    ZERO_INTERPOLATION = "zero_interpolation"
    LINEAR_INTERPOLATION = "linear_interpolation"
    QUADRATIC_INTERPOLATION = "quadratic_interpolation"
    CUBIC_INTERPOLATION = "cubic_interpolation"
    MOVING_AVERAGE_IMPUTATION = "moving_average_imputation"
    LAST_OBSERVATION_CARRIED_FORWARD_IMPUTATION = "last_observation_carried_forward_imputation"


class Normalization(str, Enum):
    MIN_MAX_SCALER = "min_max_scaler"
    NORMALIZER = "normalizer"
    QUANTILE_TRANSFORMER = "quantile_transformer"
    ROBUST_SCALER = "robust_scaler"
    STANDARD_SCALER = "standard_scaler"

class Feature(str, Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    VARIANCE = "variance"
    ABS_ENERGY = "abs_energy"
    MEAN = "mean"
    MEDIAN = "median"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    # TODO add more features

class Classifier(str, Enum):
    KNEIGHBORS_CLASSIFIER = "kneighbors_classifier"
    MLP_CLASSIFIER = "mlp_classifier"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    SVC_CLASSIFIER = "svc_classifier"
