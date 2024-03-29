from enum import Enum

class Imputation(str, Enum):
    MEAN_IMPUTATION = "MEAN_IMPUTATION"
    ZERO_INTERPOLATION = "ZERO_INTERPOLATION"
    LINEAR_INTERPOLATION = "LINEAR_INTERPOLATION"
    QUADRATIC_INTERPOLATION = "QUADRATIC_INTERPOLATION"
    CUBIC_INTERPOLATION = "CUBIC_INTERPOLATION"
    # MOVING_AVERAGE_IMPUTATION = "MOVING_AVERAGE_IMPUTATION"
    # LAST_OBSERVATION_CARRIED_FORWARD_IMPUTATION = "LAST_OBSERVATION_CARRIED_FORWARD_IMPUTATION"