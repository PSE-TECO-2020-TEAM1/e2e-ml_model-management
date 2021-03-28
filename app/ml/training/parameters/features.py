from enum import Enum

class Feature(str, Enum):
    MINIMUM = "MINIMUM"
    MAXIMUM = "MAXIMUM"
    VARIANCE = "VARIANCE"
    ABS_ENERGY = "ABS_ENERGY"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    SKEWNESS = "SKEWNESS"
    KURTOSIS = "KURTOSIS"