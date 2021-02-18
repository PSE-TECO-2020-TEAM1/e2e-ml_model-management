from enum import Enum


class PerformanceMetric(str, Enum):
    PRECISION = "Precision"
    RECALL = "Recall"
    F1_SCORE = "F1-score"
    SUPPORT = "Support"

