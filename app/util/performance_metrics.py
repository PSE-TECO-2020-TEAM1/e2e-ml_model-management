from enum import Enum


class PerformanceMetric(str, Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1-score"
    SUPPORT = "support"

