from typing import List
from dataclasses import dataclass

@dataclass(frozen=True)
class SingleMetric():
    name: str
    score: float

@dataclass(frozen=True)
class PerformanceMetrics():
    label: str
    metrics: List[SingleMetric]