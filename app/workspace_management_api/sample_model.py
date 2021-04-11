from typing import List
from dataclasses import dataclass

@dataclass(frozen=True)
class Timeframe():
    start: int
    end: int

@dataclass
class DataPoint():
    data: List[float]
    timestamp: int


@dataclass
class DataPointsPerSensor():
    sensorName: str
    dataPoints: List[DataPoint]

@dataclass
class SampleFromWorkspace():
    label: str
    start: int
    end: int
    timeFrames: List[Timeframe]
    sensorDataPoints: List[DataPointsPerSensor]

