from typing import List
from dataclasses import dataclass


@dataclass
class DataPoint():
    data: List[float]
    timestamp: int


@dataclass
class DataPointsPerSensor():
    sensor: str
    dataPoints: List[DataPoint]

@dataclass
class Timeframe():
    start: int
    end: int

@dataclass
class SampleFromWorkspace():
    label: str
    start: int
    end: int
    timeFrames: List[Timeframe]
    sensorDataPoints: List[DataPointsPerSensor]

