from typing import List
from pydantic import BaseModel


class DataPointInPredict(BaseModel):
    data: List[float]
    timestamp: int


class DataPointsPerSensorInPredict(BaseModel):
    sensor: str
    dataPoints: List[DataPointInPredict]

class SampleInPredict:
    start: int
    end: int
    sensorDataPoints: List[DataPointsPerSensorInPredict]

class PredictionData(BaseModel):
    predictionId: str
    sample: SampleInPredict
