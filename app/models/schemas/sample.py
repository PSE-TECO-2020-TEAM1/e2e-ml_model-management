from typing import List
from pydantic import BaseModel


class DataPoint(BaseModel):
    data: List[float]
    timestamp: int


class DataPointsPerSensor(BaseModel):
    sensor: str
    dataPoints: List[DataPoint]


class SampleInSubmit(BaseModel):
    # predictionId: str (create a new class to encapsulate this and sample in submit field maybe with some method?)
    start: int
    end: int
    sensorDataPoints: List[DataPointsPerSensor]
