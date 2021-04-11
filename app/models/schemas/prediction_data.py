from app.models.schemas.mongo_model import MongoModel, OID
from typing import List


class DataPointInPredict(MongoModel):
    data: List[float]
    timestamp: int


class DataPointsPerSensorInPredict(MongoModel):
    sensor: str
    dataPoints: List[DataPointInPredict]

class SampleInPredict(MongoModel):
    start: int
    end: int
    sensorDataPoints: List[DataPointsPerSensorInPredict]

class PredictionData(MongoModel):
    predictionId: OID
    sample: SampleInPredict
