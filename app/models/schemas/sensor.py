from app.models.schemas.mongo_model import MongoModel
from typing import List
from pydantic import BaseModel

class SensorInWorkspace(MongoModel):
    name: str
    samplingRate: int
    dataFormat: List[str]

class SensorInPredictionConfig(MongoModel):
    name: str
    samplingRate: int