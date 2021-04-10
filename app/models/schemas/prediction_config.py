from typing import List
from pydantic import BaseModel
from app.models.schemas.sensor import SensorInPredictionConfig

class PredictionConfig(BaseModel):
    # TODO need anything else ?
    sensors: List[SensorInPredictionConfig]