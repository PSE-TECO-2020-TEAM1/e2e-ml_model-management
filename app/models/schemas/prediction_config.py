from app.models.schemas.mongo_model import MongoModel, OID
from typing import List
from app.models.schemas.sensor import SensorInPredictionConfig

class PredictionIdInResponse(MongoModel):
    predictionId: OID

class PredictionConfig(MongoModel):
    # TODO need anything else ?
    sensors: List[SensorInPredictionConfig]