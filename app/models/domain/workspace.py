from app.models.domain.training_data_set import TrainingDataSet
from app.models.domain.trained_models import MachineLearningModel
from app.models.domain.sensor import Sensor
from typing import List
from pydantic.fields import Field

from app.models.domain.prediction_id import PredictionID
from app.models.domain.training_state import TrainingState
from app.models.domain.mongo_model import MongoModel, OID

class Workspace(MongoModel):
    id: OID = Field(None, alias="_id")
    user_id: OID
    training_state: TrainingState
    prediction_IDs: List[PredictionID]
    sensors: List[Sensor]
    training_data_set: TrainingDataSet
    trained_models: List[MachineLearningModel]
