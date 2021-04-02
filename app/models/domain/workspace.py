from app.models.domain.ml_model import MlModel
from bson.objectid import ObjectId
from app.models.domain.training_data_set import TrainingDataSet
from app.models.domain.sensor import Sensor
from typing import List
from dataclasses import dataclass

from app.models.domain.prediction_id import PredictionID
from app.models.domain.training_state import TrainingState

@dataclass
class Workspace():
    user_id: ObjectId
    sensors: List[Sensor]
    training_data_set: TrainingDataSet
    training_state: TrainingState
    prediction_IDs: List[PredictionID] = []
    trained_models: List[MlModel] = []
