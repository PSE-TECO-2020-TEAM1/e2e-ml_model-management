from app.models.domain.ml_model import MlModel
from bson.objectid import ObjectId
from app.models.domain.training_data_set import TrainingDataSet
from app.models.domain.sensor import Sensor
from typing import Dict, List
from dataclasses import dataclass, field

from app.models.domain.prediction_id import PredictionID
from app.models.domain.training_state import TrainingState

@dataclass
class Workspace():
    _id: ObjectId
    user_id: ObjectId
    sensors: Dict[str, Sensor]
    training_data_set: TrainingDataSet
    training_state: TrainingState
    prediction_IDs: List[PredictionID] = field(default_factory=list)
    trained_models: List[MlModel] = field(default_factory=list)
