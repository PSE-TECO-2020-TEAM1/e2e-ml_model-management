from bson.objectid import ObjectId
from app.models.domain.training_data_set import TrainingDataSet
from app.models.domain.sensor import Sensor
from typing import Dict, List
from dataclasses import dataclass, field

from app.models.domain.training_state import TrainingState

@dataclass
class Workspace():
    _id: ObjectId
    user_id: ObjectId
    sensors: Dict[str, Sensor]
    training_data_set: TrainingDataSet
    training_state: TrainingState
    trained_ml_model_refs: List[ObjectId] = field(default_factory=list)
