from app.models.domain.db_doc import DbDoc
from bson.objectid import ObjectId
from app.models.domain.training_data_set import TrainingDataSet
from app.models.domain.sensor import Sensor
from typing import Dict, List
from dataclasses import dataclass, field
from app.ml.training.training_state import TrainingState

@dataclass
class Workspace(DbDoc):
    user_id: ObjectId
    sensors: Dict[str, Sensor]
    training_data_set: TrainingDataSet = TrainingDataSet()
    training_state: TrainingState = TrainingState.NO_TRAINING_YET
    ml_model_refs: List[ObjectId] = field(default_factory=list)
