from bson.objectid import ObjectId
from app.ml.training.training_state import TrainingState
from app.models.domain.sensor import Sensor
from app.models.domain.workspace import Workspace

from tests.stubs.models.domain.training_data_set import get_training_data_set_stub

def get_workspace_stub():
    return Workspace(
        _id=ObjectId("60706e79e8584d11339c91f3"),
        user_id=ObjectId("60706eabae5c8c01b0c43e33"),
        sensors={"Accelerometer": Sensor(sampling_rate=60, components=["x", "y", "z"]),
                 "Gyroscope": Sensor(sampling_rate=50, components=["x", "y", "z"])},
        training_data_set=get_training_data_set_stub(),
        training_state=TrainingState.NO_TRAINING_YET,
        ml_model_refs=["60736013b961d239c76711a3", "607367362d98418cae5a1522"])
