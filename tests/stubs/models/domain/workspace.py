from app.models.domain.training_state import TrainingState
from app.models.domain.sensor import Sensor
from app.models.domain.workspace import Workspace

from tests.stubs.models.domain.training_data_set import training_data_set_stub

workspace_stub = Workspace(
    _id="60706e79e8584d11339c91f3",
    user_id="60706eabae5c8c01b0c43e33",
    sensors={"Accelerometer": Sensor(sampling_rate=60, components=["x", "y", "z"]),
                "Gyroscope": Sensor(sampling_rate=50, components=["x", "y", "z"])},
    training_data_set=training_data_set_stub,
    training_state=TrainingState.NO_ACTIVE_TRAINING,
    prediction_IDs=[], # TODO
    trained_ml_model_refs=[]
)