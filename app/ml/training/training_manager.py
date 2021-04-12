from app.ml.training.config_parsing import parse_config_for_training
from multiprocessing.context import Process
from app.db.syncdb import create_sync_db
from app.workspace_management_api.workspace_data_source import WorkspaceDataSource
from app.ml.training.trainer import Trainer
from bson.objectid import ObjectId
from app.ml.training.data_set_manager import DataSetManager
from app.models.domain.sensor import Sensor, make_sensor_component
from typing import Dict
from app.ml.objects.classification.classifier_config_spaces.util import validate_and_parse_hyperparameters
from app.models.schemas.training_config import TrainingConfigInTrain


def initiate_new_training(workspace_id: ObjectId, training_config_in_train: TrainingConfigInTrain):
    config = parse_config_for_training(training_config_in_train)
    data_set_manager = DataSetManager(workspace_id, WorkspaceDataSource())
    trainer = Trainer(training_config=config, data_set_manager=data_set_manager, create_db=create_sync_db)
    Process(target=trainer.train).start()


def validate_config_and_parse_hyperparameters(training_config: TrainingConfigInTrain, workspace_sensors: Dict[str, Sensor]):
    if training_config.windowSize < 4:
        raise ValueError("Window size must be greater than or equal to 4")
    if training_config.slidingStep < 1:
        raise ValueError("Sliding step must be greater than or equal to 1")
    if training_config.slidingStep > training_config.windowSize:
        raise ValueError("Sliding step cannot excess the window size")

    for per_component_config in training_config.perComponentConfigs:
        sensor_name = per_component_config.sensor
        component = make_sensor_component(sensor_name, per_component_config.component)
        if sensor_name not in workspace_sensors:
            raise ValueError(sensor_name + " is not a sensor of this workspace")
        if component not in workspace_sensors[sensor_name].components:
            raise ValueError(component + " is not a valid component of " + sensor_name)

    validate_and_parse_hyperparameters(training_config.classifier, training_config.hyperparameters)
