from app.ml.training.config_parsing import parse_config_for_training
from multiprocessing.context import Process
from app.db.syncdb import create_sync_db
from app.workspace_management_api.stub_source_delete import WorkspaceDataSource
from app.ml.training.trainer import Trainer
from bson.objectid import ObjectId
from app.ml.training.data_set_manager import DataSetManager
from app.models.domain.sensor import Sensor
from typing import Dict
from app.ml.objects.classification.classifier_config_spaces.util import validate_hyperparameters
from app.models.schemas.training_config import TrainingConfigInTrain


def initiate_new_training(workspace_id: ObjectId, training_config_in_train: TrainingConfigInTrain):
    config = parse_config_for_training(training_config_in_train)
    data_set_manager = DataSetManager(workspace_id, WorkspaceDataSource())
    trainer = Trainer(training_config=config, data_set_manager=data_set_manager, create_db=create_sync_db)
    Process(target=trainer.train).start()


def validate_config(training_config: TrainingConfigInTrain, workspace_sensors: Dict[str, Sensor]):
    if training_config.windowSize < 4:
        raise ValueError("Window size must be greater than or equal to 4")
    if training_config.slidingStep < 1:
        raise ValueError("Sliding step must be greater than or equal to 1")
    if training_config.slidingStep > training_config.windowSize:
        raise ValueError("Sliding step cannot excess the window size")

    # TODO IMPORTANT validate that feature extraction and pipeline configs are consistent (component names, number of components for each ?)
    for per_component_config in training_config.perComponentConfigs:
        sensor_name = per_component_config.sensor
        component_name = per_component_config.component
        if sensor_name not in workspace_sensors:
            raise ValueError(sensor_name + " is not a sensor of this workspace")
        if component_name not in workspace_sensors[sensor_name].components:
            raise ValueError(component_name + " not a valid component of " + sensor_name)

    validate_hyperparameters(training_config.classifier, training_config.hyperparameters)
