from multiprocessing import process
from multiprocessing.context import Process
from app.db.sync import create_sync_db
from app.models.domain.sliding_window import SlidingWindow
from app.workspace_management_api.stub_source_delete import WorkspaceDataSource
from app.ml.training.trainer import Trainer
from bson.objectid import ObjectId
from app.ml.training.data_set_manager import DataSetManager
from app.ml.objects.feature.enum import Feature
from app.models.domain.sensor import Sensor, SensorComponent, make_sensor_component
from typing import Dict, List
from app.models.domain.training_config import FeatureExtractionConfig, PipelineConfig, TrainingConfig
from app.ml.objects.classification.classifier_config_spaces.util import validate_hyperparameters
from app.models.schemas.training_config import FeatureExtractionConfigPerSensorComponent, PipelineConfigPerSensorComponent, TrainingConfigInTrain


def initiate_new_training(workspace_id: ObjectId, training_config_in_train: TrainingConfigInTrain):
    config = parse_config(training_config_in_train)
    data_set_manager = DataSetManager(workspace_id, WorkspaceDataSource())
    trainer = Trainer(training_config=config, data_set_manager=data_set_manager, create_db=create_sync_db)
    import time
    start = time.time()
    Process(target=trainer.train).start()
    print(time.time() - start)


def validate_config(training_config: TrainingConfigInTrain, workspace_sensors: Dict[str, Sensor]):
    if training_config.windowSize < 4:
        raise ValueError("Window size must be greater than or equal to 4")
    if training_config.slidingStep < 1:
        raise ValueError("Sliding step must be greater than or equal to 1")
    if training_config.slidingStep > training_config.windowSize:
        raise ValueError("Sliding step cannot excess the window size")

    # TODO IMPORTANT validate that feature extraction and pipeline configs are consistent (component names, number of components for each ?)

    validate_hyperparameters(training_config.classifier, training_config.hyperparameters)


def parse_config(training_config: TrainingConfigInTrain) -> TrainingConfig:
    params = {}
    params["model_name"] = training_config.modelName
    sliding_window = SlidingWindow(training_config.windowSize, training_config.slidingStep)
    parsed = parse_feature_extraction_config(training_config.featureExtractionConfig)
    params["feature_extraction_config"] = FeatureExtractionConfig(sliding_window, parsed)
    params["pipeline_config"] = parse_pipeline_config(training_config.pipelineConfig)
    params["classifier"] = training_config.classifier
    params["hyperparameters"] = training_config.hyperparameters
    return TrainingConfig(**params)


def parse_feature_extraction_config(feature_extraction_config: List[FeatureExtractionConfigPerSensorComponent]) -> Dict[SensorComponent, List[Feature]]:
    res = {}
    for item in feature_extraction_config:
        features = item.features
        sensor_component = make_sensor_component(item.sensor, item.component)
        res[sensor_component] = features
    return res


def parse_pipeline_config(pipeline_config: List[PipelineConfigPerSensorComponent]) -> Dict[SensorComponent, PipelineConfig]:
    res = {}
    for item in pipeline_config:
        sensor_component = make_sensor_component(item.sensor, item.component)
        res[sensor_component] = PipelineConfig(imputation=item.imputation, normalization=item.normalization)
    return res
