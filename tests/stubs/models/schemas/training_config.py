from app.ml.objects.classification.enum import Classifier
from app.models.schemas.training_config import HyperparameterInResponse, PerComponentConfigInTrain, TrainingConfigInResponse, TrainingConfigInTrain
from tests.stubs.models.domain.ml_model import get_training_config_stub_5_1
from app.ml.objects.classification.classifier_config_spaces.util import config_spaces

training_config_stub_5_1 = get_training_config_stub_5_1()


def get_training_config_in_train_stub_5_1():
    return TrainingConfigInTrain(
        modelName=training_config_stub_5_1.model_name,
        windowSize=training_config_stub_5_1.sliding_window.window_size,
        slidingStep=training_config_stub_5_1.sliding_window.sliding_step,
        perComponentConfigs=[PerComponentConfigInTrain(
            sensor=sensor_component[2:],
            component=sensor_component[0],
            features=per_component_configs.features,
            imputation=per_component_configs.pipeline_config.imputation,
            normalizer=per_component_configs.pipeline_config.normalization
        ) for sensor_component, per_component_configs in training_config_stub_5_1.perComponentConfigs.items()],
        classifier=training_config_stub_5_1.classifier,
        hyperparameters=training_config_stub_5_1.hyperparameters
    )

def get_training_config_in_response_stub_5_1():
    return TrainingConfigInResponse(
        modelName=training_config_stub_5_1.model_name,
        windowSize=training_config_stub_5_1.sliding_window.window_size,
        slidingStep=training_config_stub_5_1.sliding_window.sliding_step,
        perComponentConfigs=[PerComponentConfigInTrain(
            sensor=sensor_component[2:],
            component=sensor_component[:1],
            features=per_component_configs.features,
            imputation=per_component_configs.pipeline_config.imputation,
            normalizer=per_component_configs.pipeline_config.normalization
        ) for sensor_component, per_component_configs in training_config_stub_5_1.perComponentConfigs.items()],
        classifier=training_config_stub_5_1.classifier,
        hyperparameters=[HyperparameterInResponse(name=name, value=value) for name, value in training_config_stub_5_1.hyperparameters.items()]
    )