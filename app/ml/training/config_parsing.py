from app.models.domain.sensor import make_sensor_component, reverse_sensor_component
from app.models.domain.sliding_window import SlidingWindow
from app.models.domain.training_config import PerComponentConfig, PipelineConfig, TrainingConfig
from app.models.schemas.training_config import HyperparameterInResponse, PerComponentConfigInTrain, TrainingConfigInResponse, TrainingConfigInTrain


def parse_config_for_training(training_config: TrainingConfigInTrain) -> TrainingConfig:
    return TrainingConfig(
        model_name=training_config.modelName,
        sliding_window=SlidingWindow(training_config.windowSize, training_config.slidingStep),
        perComponentConfigs={make_sensor_component(config.sensor, config.component): PerComponentConfig(
            features=config.features,
            pipeline_config=PipelineConfig(
                imputation=config.imputation,
                normalization=config.normalizer
            )
        ) for config in training_config.perComponentConfigs},
        classifier=training_config.classifier,
        hyperparameters=training_config.hyperparameters
    )


def parse_config_for_response(training_config: TrainingConfig) -> TrainingConfigInResponse:
    return TrainingConfigInResponse(modelName=training_config.model_name,
                                    windowSize=training_config.sliding_window.window_size,
                                    slidingStep=training_config.sliding_window.sliding_step,
                                    perComponentConfigs=[PerComponentConfigInTrain(
                                        sensor=reverse_sensor_component(component)[0],
                                        component=reverse_sensor_component(component)[1],
                                        features=config.features,
                                        imputation=config.pipeline_config.imputation,
                                        normalizer=config.pipeline_config.normalization
                                    )
                                        for component, config in training_config.perComponentConfigs.items()],
                                    classifier=training_config.classifier,
                                    hyperparameters=[HyperparameterInResponse(name=name,
                                                                              value=value)
                                                     for name, value in training_config.hyperparameters.items()])
