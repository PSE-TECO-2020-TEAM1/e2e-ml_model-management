from app.models.domain.sensor import SensorComponent
from dataclasses import dataclass
from typing import Any, Dict, List
from app.ml.objects.classification import Classifier
from app.ml.objects.normalization import Normalization
from app.ml.objects.imputation import Imputation
from app.ml.objects.feature import Feature
from app.models.domain.sliding_window import SlidingWindow

@dataclass
class PipelineConfig():
    imputation: Imputation
    normalization: Normalization

@dataclass
class PerComponentConfig():
    features: List[Feature]
    pipeline_config: PipelineConfig

@dataclass
class TrainingConfig():
    model_name: str
    sliding_window: SlidingWindow
    perComponentConfigs: Dict[SensorComponent, PerComponentConfig]
    classifier: Classifier
    hyperparameters: Dict[str, Any]

    def get_component_features(self) -> Dict[SensorComponent, List[Feature]]:
        features = {}
        for sensor_component, config in self.perComponentConfigs.items():
            features[sensor_component] = config.features
        return features

    def get_component_pipeline_configs(self) -> Dict[SensorComponent, PipelineConfig]:
        pipeline_configs = {}
        for sensor_component, config in self.perComponentConfigs.items():
            pipeline_configs[sensor_component] = config.pipeline_config
        return pipeline_configs