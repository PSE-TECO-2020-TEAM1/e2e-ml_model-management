from app.models.domain.sensor import SensorComponent
from dataclasses import dataclass
from app.ml.training.parameters.classifiers import Classifier
from app.ml.training.parameters.normalizations import Normalization
from typing import Any, Dict, List
from app.ml.training.parameters.imputations import Imputation
from app.ml.training.parameters.features import Feature
from app.models.domain.sliding_window import SlidingWindow

@dataclass
class FeatureExtractionConfig():
    sliding_window: SlidingWindow
    sensor_component_features: Dict[SensorComponent, List[Feature]]

@dataclass
class PipelineConfig():
    imputation: Imputation
    normalization: Normalization

@dataclass
class TrainingConfig():
    model_name: str
    feature_extraction_config: FeatureExtractionConfig
    pipeline_config: Dict[SensorComponent, PipelineConfig]
    classifier: Classifier
    hyperparameters: Dict[str, Any]
