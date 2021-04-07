from app.models.domain.sensor import SensorComponent
from dataclasses import dataclass
from typing import Any, Dict, List
from app.ml.objects.classification import Classifier
from app.ml.objects.normalization import Normalization
from app.ml.objects.imputation import Imputation
from app.ml.objects.feature import Feature
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
