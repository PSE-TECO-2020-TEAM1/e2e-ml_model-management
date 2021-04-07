from typing import Any, Dict, List
from pydantic.main import BaseModel
from app.ml.objects.classification import Classifier
from app.ml.objects.normalization import Normalization
from app.ml.objects.imputation import Imputation
from app.ml.objects.feature import Feature

class FeatureExtractionConfigPerSensorComponent(BaseModel):
    sensor: str
    component: str
    features: List[Feature]

class PipelineConfigPerSensorComponent(BaseModel):
    sensor: str
    component: str
    imputation: Imputation
    normalization: Normalization

class TrainingConfigInTrain(BaseModel):
    modelName: str
    windowSize: int
    slidingStep: int
    featureExtractionConfig: List[FeatureExtractionConfigPerSensorComponent]
    pipelineConfig: List[PipelineConfigPerSensorComponent]
    classifier: Classifier
    hyperparameters: Dict[str, Any]