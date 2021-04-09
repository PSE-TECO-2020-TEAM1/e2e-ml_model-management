from app.models.domain.prediction_id import PredictionID
from sklearn.pipeline import Pipeline
from app.models.domain.performance_metrics import PerformanceMetrics
from app.models.domain.training_config import TrainingConfig
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder
from bson.objectid import ObjectId
import pickle

@dataclass(frozen=True)
class MlModel():
    _id: Optional[ObjectId]
    config: TrainingConfig
    label_performance_metrics: List[PerformanceMetrics]
    column_order: List[str] # order of features e.g x_Accelerometer__min
    label_encoder_object_file_ID: ObjectId 
    pipeline_object_file_ID: ObjectId
    prediction_IDs: List[PredictionID] = field(default_factory=list)

    @staticmethod
    def serialize_label_encoder(label_encoder: LabelEncoder) -> bytes:
        return pickle.dumps(label_encoder)

    @staticmethod
    def deserialize_label_encoder(label_encoder: bytes) -> LabelEncoder:
        return pickle.loads(label_encoder)

    @staticmethod
    def serialize_pipeline(pipeline: Pipeline) -> bytes:
        return pickle.dumps(pipeline)

    @staticmethod
    def deserialize_pipeline(pipeline: bytes) -> Pipeline:
        return pickle.loads(pipeline)