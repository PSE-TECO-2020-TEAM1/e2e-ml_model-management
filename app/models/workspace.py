from typing import Dict, List

from app.models.mongo_model import OID, MongoModel
from pydantic import Field


class Sensor(MongoModel):
    name: str
    samplingRate: int


class Timeframe(MongoModel):
    start: int
    end: int


class DataPoint(MongoModel):
    values: List[float]
    timestamp: int


class Sample(MongoModel):
    label: str
    timeframes: List[Timeframe] = Field(..., description="Valid intervals of the sample")
    sensor_data_points: Dict[str, List[DataPoint]]


class WorkspaceData(MongoModel):
    lastModified: int = Field(..., description="Unix Timestamp")
    label_to_int: Dict[str, int] = Field(..., description="label -> identifier_number")
    int_to_label: Dict[int, str] = Field(..., description="identifier_number -> label")
    samples: List[Sample]
    sliding_windows: Dict[str, OID] = Field([], description="window size_sliding step -> References to sliding windows")


class Workspace(MongoModel):
    id: OID
    user_id: OID
    prediction_ids: Dict[str, OID] = Field({}, description="predictionId -> modelId")
    sensors: List[Sensor]
    workspace_data: WorkspaceData = Field(None, description="Data samples of the workspace (must be fetched if None)")
    ml_models: List[OID] = Field([], description="References to the machine learning models of the workspace")
