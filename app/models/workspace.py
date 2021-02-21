from typing import Dict, List, Optional

from app.models.mongo_model import OID, MongoModel
from pydantic import Field


class Sensor(MongoModel):
    name: str
    samplingRate: int


class Timeframe(MongoModel):
    start: int = Field(..., description="start index in the sample")
    end: int = Field(..., description="end index in the sample")


class DataPoint(MongoModel):
    values: List[float]
    #TODO include back timestamp: int


class Sample(MongoModel):
    label: str
    timeframes: List[Timeframe] = Field(..., description="Valid intervals of the sample")
    sensor_data_points: Dict[str, List[DataPoint]]

class WorkspaceData(MongoModel):
    #TODO include back last_modified: int = Field(..., description="Unix Timestamp")
    label_to_label_code: Dict[str, int] = Field(..., description="label -> identifier_number")
    label_code_to_label: Dict[int, str] = Field(..., description="identifier_number -> label")
    samples: List[OID]
    sliding_windows: Dict[str, OID] = Field([], description="window size_sliding step -> References to sliding windows")
    
class Workspace(MongoModel):
    id: Optional[OID]
    user_id: OID
    progress: int = Field(-1, description="progress of current training in percentage, -1 if there is no training in progress")
    prediction_ids: Dict[str, OID] = Field({}, description="predictionId -> modelId")
    #TODO include back amk sensors: List[Sensor]
    workspace_data: WorkspaceData = Field(None, description="Data samples of the workspace (must be fetched if None)")
    ml_models: List[OID] = Field([], description="References to the machine learning models of the workspace")
