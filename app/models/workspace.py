from typing import Dict, List, Optional
from pydantic import Field

from app.models.mongo_model import OID, MongoModel

class Sensor(MongoModel):
    name: str
    samplingRate: int

class Timeframe(MongoModel):
    start: int
    end: int

class DataPoint(MongoModel):
    values: List[int]
    timestamp: int

class Sample(MongoModel):
    label: str
    timeframes: List[Timeframe] = Field(..., description="Valid intervals of the sample")
    sensorDataPoints: Dict[str, DataPoint]

class WorkspaceData(MongoModel):
    lastModified: int = Field(..., description = "Unix Timestamp")
    samples: List[Sample]
    slidingWindows: List[OID] = Field([], description="References to data split to windows")

class Workspace(MongoModel):
    _id: Optional[OID] = None
    userId: OID
    predictionIds: Dict[str, OID] = Field({}, description="predictionId: modelId")
    sensors: List[Sensor]
    workspaceData: WorkspaceData = Field(None, description="Data samples of the workspace (must be fetched if None)")
    ml_models: List[OID] = Field([], description="References to the machine learning models of the workspace")