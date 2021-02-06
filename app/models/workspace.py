from typing import Dict, List, Optional
from pydantic import Field

from app.models.mongo_model import OID, MongoModel

class Sensor(MongoModel):
    name: str
    samplingRate: int

class Sample(MongoModel):
    label: str
    #TODO left here
    allSensorDataPoints: List[]

class WorkspaceData(MongoModel):
    lastModified: int = Field(..., description = "Unix Timestamp")
    data: List[Sample]

class Workspace(MongoModel):
    _id: Optional[OID] = None
    userId: int
    predictionIds: Dict[str, OID] = Field({}, description="predictionId: modelId")
    sensors: List[Sensor]
    ml_models: List[OID] = Field([], description="references to the machine learning models of the workspace")