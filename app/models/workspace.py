from typing import Dict, List

from app.models.mongo_model import OID, MongoModel
from pydantic import Field


class Sensor(MongoModel):
    name: str
    samplingRate: int
    dataFormat: List[str]


class Timeframe(MongoModel):
    start: int = Field(..., description="start index in the sample")
    end: int = Field(..., description="end index in the sample")

class DataPoint(MongoModel):
    data: List[float]
    timestamp: int

class SampleInJson(MongoModel):
    """
    The only difference from the Sample class is the sensorDataPoints field and the id. We receive the sample over network as
    normal dictionary, but save the sample in database after the validation as pickled dictionary. 
    """
    id: OID
    label: str
    timeframes: List[Timeframe] = Field(..., description="Valid intervals of the sample")
    sensorDataPoints: Dict[str, List[DataPoint]] = Field(..., description="Dictionary that maps sensors to their data points")
    

class Sample(MongoModel):
    label: str
    sensorDataPoints: OID = Field(...,
                                  description="Reference to pickled dictionary that maps sensors to their data points")


class WorkspaceData(MongoModel):
    # TODO add this: last_modified: int = Field(..., description="Unix Timestamp")
    labelToLabelCode: Dict[str, str] = Field(..., description="label -> identifier_number")
    labelCodeToLabel: Dict[str, str] = Field(..., description="identifier_number -> label")
    samples: List[Sample] = Field([], description="samples are set during the training")
    slidingWindows: Dict[str, OID] = Field([], description="window size_sliding step -> References to sliding windows")


class Workspace(MongoModel):
    id: OID = Field(None, alias="_id")
    userId: OID
    progress: int = Field(-1, description="progress of current training in percentage, -1 if there is no training in progress")
    predictionIds: Dict[str, OID] = Field({}, description="predictionId -> modelId")
    sensors: List[Sensor]
    workspaceData: WorkspaceData = Field(None, description="Data samples of the workspace (must be fetched if None)")
    mlModels: List[OID] = Field([], description="References to the machine learning models of the workspace")
