from typing import List
from pydantic import BaseModel

class SensorInWorkspace(BaseModel):
    name: str
    samplingRate: int
    dataFormat: List[str]