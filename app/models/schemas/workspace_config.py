from typing import List
from app.models.schemas.sensor import SensorInWorkspace
from pydantic import BaseModel
from app.models.schemas.mongo_model import OID

class WorkspaceConfig(BaseModel):
    workspaceId: OID
    sensors: List[SensorInWorkspace]