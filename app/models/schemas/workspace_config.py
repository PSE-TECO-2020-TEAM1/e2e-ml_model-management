from typing import List
from app.models.schemas.sensor import SensorInWorkspace
from app.models.schemas.mongo_model import MongoModel, OID

class WorkspaceConfig(MongoModel):
    workspaceId: OID
    sensors: List[SensorInWorkspace]