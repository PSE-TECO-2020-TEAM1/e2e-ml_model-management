from typing import Any
from app.models.domain.prediction_id import PredictionID
from app.ml.objects.feature.enum import Feature
from app.models.domain.sensor import SensorComponent
from app.core.config import WORKSPACE_COLLECTION_NAME
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson.objectid import ObjectId
from app.models.domain.workspace import Workspace
from app.db.error.non_existent_error import NonExistentError
import dacite


class WorkspaceRepository():
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[WORKSPACE_COLLECTION_NAME]

    async def get_workspace(self, workspace_id: ObjectId) -> Workspace:
        workspace = await self.collection.find_one({"_id": workspace_id})
        if workspace is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return dacite.from_dict(data_class=Workspace, data=workspace, config=dacite.Config(cast=[SensorComponent, Feature]))

    async def set_workspace_field(self, workspace_id: ObjectId, field: str, value: Any):
        """
        field with dot notation
        """
        result = await self.collection.update_one({"_id": workspace_id}, {"$set": {field: value}})
        if not result:
            raise NonExistentError("Could not set " + field)