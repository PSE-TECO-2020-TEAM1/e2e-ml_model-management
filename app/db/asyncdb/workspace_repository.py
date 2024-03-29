from app.ml.training.training_state import TrainingState
from enum import Enum
from typing import Any
from app.models.domain.sensor import SensorComponent
from app.core.config import WORKSPACE_COLLECTION_NAME
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from bson.objectid import ObjectId
from app.models.domain.workspace import Workspace
from app.db.error.non_existent_error import NonExistentError
import dacite


class WorkspaceRepository():
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection: AsyncIOMotorCollection = db[WORKSPACE_COLLECTION_NAME]

    async def get_workspace(self, workspace_id: ObjectId) -> Workspace:
        workspace = await self.collection.find_one({"_id": workspace_id})
        if workspace is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return dacite.from_dict(data_class=Workspace, data=workspace, config=dacite.Config(cast=[SensorComponent, Enum]))

    async def workspace_exists(self, workspace_id: ObjectId) -> bool:
        return bool(await self.collection.find_one({"_id": workspace_id}))

    async def add_workspace(self, workspace: Workspace) -> ObjectId:
        workspace_dict = workspace.dict_for_db_insertion()
        result = await self.collection.insert_one(workspace_dict)
        if not result:
            raise NonExistentError("Could not add workspace because of a database error")
        return result.inserted_id

    async def set_workspace_field(self, workspace_id: ObjectId, field: str, value: Any):
        """
        field with dot notation
        """
        result = await self.collection.update_one({"_id": workspace_id}, {"$set": {field: value}})
        if not result:
            raise NonExistentError("Could not set " + field)

    async def set_training_state(self, workspace_id: ObjectId, training_state: TrainingState):
        await self.set_workspace_field(workspace_id, "training_state", training_state)

    async def delete_ml_model_ref(self, workspace_id: ObjectId, ml_model_ref: ObjectId):
        await self.collection.update_one({"_id": workspace_id}, {"$pull": {"ml_model_refs": ml_model_ref}})
