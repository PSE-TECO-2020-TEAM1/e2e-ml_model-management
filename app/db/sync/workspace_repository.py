from typing import Any
from bson.objectid import ObjectId
from app.models.domain.training_data_set import TrainingDataSet
from app.db.error.non_existant_error import NonExistentError
from app.models.domain.workspace import Workspace
from pymongo.database import Database


class WorkspaceRepository():
    def __init__(self, db: Database):
        self.collection = db["workspaces"]

    def get_workspace(self, workspace_id: ObjectId) -> Workspace:
        workspace = self.collection.find_one({"_id": workspace_id})
        if workspace is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return Workspace(**workspace)

    def set_workspace_field(self, workspace_id: ObjectId, field: str, value: Any):
        result = self.collection.update_one({"_id": workspace_id}, {"$set": {field: value}})
        if not result:
            raise NonExistentError("Could not set " + field)

    def get_training_data_set(self, workspace_id: ObjectId) -> TrainingDataSet:
        workspace = self.get_workspace(workspace_id)
        return workspace.training_data_set

    def set_training_data_set(self, workspace_id: ObjectId, training_data_set: TrainingDataSet):
        self.set_workspace_field(workspace_id, "training_data_set", training_data_set)
        