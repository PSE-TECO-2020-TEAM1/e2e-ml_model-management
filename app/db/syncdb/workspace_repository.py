from enum import Enum
from app.ml.objects.feature.enum import Feature
from app.models.domain.sensor import SensorComponent
from dataclasses import asdict
from typing import Any, Dict
from bson.objectid import ObjectId
from app.models.domain.training_data_set import TrainingDataSet
from app.db.error.non_existent_error import NonExistentError
from app.models.domain.workspace import Workspace
from pymongo.database import Database
from app.core.config import WORKSPACE_COLLECTION_NAME
import dacite

class WorkspaceRepository():
    def __init__(self, db: Database):
        self.collection = db[WORKSPACE_COLLECTION_NAME]

    def get_workspace(self, workspace_id: ObjectId) -> Workspace:
        workspace = self.collection.find_one({"_id": workspace_id})
        if workspace is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return dacite.from_dict(data_class=Workspace, data=workspace, config=dacite.Config(cast=[SensorComponent, Enum]))

    def set_workspace_field(self, workspace_id: ObjectId, field: str, value: Any):
        """
        field with dot notation
        """
        result = self.collection.update_one({"_id": workspace_id}, {"$set": {field: value}})
        if not result:
            raise NonExistentError("Could not set " + field)

    def get_training_data_set(self, workspace_id: ObjectId) -> TrainingDataSet:
        workspace = self.get_workspace(workspace_id)
        return workspace.training_data_set

    def set_training_data_set(self, workspace_id: ObjectId, training_data_set: TrainingDataSet):
        self.set_workspace_field(workspace_id, "training_data_set", asdict(training_data_set))
        
    def add_ml_model_ref(self, workspace_id: ObjectId, ml_model_id: ObjectId):
        result = self.collection.update_one({"_id": workspace_id}, {"$push": {"trained_ml_model_refs": ml_model_id}})
        if not result:
            raise NonExistentError("Could not set ML model reference")