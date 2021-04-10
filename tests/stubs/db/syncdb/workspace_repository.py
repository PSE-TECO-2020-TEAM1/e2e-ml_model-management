from app.models.domain.training_data_set import TrainingDataSet
from tests.stubs.models.domain.workspace import workspace_stub

from app.models.domain.workspace import Workspace

from typing import Any, Dict
from bson.objectid import ObjectId

class WorkspaceRepositoryStub():
    def __init__(self, db):
        self.workspaces: Dict[ObjectId, Workspace] = {}
        self.__insert_stubs__()

    def __insert_stubs__(self):
        self.workspaces[workspace_stub._id] = workspace_stub

    def get_workspace(self, workspace_id: ObjectId) -> Workspace:
        self.__assert_contains_workspace_id__(workspace_id)
        return self.workspaces[workspace_id]

    def get_training_data_set(self, workspace_id: ObjectId) -> TrainingDataSet:
        self.__assert_contains_workspace_id__(workspace_id)
        return self.workspaces[workspace_id].training_data_set

    def set_training_data_set(self, workspace_id: ObjectId, training_data_set: TrainingDataSet):
        self.__assert_contains_workspace_id__(workspace_id)
        self.workspaces[workspace_id].training_data_set = training_data_set

    def add_ml_model_ref(self, workspace_id: ObjectId, ml_model_id: ObjectId):
        self.__assert_contains_workspace_id__(workspace_id)
        self.workspaces[workspace_id].trained_ml_model_refs.append(ml_model_id)

    def __assert_contains_workspace_id__(self, workspace_id: ObjectId):
        if workspace_id not in self.workspaces:
            # TODO raise error
            pass