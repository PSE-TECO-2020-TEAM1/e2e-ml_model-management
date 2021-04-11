from app.ml.training.training_state import TrainingState
from app.models.domain.training_data_set import TrainingDataSet

from app.models.domain.workspace import Workspace
from app.db.error.non_existent_error import NonExistentError

from typing import Dict
from bson.objectid import ObjectId

class WorkspaceRepositoryStub():
    def __init__(self, init: Dict[ObjectId, Workspace]):
        self.workspaces = init

    def contains_workspace_id(self, workspace_id: ObjectId) -> Workspace:
        return workspace_id in self.workspaces

    def get_workspace(self, workspace_id: ObjectId) -> Workspace:
        if workspace_id not in self.workspaces:
            raise NonExistentError("Workspace with the given id does not exist")
        return self.workspaces[workspace_id]

    def set_training_state(self, workspace_id: ObjectId, training_state: TrainingState):
        self.workspaces[workspace_id].training_state = training_state

    def get_training_data_set(self, workspace_id: ObjectId) -> TrainingDataSet:
        self.get_workspace(workspace_id)
        return self.workspaces[workspace_id].training_data_set

    def set_training_data_set(self, workspace_id: ObjectId, training_data_set: TrainingDataSet):
        self.get_workspace(workspace_id)
        self.workspaces[workspace_id].training_data_set = training_data_set

    def add_ml_model_ref(self, workspace_id: ObjectId, ml_model_id: ObjectId):
        self.get_workspace(workspace_id)
        self.workspaces[workspace_id].trained_ml_model_refs.append(ml_model_id)
