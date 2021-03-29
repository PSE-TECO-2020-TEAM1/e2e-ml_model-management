from app.ml.util.data_preprocessing import split_data_to_windows
from typing import List

from app.models.domain.training_data_set import Sample, TrainingDataSet
from app.db.error.non_existant_error import NonExistentError
from app.models.domain.mongo_model import OID
from app.models.domain.workspace import Workspace
from gridfs import GridFS
from pymongo.database import Database
import datetime


class WorkspaceRepository():
    def __init__(self, db: Database):
        self.collection = db["workspaces"]
        self.fs = GridFS(db)

    def get_workspace_by_id(self, workspace_id: OID) -> Workspace:
        workspace = self.collection.find_one({"_id": workspace_id})
        if workspace is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return Workspace(**workspace)

    def get_training_data_set(self, workspace_id: OID):
        self.collection.find_one({"_id": workspace_id}, {"training_data_set": True})

    def set_training_data_set(self, workspace_id: OID, new_data: List[Sample]):
        old_training_data_set = self.collection.find_one({"_id": workspace_id}, {"training_data_set": True})
        if old_training_data_set is None:
            raise NonExistentError("Workspace with the given id does not exist")
        old_training_data_set = TrainingDataSet(**old_training_data_set)
        # Delete the old data saved in the file system if there is data at all
        if old_training_data_set.last_modified != 0:
            to_be_deleted_IDs = old_training_data_set.get_all_file_IDs()
            for id in to_be_deleted_IDs:
                self.fs.delete(id)
        serialized = TrainingDataSet.serialize(new_data)
        file_id = self.fs.put(serialized)
        new_training_data_set = TrainingDataSet(last_modified=datetime.utcnow(), sample_list_file_ID=file_id)
        self.collection.update_one({"_id": workspace_id}, {"$set": {"training_data_set": new_training_data_set}})
