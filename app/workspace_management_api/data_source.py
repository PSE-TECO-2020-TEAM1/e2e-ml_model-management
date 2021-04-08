from abc import ABC
from typing import List
from bson.objectid import ObjectId
from app.workspace_management_api.sample_model import SampleFromWorkspace

class ExternalDataSource(ABC):

    @staticmethod
    def last_modified(workspace_id: ObjectId) -> int:
        pass

    @staticmethod
    def fetch_samples(workspace_id: ObjectId) -> List[SampleFromWorkspace]:
        pass