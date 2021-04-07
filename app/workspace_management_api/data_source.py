from abc import ABC, abstractstaticmethod
from typing import List
from bson.objectid import ObjectId
from app.workspace_management_api.sample_model import SampleFromWorkspace

class ExternalDataSource(ABC):

    @abstractstaticmethod
    def last_modified(workspace_id: ObjectId) -> int:
        pass

    @abstractstaticmethod
    def fetch_samples(workspace_id: ObjectId) -> List[SampleFromWorkspace]:
        pass