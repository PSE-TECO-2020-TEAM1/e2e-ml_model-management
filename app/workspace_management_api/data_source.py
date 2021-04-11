from abc import ABC, abstractmethod
from typing import List
from bson.objectid import ObjectId
from app.workspace_management_api.sample_model import SampleFromWorkspace
from datetime import datetime, timedelta
from app.core.config import AUTH_SECRET
import jwt


class ExternalDataSource(ABC):

    @staticmethod
    @abstractmethod
    def last_modified(user_id: ObjectId, workspace_id: ObjectId) -> int:
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def fetch_samples(user_id: ObjectId, workspace_id: ObjectId) -> List[SampleFromWorkspace]:
        raise NotImplemented

    @staticmethod
    def get_auth_header(user_id):
        token = jwt.encode({"userId": str(user_id), "exp": datetime.utcnow() + timedelta(minutes=1)}, key=AUTH_SECRET, algorithm="HS256")
        return {"Authorization": "Bearer " + token}
