from app.workspace_management_api.error import NoSampleInWorkspaceError, WorkspaceManagementConnectionError
from typing import List
from app.workspace_management_api.sample_model import SampleFromWorkspace
from app.workspace_management_api.data_source import ExternalDataSource
from app.core.config import WORKSPACE_MANAGEMENT_IP_PORT
from bson.objectid import ObjectId
import requests
import dacite

class WorkspaceDataSource(ExternalDataSource):

    @staticmethod
    def last_modified(user_id: ObjectId, workspace_id: ObjectId) -> int:
        url = WORKSPACE_MANAGEMENT_IP_PORT + "/api/workspaces/"+ str(workspace_id) +"/samples?onlyDate=true"
        try:
            last_modified = requests.get(url=url, headers=WorkspaceDataSource.get_auth_header(user_id)).json()
        except:
            raise WorkspaceManagementConnectionError("Could not connect to the model management server")
        return last_modified

    @staticmethod
    def fetch_samples(user_id: ObjectId, workspace_id: ObjectId) -> List[SampleFromWorkspace]:
        url = WORKSPACE_MANAGEMENT_IP_PORT + "/api/workspaces/" + str(workspace_id) + "/samples?showDataPoints=true"
        try:
            received_samples = requests.get(url=url, headers=WorkspaceDataSource.get_auth_header(user_id)).json()
        except:
            raise WorkspaceManagementConnectionError("Could not connect to the model management server")
        if not received_samples:
            raise NoSampleInWorkspaceError
        return [dacite.from_dict(data_class=SampleFromWorkspace, data=sample) for sample in received_samples]
