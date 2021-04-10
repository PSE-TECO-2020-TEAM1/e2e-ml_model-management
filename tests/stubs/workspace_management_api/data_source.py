from tests.stubs.workspace_management_api.sample_model import get_sample_from_workspace_stub_1, get_sample_from_workspace_stub_2
from tests.stubs.models.domain.workspace import get_workspace_stub

from app.workspace_management_api.sample_model import SampleFromWorkspace
from app.workspace_management_api.data_source import ExternalDataSource

from typing import List
from bson.objectid import ObjectId


class DataSourceStub(ExternalDataSource):
    @staticmethod
    def last_modified(workspace_id: ObjectId) -> int:
        if workspace_id is not get_workspace_stub()._id:
            # TODO raise error
            pass
        return 1617981582111

    @staticmethod
    def fetch_samples(workspace_id: ObjectId) -> List[SampleFromWorkspace]:
        if workspace_id is not get_workspace_stub()._id:
            # TODO raise error
            pass
        return [get_sample_from_workspace_stub_1(), get_sample_from_workspace_stub_2()]
