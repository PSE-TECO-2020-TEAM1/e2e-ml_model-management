from tests.stubs.db.syncdb.file_repository import FileRepositoryStub
from tests.stubs.db.syncdb.ml_model_repository import MlModelRepositoryStub
from tests.stubs.db.syncdb.workspace_repository import WorkspaceRepositoryStub
from tests.stubs.workspace_management_api.data_source import DataSourceStub
from tests.stubs.models.domain.workspace import get_workspace_stub

class DataSetManagerStub():
    def __init__(self, workspace_id, external_data_source):
        pass