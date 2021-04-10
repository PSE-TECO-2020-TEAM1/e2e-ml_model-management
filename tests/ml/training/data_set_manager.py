from tests.stubs.db.syncdb.file_repository import FileRepositoryStub
from tests.stubs.db.syncdb.workspace_repository import WorkspaceRepositoryStub
from tests.stubs.db.syncdb.ml_model_repository import MlModelRepositoryStub
from tests.stubs.workspace_management_api.data_source import DataSourceStub
from tests.stubs.models.domain.workspace import workspace_stub

from app.ml.training.data_set_manager import DataSetManager

import pytest
from unittest import mock

@pytest.fixture
def data_set_manager():
    data_set_manager = DataSetManager(workspace_stub._id, DataSourceStub())
    data_set_manager.db_is_set = True
    data_set_manager.file_repository = FileRepositoryStub(None)
    data_set_manager.workspace_repository = WorkspaceRepositoryStub(None)
    data_set_manager.ml_model_repository = MlModelRepositoryStub(None)
    return data_set_manager

@mock.patch.object(DataSourceStub, "fetch_samples")
def test_update_training_data_set_no_update(mock, data_set_manager: DataSetManager):
    data_set_manager.update_training_data_set()
    mock.assert_not_called()