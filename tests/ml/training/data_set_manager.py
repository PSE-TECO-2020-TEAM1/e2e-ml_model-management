from app.db.error.non_existent_error import NonExistentError
from app.models.domain.feature_extraction_data import FeatureExtractionData
from app.models.domain.training_data_set import TrainingDataSet
from tests.stubs.models.domain.feature_extraction_data import get_data_windows_df_4_2, get_data_windows_df_5_1, get_feature_extraction_data_stub_4_2, get_feature_extraction_data_stub_5_1, get_labels_of_data_windows_4_2, get_labels_of_data_windows_5_1, get_sensor_component_feature_dfs_4_2, get_sensor_component_feature_dfs_5_1
from app.models.domain.workspace import Workspace
from tests.stubs.db.syncdb.file_repository import FileRepositoryStub
from tests.stubs.db.syncdb.workspace_repository import WorkspaceRepositoryStub
from tests.stubs.db.syncdb.ml_model_repository import MlModelRepositoryStub
from tests.stubs.workspace_management_api.data_source import DataSourceStub
from tests.stubs.models.domain.workspace import get_workspace_stub
from tests.stubs.models.domain.sample import get_interpolated_sample_stub_1, get_interpolated_sample_stub_2

from app.ml.training.data_set_manager import DataSetManager

import pickle
import pytest
from unittest import mock

interpolated_sample_stubs = [get_interpolated_sample_stub_1(), get_interpolated_sample_stub_2()]


@pytest.fixture
def workspace_stub() -> Workspace:
    return get_workspace_stub()


@pytest.fixture
def interpolated_sample_stub_1():
    return get_interpolated_sample_stub_1()


@pytest.fixture
def interpolated_sample_stub_2():
    return get_interpolated_sample_stub_2()

@pytest.fixture
def interpolated_sample_stubs_fixture(interpolated_sample_stub_1, interpolated_sample_stub_2):
    return [interpolated_sample_stub_1, interpolated_sample_stub_2]


@pytest.fixture
def data_windows_df_5_1():
    return get_data_windows_df_5_1()


@pytest.fixture
def labels_of_data_windows_5_1():
    return get_labels_of_data_windows_5_1()


@pytest.fixture
def sensor_component_feature_dfs_5_1():
    return get_sensor_component_feature_dfs_5_1()


@pytest.fixture
def data_windows_df_4_2():
    return get_data_windows_df_4_2()


@pytest.fixture
def labels_of_data_windows_4_2():
    return get_labels_of_data_windows_4_2()


@pytest.fixture
def sensor_component_feature_dfs_4_2():
    return get_sensor_component_feature_dfs_4_2()


@pytest.fixture
def workspace_repository_stub(workspace_stub: Workspace):
    return WorkspaceRepositoryStub(init={workspace_stub._id: workspace_stub})


@pytest.fixture
def file_repository_stub(workspace_stub, interpolated_sample_stub_1, interpolated_sample_stub_2, data_windows_df_5_1, labels_of_data_windows_5_1, sensor_component_feature_dfs_5_1, data_windows_df_4_2, labels_of_data_windows_4_2, sensor_component_feature_dfs_4_2):
    training_data_set_stub: TrainingDataSet = workspace_stub.training_data_set
    feature_extraction_data_stub_5_1: FeatureExtractionData = training_data_set_stub.feature_extraction_cache["5_1"]
    feature_extraction_data_stub_4_2: FeatureExtractionData = training_data_set_stub.feature_extraction_cache["4_2"]

    init = {}

    init[training_data_set_stub.sample_list_file_ID] = pickle.dumps(
        [interpolated_sample_stub_1, interpolated_sample_stub_2])

    init[feature_extraction_data_stub_5_1.data_windows_df_file_ID] = pickle.dumps(data_windows_df_5_1)
    init[feature_extraction_data_stub_5_1.labels_of_data_windows_file_ID] = pickle.dumps(labels_of_data_windows_5_1)
    for sensor_component in feature_extraction_data_stub_5_1.sensor_component_feature_df_file_IDs:
        for feature in feature_extraction_data_stub_5_1.sensor_component_feature_df_file_IDs[sensor_component]:
            init[feature_extraction_data_stub_5_1.sensor_component_feature_df_file_IDs[sensor_component]
                 [feature]] = sensor_component_feature_dfs_5_1[sensor_component][feature]

    init[feature_extraction_data_stub_4_2.data_windows_df_file_ID] = pickle.dumps(data_windows_df_4_2)
    init[feature_extraction_data_stub_4_2.labels_of_data_windows_file_ID] = pickle.dumps(labels_of_data_windows_4_2)
    for sensor_component in feature_extraction_data_stub_4_2.sensor_component_feature_df_file_IDs:
        for feature in feature_extraction_data_stub_4_2.sensor_component_feature_df_file_IDs[sensor_component]:
            init[feature_extraction_data_stub_4_2.sensor_component_feature_df_file_IDs[sensor_component]
                 [feature]] = sensor_component_feature_dfs_4_2[sensor_component][feature]

    return FileRepositoryStub(init=init)


@pytest.fixture
def ml_model_repository_stub():
    init = {} # TODO
    return MlModelRepositoryStub(init=init)


@pytest.fixture
def data_set_manager(file_repository_stub, workspace_repository_stub, ml_model_repository_stub):
    data_set_manager = DataSetManager(get_workspace_stub()._id, DataSourceStub())

    data_set_manager.db_is_set = True
    data_set_manager.file_repository = file_repository_stub
    data_set_manager.workspace_repository = workspace_repository_stub
    data_set_manager.ml_model_repository = ml_model_repository_stub

    return data_set_manager


@mock.patch.object(DataSourceStub, "fetch_samples")
def test_update_training_data_set_no_update(mock, data_set_manager, file_repository_stub, workspace_stub):
    initial_file_count_in_repository = len(file_repository_stub.files)
    data_set_manager.update_training_data_set()
    mock.assert_not_called()
    assert workspace_stub.training_data_set.last_modified == DataSourceStub.last_modified(workspace_stub._id)
    assert len(file_repository_stub.files) == initial_file_count_in_repository


@mock.patch("app.ml.training.data_set_manager.parse_samples_from_workspace", return_value=interpolated_sample_stubs)
def test_update_training_data_set_with_update(mock, data_set_manager, file_repository_stub, workspace_stub, interpolated_sample_stubs_fixture):
    workspace_stub.training_data_set.last_modified = 1617981582110
    data_set_manager.update_training_data_set()
    mock.assert_called()
    assert workspace_stub.training_data_set.last_modified == DataSourceStub.last_modified(workspace_stub._id)
    assert len(file_repository_stub.files) == 1

    files_in_repository = pickle.loads(file_repository_stub.get_file(
        workspace_stub.training_data_set.sample_list_file_ID))
    assert len(files_in_repository) == len(interpolated_sample_stubs_fixture)
    for i in range(len(files_in_repository)):
        assert files_in_repository[i].label == interpolated_sample_stubs_fixture[i].label
        assert files_in_repository[i].data_frame.equals(interpolated_sample_stubs_fixture[i].data_frame)


def test_get_sample_list(data_set_manager, interpolated_sample_stubs_fixture):
    sample_list = data_set_manager.get_sample_list()
    assert len(sample_list) == len(interpolated_sample_stubs_fixture)
    for i in range(len(sample_list)):
        assert sample_list[i].label == interpolated_sample_stubs_fixture[i].label
        assert sample_list[i].data_frame.equals(interpolated_sample_stubs_fixture[i].data_frame)

def test_get_sample_list_with_no_sample_list(data_set_manager, workspace_stub):
    workspace_stub.training_data_set.sample_list_file_ID = None
    with pytest.raises(NonExistentError):
        data_set_manager.get_sample_list()
