import numpy
from tests.stubs.models.domain.ml_model import get_column_order_stub_5_1, get_label_encoder_stub_5_1, get_ml_model_stub_5_1, get_pipeline_stub_5_1, get_training_config_stub_5_1, get_label_performance_metrics_stub_5_1
from app.ml.training.training_state import TrainingState
from app.ml.training.error.training_error import TrainingError
import pickle
from unittest import mock
import pytest

from app.db.error.non_existent_error import NonExistentError
from app.ml.objects.feature.enum import Feature
from app.ml.training.data_set_manager import DataSetManager
from app.models.domain.feature_extraction_data import FeatureExtractionData
from app.models.domain.sliding_window import SlidingWindow
from app.models.domain.training_data_set import TrainingDataSet
from app.models.domain.workspace import Workspace

from tests.stubs.db.syncdb.file_repository import FileRepositoryStub
from tests.stubs.db.syncdb.ml_model_repository import MlModelRepositoryStub
from tests.stubs.db.syncdb.workspace_repository import WorkspaceRepositoryStub
from tests.stubs.models.domain.feature_extraction_data import (
    get_data_windows_df_4_2, get_data_windows_df_5_1,
    get_labels_of_data_windows_4_2, get_labels_of_data_windows_5_1,
    get_sensor_component_feature_dfs_4_2, get_sensor_component_feature_dfs_5_1)
from tests.stubs.models.domain.sample import (get_interpolated_sample_stub_1,
                                              get_interpolated_sample_stub_2)
from tests.stubs.models.domain.workspace import get_workspace_stub
from tests.stubs.workspace_management_api.data_source import DataSourceStub

# TODO maybe create fixtures for sensor components and features for each sliding window ?

interpolated_sample_stubs = [get_interpolated_sample_stub_1(), get_interpolated_sample_stub_2()]


@pytest.fixture
def workspace_stub() -> Workspace:
    workspace = get_workspace_stub()
    workspace.trained_ml_model_refs = []
    return workspace


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
def sliding_window_4_2():
    return SlidingWindow(window_size=4, sliding_step=2)


@pytest.fixture
def sliding_window_5_1():
    return SlidingWindow(window_size=5, sliding_step=1)


@pytest.fixture
def training_config_5_1():
    return get_training_config_stub_5_1()


@pytest.fixture
def label_performance_metrics_stub_5_1():
    return get_label_performance_metrics_stub_5_1()


@pytest.fixture
def column_order_stub_5_1():
    return get_column_order_stub_5_1()


@pytest.fixture
def label_encoder_stub_5_1():
    return get_label_encoder_stub_5_1()


@pytest.fixture
def pipeline_stub_5_1():
    return get_pipeline_stub_5_1()


@pytest.fixture
def ml_model_stub_5_1():
    return get_ml_model_stub_5_1()


@pytest.fixture
def workspace_repository_stub(workspace_stub: Workspace):
    return WorkspaceRepositoryStub(init={workspace_stub._id: workspace_stub})


@pytest.fixture
def file_repository_stub(workspace_stub, interpolated_sample_stub_1, interpolated_sample_stub_2, data_windows_df_5_1, labels_of_data_windows_5_1, sensor_component_feature_dfs_5_1, sliding_window_5_1, data_windows_df_4_2, labels_of_data_windows_4_2, sensor_component_feature_dfs_4_2, sliding_window_4_2):
    training_data_set_stub: TrainingDataSet = workspace_stub.training_data_set
    feature_extraction_data_stub_5_1: FeatureExtractionData = training_data_set_stub.feature_extraction_cache[str(
        sliding_window_5_1)]
    feature_extraction_data_stub_4_2: FeatureExtractionData = training_data_set_stub.feature_extraction_cache[str(
        sliding_window_4_2)]

    init = {}

    init[training_data_set_stub.sample_list_file_ID] = pickle.dumps(
        [interpolated_sample_stub_1, interpolated_sample_stub_2])

    init[feature_extraction_data_stub_5_1.data_windows_df_file_ID] = pickle.dumps(data_windows_df_5_1)
    init[feature_extraction_data_stub_5_1.labels_of_data_windows_file_ID] = pickle.dumps(labels_of_data_windows_5_1)
    for sensor_component in feature_extraction_data_stub_5_1.sensor_component_feature_df_file_IDs:
        for feature in feature_extraction_data_stub_5_1.sensor_component_feature_df_file_IDs[sensor_component]:
            init[feature_extraction_data_stub_5_1.sensor_component_feature_df_file_IDs[sensor_component]
                 [feature]] = pickle.dumps(sensor_component_feature_dfs_5_1[sensor_component][feature])

    init[feature_extraction_data_stub_4_2.data_windows_df_file_ID] = pickle.dumps(data_windows_df_4_2)
    init[feature_extraction_data_stub_4_2.labels_of_data_windows_file_ID] = pickle.dumps(labels_of_data_windows_4_2)
    for sensor_component in feature_extraction_data_stub_4_2.sensor_component_feature_df_file_IDs:
        for feature in feature_extraction_data_stub_4_2.sensor_component_feature_df_file_IDs[sensor_component]:
            init[feature_extraction_data_stub_4_2.sensor_component_feature_df_file_IDs[sensor_component]
                 [feature]] = pickle.dumps(sensor_component_feature_dfs_4_2[sensor_component][feature])

    return FileRepositoryStub(init=init)


@pytest.fixture
def ml_model_repository_stub():
    init = {}
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
    assert not workspace_stub.training_data_set.feature_extraction_cache

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


def test_is_cached_split_to_windows(data_set_manager, sliding_window_5_1, sliding_window_4_2):
    assert data_set_manager.is_cached_split_to_windows(sliding_window_5_1)
    assert data_set_manager.is_cached_split_to_windows(sliding_window_4_2)
    assert not data_set_manager.is_cached_split_to_windows(SlidingWindow(window_size=5, sliding_step=2))


def test_get_labels_of_data_windows(data_set_manager, labels_of_data_windows_4_2, labels_of_data_windows_5_1, sliding_window_4_2, sliding_window_5_1):
    assert data_set_manager.get_labels_of_data_windows(sliding_window_4_2) == labels_of_data_windows_4_2
    assert data_set_manager.get_labels_of_data_windows(sliding_window_5_1) == labels_of_data_windows_5_1
    with pytest.raises(TrainingError):
        data_set_manager.get_labels_of_data_windows(SlidingWindow(window_size=4, sliding_step=1))


def test_get_cached_split_to_windows(data_set_manager, data_windows_df_4_2, data_windows_df_5_1, sliding_window_4_2, sliding_window_5_1):
    assert data_set_manager.get_cached_split_to_windows(sliding_window_4_2).equals(data_windows_df_4_2)
    assert data_set_manager.get_cached_split_to_windows(sliding_window_5_1).equals(data_windows_df_5_1)
    with pytest.raises(TrainingError):
        data_set_manager.get_cached_split_to_windows(SlidingWindow(window_size=8, sliding_step=6))


def test_add_split_to_windows(data_set_manager, workspace_stub, file_repository_stub, data_windows_df_4_2, labels_of_data_windows_4_2, sliding_window_4_2, data_windows_df_5_1, labels_of_data_windows_5_1, sliding_window_5_1):
    feature_extraction_data_stub_4_2 = workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_4_2)]
    for file_id in feature_extraction_data_stub_4_2.get_all_file_IDs():
        file_repository_stub.delete_file(file_id)
    workspace_stub.training_data_set.feature_extraction_cache.pop(str(sliding_window_4_2))

    data_set_manager.add_split_to_windows(sliding_window_4_2, data_windows_df_4_2, labels_of_data_windows_4_2)

    assert str(sliding_window_4_2) in workspace_stub.training_data_set.feature_extraction_cache

    feature_extraction_data_stub_4_2 = workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_4_2)]
    assert not feature_extraction_data_stub_4_2.sensor_component_feature_df_file_IDs
    assert pickle.loads(file_repository_stub.get_file(
        feature_extraction_data_stub_4_2.data_windows_df_file_ID)).equals(data_windows_df_4_2)
    assert pickle.loads(file_repository_stub.get_file(
        feature_extraction_data_stub_4_2.labels_of_data_windows_file_ID)) == labels_of_data_windows_4_2

    assert str(sliding_window_5_1) in workspace_stub.training_data_set.feature_extraction_cache

    feature_extraction_data_stub_5_1 = workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)]
    assert feature_extraction_data_stub_5_1.sensor_component_feature_df_file_IDs
    assert pickle.loads(file_repository_stub.get_file(
        feature_extraction_data_stub_5_1.data_windows_df_file_ID)).equals(data_windows_df_5_1)
    assert pickle.loads(file_repository_stub.get_file(
        feature_extraction_data_stub_5_1.labels_of_data_windows_file_ID)) == labels_of_data_windows_5_1


def test_add_split_to_windows_with_already_existing_split_to_windows(data_set_manager, data_windows_df_4_2, labels_of_data_windows_4_2, sliding_window_4_2):
    with pytest.raises(AssertionError):
        data_set_manager.add_split_to_windows(sliding_window_4_2, data_windows_df_4_2, labels_of_data_windows_4_2)


def test_is_cached_sensor_component_feature(data_set_manager, sliding_window_5_1, sliding_window_4_2):
    assert data_set_manager.is_cached_sensor_component_feature(sliding_window_5_1, "x_Accelerometer", Feature.MINIMUM)
    assert data_set_manager.is_cached_sensor_component_feature(sliding_window_5_1, "y_Accelerometer", Feature.MAXIMUM)
    assert data_set_manager.is_cached_sensor_component_feature(sliding_window_4_2, "z_Accelerometer", Feature.MEAN)
    assert data_set_manager.is_cached_sensor_component_feature(sliding_window_4_2, "z_Gyroscope", Feature.MEDIAN)

    assert not data_set_manager.is_cached_sensor_component_feature(
        SlidingWindow(window_size=5, sliding_step=2), "x_Accelerometer", Feature.MINIMUM)
    assert not data_set_manager.is_cached_sensor_component_feature(
        sliding_window_5_1, "a_Accelerometer", Feature.MINIMUM)
    assert not data_set_manager.is_cached_sensor_component_feature(sliding_window_5_1, "x_Accelerometer", Feature.MEAN)


def test_get_cached_sensor_component_feature(data_set_manager, sensor_component_feature_dfs_5_1, sliding_window_5_1, sensor_component_feature_dfs_4_2, sliding_window_4_2):
    assert data_set_manager.get_cached_sensor_component_feature(sliding_window_5_1, "x_Accelerometer", Feature.MINIMUM).equals(
        sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MINIMUM])
    assert data_set_manager.get_cached_sensor_component_feature(sliding_window_5_1, "y_Accelerometer", Feature.MAXIMUM).equals(
        sensor_component_feature_dfs_5_1["y_Accelerometer"][Feature.MAXIMUM])
    assert data_set_manager.get_cached_sensor_component_feature(sliding_window_4_2, "z_Accelerometer", Feature.MEAN).equals(
        sensor_component_feature_dfs_4_2["z_Accelerometer"][Feature.MEAN])
    assert data_set_manager.get_cached_sensor_component_feature(sliding_window_4_2, "z_Gyroscope", Feature.MEDIAN).equals(
        sensor_component_feature_dfs_4_2["z_Gyroscope"][Feature.MEDIAN])

    with pytest.raises(TrainingError):
        data_set_manager.get_cached_sensor_component_feature(SlidingWindow(
            window_size=5, sliding_step=2), "x_Accelerometer", Feature.MINIMUM)

    with pytest.raises(TrainingError):
        data_set_manager.get_cached_sensor_component_feature(sliding_window_5_1, "a_Accelerometer", Feature.MINIMUM)

    with pytest.raises(TrainingError):
        data_set_manager.get_cached_sensor_component_feature(sliding_window_5_1, "x_Accelerometer", Feature.MEAN)


def test_add_sensor_component_feature_with_invalid_sliding_window(data_set_manager, sensor_component_feature_dfs_5_1):
    with pytest.raises(AssertionError):
        data_set_manager.add_sensor_component_feature(SlidingWindow(
            window_size=4, sliding_step=1), "x_Accelerometer", Feature.MAXIMUM, sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MAXIMUM])


def test_add_sensor_component_feature_with_new_sensor_component(data_set_manager, file_repository_stub, workspace_stub, sensor_component_feature_dfs_5_1, sliding_window_5_1, sensor_component_feature_dfs_4_2, sliding_window_4_2):
    # In this case, the sensor component is not a key in sensor_component_feature_df_file_IDs
    file_repository_stub.delete_file(
        workspace_stub.training_data_set.feature_extraction_cache[str(sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"][Feature.MINIMUM])
    file_repository_stub.delete_file(workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"][Feature.MAXIMUM])
    workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs.pop("x_Accelerometer")

    data_set_manager.add_sensor_component_feature(
        sliding_window_5_1, "x_Accelerometer", Feature.MINIMUM, sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MINIMUM])

    assert "x_Accelerometer" in workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs
    assert Feature.MINIMUM in workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"]

    new_sensor_component_feature_5_1_file_id = workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"][Feature.MINIMUM]

    assert pickle.loads(file_repository_stub.get_file(new_sensor_component_feature_5_1_file_id)).equals(
        sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MINIMUM])

    # In this case, the sensor component is a key in sensor_component_feature_df_file_IDs but it maps to an empty dictionary

    file_repository_stub.delete_file(
        workspace_stub.training_data_set.feature_extraction_cache[str(sliding_window_4_2)].sensor_component_feature_df_file_IDs["y_Gyroscope"][Feature.MEAN])
    file_repository_stub.delete_file(workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_4_2)].sensor_component_feature_df_file_IDs["y_Gyroscope"][Feature.MEDIAN])
    workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_4_2)].sensor_component_feature_df_file_IDs["y_Gyroscope"] = {}

    data_set_manager.add_sensor_component_feature(
        sliding_window_4_2, "y_Gyroscope", Feature.MEAN, sensor_component_feature_dfs_4_2["y_Gyroscope"][Feature.MEAN])

    assert "y_Gyroscope" in workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_4_2)].sensor_component_feature_df_file_IDs
    assert Feature.MEAN in workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_4_2)].sensor_component_feature_df_file_IDs["y_Gyroscope"]

    new_sensor_component_feature_4_2_file_id = workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_4_2)].sensor_component_feature_df_file_IDs["y_Gyroscope"][Feature.MEAN]

    assert pickle.loads(file_repository_stub.get_file(new_sensor_component_feature_4_2_file_id)).equals(
        sensor_component_feature_dfs_4_2["y_Gyroscope"][Feature.MEAN])


def test_add_sensor_component_feature_with_existing_sensor_component_and_new_feature(data_set_manager, file_repository_stub, workspace_stub, sliding_window_5_1, sensor_component_feature_dfs_5_1):
    file_repository_stub.delete_file(
        workspace_stub.training_data_set.feature_extraction_cache[str(sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"][Feature.MINIMUM])
    workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"].pop(Feature.MINIMUM)

    data_set_manager.add_sensor_component_feature(
        sliding_window_5_1, "x_Accelerometer", Feature.MINIMUM, sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MINIMUM])

    assert "x_Accelerometer" in workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs
    assert Feature.MINIMUM in workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"]

    new_sensor_component_feature_5_1_file_id = workspace_stub.training_data_set.feature_extraction_cache[str(
        sliding_window_5_1)].sensor_component_feature_df_file_IDs["x_Accelerometer"][Feature.MINIMUM]

    assert pickle.loads(file_repository_stub.get_file(new_sensor_component_feature_5_1_file_id)).equals(
        sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MINIMUM])


def test_add_sensor_component_feature_with_existing_sensor_component_and_existing_feature(data_set_manager, sliding_window_5_1, sensor_component_feature_dfs_5_1):
    with pytest.raises(AssertionError):
        data_set_manager.add_sensor_component_feature(
            sliding_window_5_1, "x_Accelerometer", Feature.MINIMUM, sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MINIMUM])


def test_set_training_state(data_set_manager, workspace_stub):
    data_set_manager.set_training_state(TrainingState.NO_ACTIVE_TRAINING)
    assert workspace_stub.training_state == TrainingState.NO_ACTIVE_TRAINING

    data_set_manager.set_training_state(TrainingState.FEATURE_EXTRACTION)
    assert workspace_stub.training_state == TrainingState.FEATURE_EXTRACTION

    data_set_manager.set_training_state(TrainingState.TRAINING_INITIATED)
    assert workspace_stub.training_state == TrainingState.TRAINING_INITIATED


def test_save_mode(data_set_manager, ml_model_repository_stub, file_repository_stub, workspace_stub, training_config_5_1, label_performance_metrics_stub_5_1, column_order_stub_5_1, label_encoder_stub_5_1, pipeline_stub_5_1, labels_of_data_windows_5_1):
    data_set_manager.save_model(training_config_5_1, label_performance_metrics_stub_5_1,
                                column_order_stub_5_1, label_encoder_stub_5_1, pipeline_stub_5_1)

    assert len(workspace_stub.trained_ml_model_refs) == 1

    ml_model_id = workspace_stub.trained_ml_model_refs[0]
    ml_model = ml_model_repository_stub.get_ml_model(ml_model_id)

    assert ml_model._id == ml_model_id
    assert ml_model.config == training_config_5_1
    assert ml_model.label_performance_metrics == label_performance_metrics_stub_5_1
    assert ml_model.column_order == column_order_stub_5_1

    assert numpy.array_equal(pickle.loads(file_repository_stub.get_file(
        ml_model.label_encoder_object_file_ID)).classes_, label_encoder_stub_5_1.classes_)
    # compare pipeline in file repository stub and pipeline_stub_5_1
