from app.ml.objects.feature.enum import Feature
import pandas as pd
from pandas.core.frame import DataFrame
from app.ml.util.data_processing import calculate_classification_report, extract_features, roll_data_frame, split_to_data_windows
from tests.stubs.models.domain.feature_extraction_data import get_data_windows_df_4_2, get_data_windows_df_5_1, get_labels_of_data_windows_4_2, get_labels_of_data_windows_5_1, get_sensor_component_feature_dfs_4_2, get_sensor_component_feature_dfs_5_1
from app.models.domain.sliding_window import SlidingWindow
import pytest
from tests.stubs.models.domain.sample import get_interpolated_sample_stub_1, get_interpolated_sample_stub_2
from tests.stubs.models.domain.ml_model import get_label_encoder_stub_4_2, get_label_encoder_stub_5_1, get_label_performance_metrics_stub_4_2, get_label_performance_metrics_stub_5_1
from sklearn.model_selection import train_test_split


@pytest.fixture
def sliding_window_5_1():
    return SlidingWindow(window_size=5, sliding_step=1)


@pytest.fixture
def sliding_window_4_2():
    return SlidingWindow(window_size=4, sliding_step=2)


@pytest.fixture
def interpolated_sample_list():
    return [get_interpolated_sample_stub_1(), get_interpolated_sample_stub_2()]


@pytest.fixture
def data_windows_df_5_1():
    return get_data_windows_df_5_1()


@pytest.fixture
def data_windows_df_4_2():
    return get_data_windows_df_4_2()


@pytest.fixture
def labels_of_data_windows_5_1():
    return get_labels_of_data_windows_5_1()


@pytest.fixture
def labels_of_data_windows_4_2():
    return get_labels_of_data_windows_4_2()


@pytest.fixture
def sensor_component_feature_dfs_5_1():
    return get_sensor_component_feature_dfs_5_1()


@pytest.fixture
def sensor_component_feature_dfs_4_2():
    return get_sensor_component_feature_dfs_4_2()

@pytest.fixture
def label_encoder_5_1():
    return get_label_encoder_stub_5_1()

@pytest.fixture
def label_encoder_4_2():
    return get_label_encoder_stub_4_2()

@pytest.fixture
def label_performance_metrics_5_1():
    return get_label_performance_metrics_stub_5_1()

@pytest.fixture
def label_performance_metrics_4_2():
    return get_label_performance_metrics_stub_4_2()


def test_split_to_data_windows(sliding_window_5_1, sliding_window_4_2, interpolated_sample_list, data_windows_df_5_1, data_windows_df_4_2, labels_of_data_windows_5_1, labels_of_data_windows_4_2):
    result = split_to_data_windows(sliding_window_5_1, interpolated_sample_list)
    assert result[0].equals(data_windows_df_5_1)
    assert result[1] == labels_of_data_windows_5_1

    result = split_to_data_windows(sliding_window_4_2, interpolated_sample_list)
    assert result[0].equals(data_windows_df_4_2)
    assert result[1] == labels_of_data_windows_4_2


def test_roll_data_frame(sliding_window_5_1, sliding_window_4_2, interpolated_sample_list):
    sample_df = interpolated_sample_list[0].data_frame
    assert roll_data_frame(sliding_window_5_1, sample_df).equals(DataFrame([
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1}]))

    assert roll_data_frame(sliding_window_4_2, sample_df).equals(DataFrame([
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 0},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1}]))


def test_extract_features(data_windows_df_5_1, data_windows_df_4_2, sensor_component_feature_dfs_5_1, sensor_component_feature_dfs_4_2):
    result = extract_features(data_windows_df_5_1[["x_Accelerometer", "id"]], [Feature.MINIMUM, Feature.MAXIMUM])
    assert list(result.keys()) == [Feature.MINIMUM, Feature.MAXIMUM]
    assert result[Feature.MINIMUM].equals(sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MINIMUM])
    assert result[Feature.MAXIMUM].equals(sensor_component_feature_dfs_5_1["x_Accelerometer"][Feature.MAXIMUM])
    
    result = extract_features(data_windows_df_4_2[["z_Gyroscope", "id"]], [Feature.MEDIAN, Feature.MEAN])
    assert list(result.keys()) == [Feature.MEDIAN, Feature.MEAN]
    assert result[Feature.MEDIAN].equals(sensor_component_feature_dfs_4_2["z_Gyroscope"][Feature.MEDIAN])
    assert result[Feature.MEAN].equals(sensor_component_feature_dfs_4_2["z_Gyroscope"][Feature.MEAN])

# TODO
# def test_calculate_classification_report(labels_of_data_windows_5_1, labels_of_data_windows_4_2, label_encoder_5_1, label_encoder_4_2, label_performance_metrics_5_1, label_performance_metrics_4_2):
    