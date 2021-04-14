from app.models.schemas.prediction_data import DataPointInPredict, DataPointsPerSensorInPredict
from app.ml.util.sample_parsing import parse_sensor_data_points_in_predict, validate_sensor_data_points_in_predict
from tests.stubs.models.schemas.prediction_data import get_sample_in_predict_stub_1
from tests.stubs.models.domain.workspace import get_workspace_stub
from tests.stubs.models.domain.sample import get_interpolated_sample_stub_1
import pytest
import re


@pytest.fixture
def sample_in_predict_1():
    return get_sample_in_predict_stub_1()


@pytest.fixture
def workspace_sensors():
    return get_workspace_stub().sensors

@pytest.fixture
def interpolated_sample_stub_1():
    return get_interpolated_sample_stub_1()


def test_validate_sensor_data_points_in_predict_with_valid_sample(sample_in_predict_1, workspace_sensors):
    validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


def test_validate_sensor_data_points_in_predict_start_larger_than_end(sample_in_predict_1, workspace_sensors):
    sample_in_predict_1.start = sample_in_predict_1.end + 1
    with pytest.raises(ValueError, match="Sample start cannot be bigger than the sample end"):
        validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


def test_validate_sensor_data_points_in_predict_missing_workspace_sensor(sample_in_predict_1, workspace_sensors):
    missing_sensor = sample_in_predict_1.sensorDataPoints.pop().sensor
    with pytest.raises(ValueError, match="Data from sensor not present in the sample: " + missing_sensor):
        validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


def test_validate_sensor_data_points_in_predict_invalid_sensor(sample_in_predict_1, workspace_sensors):
    data_points_per_sensor_in_predict = DataPointsPerSensorInPredict(
        sensor="Magnetometer",
        dataPoints=[DataPointInPredict(
            data=[1.0, 1.0, 1.0],
            timestamp=(sample_in_predict_1.start + sample_in_predict_1.end) // 2
        )]
    )
    sample_in_predict_1.sensorDataPoints.append(data_points_per_sensor_in_predict)

    with pytest.raises(ValueError, match="Magnetometer"):  # TODO more specific error message
        validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


def test_validate_sensor_data_points_in_predict_invalid_sensor_component_count(sample_in_predict_1, workspace_sensors):
    sensor_name = sample_in_predict_1.sensorDataPoints[0].sensor
    sample_in_predict_1.sensorDataPoints[0].dataPoints[0].data.append(1.0)
    with pytest.raises(ValueError, match="Data for sensor " + sensor_name + " does not have the supported number of components."):
        validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)

    sample_in_predict_1.sensorDataPoints[0].dataPoints[0].data.pop()
    sample_in_predict_1.sensorDataPoints[0].dataPoints[0].data.pop()
    with pytest.raises(ValueError, match="Data for sensor " + sensor_name + " does not have the supported number of components."):
        validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)

def test_validate_sensor_data_points_in_predict_invalid_timestamp(sample_in_predict_1, workspace_sensors):
    sample_in_predict_1.sensorDataPoints[0].dataPoints[0].timestamp = sample_in_predict_1.start - 1
    with pytest.raises(ValueError, match=re.escape("Data point has an invalid timestamp (outside of the sample timeframe)")):
        validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)
    
    sample_in_predict_1.sensorDataPoints[0].dataPoints[0].timestamp = sample_in_predict_1.end + 1
    with pytest.raises(ValueError, match=re.escape("Data point has an invalid timestamp (outside of the sample timeframe)")):
        validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)

def test_parse_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors, interpolated_sample_stub_1):
    print(parse_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors))
    print(interpolated_sample_stub_1.data_frame)
    assert parse_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors).equals(interpolated_sample_stub_1.data_frame)
