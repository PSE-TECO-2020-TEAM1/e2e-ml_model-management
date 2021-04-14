from tests.stubs.workspace_management_api.sample_model import get_sample_from_workspace_stub_1, get_sample_from_workspace_stub_2
from app.models.schemas.prediction_data import DataPointInPredict, DataPointsPerSensorInPredict
from app.ml.util.sample_parsing import parse_sample_from_workspace, parse_samples_from_workspace, parse_sensor_data_points_in_predict, split_data_by_timeframe, validate_sensor_data_points_in_predict
from tests.stubs.models.schemas.prediction_data import get_sample_in_predict_stub_1, get_sample_in_predict_stub_2
from tests.stubs.models.domain.workspace import get_workspace_stub
from tests.stubs.models.domain.sample import get_interpolated_sample_stub_1, get_interpolated_sample_stub_2, get_interpolated_sample_stub_3, get_interpolated_sample_stub_4, get_interpolated_sample_stub_5
import pytest
import re


@pytest.fixture
def sample_in_predict_1():
    return get_sample_in_predict_stub_1()


@pytest.fixture
def workspace_sensors():
    return get_workspace_stub().sensors


@pytest.fixture
def interpolated_sample_1():
    return get_interpolated_sample_stub_1()

@pytest.fixture
def sample_from_workspace_1():
    return get_sample_from_workspace_stub_1()

@pytest.fixture
def sample_from_workspace_2():
    return get_sample_from_workspace_stub_2()

@pytest.fixture
def parsed_interpolated_sample_list():
    return [get_interpolated_sample_stub_1(), get_interpolated_sample_stub_3(), get_interpolated_sample_stub_2,
            get_interpolated_sample_stub_4(), get_interpolated_sample_stub_5()]


# def test_validate_sensor_data_points_in_predict_with_valid_sample(sample_in_predict_1, workspace_sensors):
#     validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


# def test_validate_sensor_data_points_in_predict_start_larger_than_end(sample_in_predict_1, workspace_sensors):
#     sample_in_predict_1.start = sample_in_predict_1.end + 1
#     with pytest.raises(ValueError, match="Sample start cannot be bigger than the sample end"):
#         validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


# def test_validate_sensor_data_points_in_predict_missing_workspace_sensor(sample_in_predict_1, workspace_sensors):
#     missing_sensor = sample_in_predict_1.sensorDataPoints.pop().sensor
#     with pytest.raises(ValueError, match="Data from sensor not present in the sample: " + missing_sensor):
#         validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


# # def test_validate_sensor_data_points_in_predict_invalid_sensor(sample_in_predict_1, workspace_sensors):
# #     data_points_per_sensor_in_predict = DataPointsPerSensorInPredict(
# #         sensor="Magnetometer",
# #         dataPoints=[DataPointInPredict(
# #             data=[1.0, 1.0, 1.0],
# #             timestamp=(sample_in_predict_1.start + sample_in_predict_1.end) // 2
# #         )]
# #     )
# #     sample_in_predict_1.sensorDataPoints.append(data_points_per_sensor_in_predict)
# #     with pytest.raises(ValueError, match="Magnetometer"):  # TODO more specific error message
# #         validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


# def test_validate_sensor_data_points_in_predict_invalid_sensor_component_count(sample_in_predict_1, workspace_sensors):
#     sensor_name = sample_in_predict_1.sensorDataPoints[0].sensor
#     sample_in_predict_1.sensorDataPoints[0].dataPoints[0].data.append(1.0)
#     with pytest.raises(ValueError, match="Data for sensor " + sensor_name + " does not have the supported number of components."):
#         validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)

#     sample_in_predict_1.sensorDataPoints[0].dataPoints[0].data.pop()
#     sample_in_predict_1.sensorDataPoints[0].dataPoints[0].data.pop()
#     with pytest.raises(ValueError, match="Data for sensor " + sensor_name + " does not have the supported number of components."):
#         validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


# def test_validate_sensor_data_points_in_predict_invalid_timestamp(sample_in_predict_1, workspace_sensors):
#     sample_in_predict_1.sensorDataPoints[0].dataPoints[0].timestamp = sample_in_predict_1.start - 1
#     with pytest.raises(ValueError, match=re.escape("Data point has an invalid timestamp (outside of the sample timeframe)")):
#         validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)

#     sample_in_predict_1.sensorDataPoints[0].dataPoints[0].timestamp = sample_in_predict_1.end + 1
#     with pytest.raises(ValueError, match=re.escape("Data point has an invalid timestamp (outside of the sample timeframe)")):
#         validate_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors)


# def test_parse_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors, interpolated_sample_1):
#     # print(parse_sensor_data_points_in_predict(sample_in_predict_1, workspace_sensors))
#     # print(interpolated_sample_1.data_frame)
#     assert parse_sensor_data_points_in_predict(
#         sample_in_predict_1, workspace_sensors).equals(interpolated_sample_1.data_frame)

def test_parse_samples_from_workspace(sample_from_workspace_1, sample_from_workspace_2, workspace_sensors, parsed_interpolated_sample_list):
    # result_samples = parse_sample_from_workspace(sample_from_workspace_1, workspace_sensors)
    # assert len(result_samples) == 1
    # assert result_samples[0].label == parsed_interpolated_sample_list[0].label
    # print(result_samples[0].data_frame)
    # print(parsed_interpolated_sample_list[0].data_frame)
    # assert result_samples[0].data_frame.equals(parsed_interpolated_sample_list[0].data_frame)
    result_samples = parse_samples_from_workspace([sample_from_workspace_2], workspace_sensors)
    # assert len(result_samples) == len(parsed_interpolated_sample_list)

# def test_split_data_by_timeframe(sample_from_workspace_1):
#     result = split_data_by_timeframe(sample_from_workspace_1)
#     assert len(result) == len(sample_from_workspace_1.timeFrames)
#     for sensor_data_points_from_workspace in result.values():
#         assert len(sensor_data_points_from_workspace) == len(sample_from_workspace_1.sensorDataPoints)
#         for i in range(len(sample_from_workspace_1.sensorDataPoints)):
#             sensor_name = sample_from_workspace_1.sensorDataPoints[i].sensorName
#             assert sensor_name in sensor_data_points_from_workspace
#             print(sample_from_workspace_1.sensorDataPoints[i].dataPoints)
#             print(sensor_data_points_from_workspace[sensor_name])
#             print(sensor_name)
#             assert sample_from_workspace_1.sensorDataPoints[i].dataPoints == sensor_data_points_from_workspace[sensor_name]