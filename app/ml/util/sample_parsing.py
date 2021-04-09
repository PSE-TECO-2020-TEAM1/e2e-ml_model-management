from app.workspace_management_api.sample_model import DataPoint, SampleFromWorkspace, Timeframe
from app.models.domain.sample import InterpolatedSample
import bisect
from typing import Dict, List
from pandas.core.frame import DataFrame
from app.models.domain.sensor import Sensor


def parse_samples_from_workspace(samples: List[SampleFromWorkspace], workspace_sensors: Dict[str, Sensor]) -> List[InterpolatedSample]:
    all_interpolated_samples = []
    for sample in samples:
        all_interpolated_samples += parse_sample_from_workspace(sample, workspace_sensors)
        # As negative data, also save the data points out of the selected time frames
        invert_timeframes(sample)
        all_interpolated_samples += parse_sample_from_workspace(sample, workspace_sensors)
    return all_interpolated_samples


def invert_timeframes(sample: SampleFromWorkspace):
    timeframes = sample.timeFrames
    inverted_timeframes: List[Timeframe] = []
    # Add the beginning and the end
    if timeframes[0].start - sample.start > 0:
        inverted_timeframes.append(Timeframe(start=sample.start, end=timeframes[0].start))
    if sample.end - timeframes[-1].end > 0:
        inverted_timeframes.append(Timeframe(start=timeframes[-1].end, end=sample.end))
    # Handle rest
    for i in range(len(timeframes) - 1):
        between_two = Timeframe(start=timeframes[i].end, end=timeframes[i+1].start)
        inverted_timeframes.append(between_two)
    sample.timeFrames = inverted_timeframes


def parse_sample_from_workspace(sample: SampleFromWorkspace, workspace_sensors: Dict[str, Sensor]) -> List[InterpolatedSample]:
    """
    We treat each timeframe of a sample as a new sample internally, so this method also returns a list of interpolated samples
    """
    data_by_timeframe = split_data_by_timeframe(sample)  # Timeframe -> (Sensor name -> Data points in this timeframe)
    interpolated_timeframe_data_frames = []
    delta = delta_of_sensors(list(workspace_sensors.values()))  # Number of data points per second
    for timeframe, timeframe_data in data_by_timeframe.items():
        data_frame_data: Dict[Sensor, List[List[float]]] = {}
        for sensor_name, sensor_data in timeframe_data.items():
            sensor = workspace_sensors[sensor_name]
            data_frame_data[sensor_name] = interpolate_sensor_data_points_in_timeframe(
                sensor_data, sensor, timeframe, delta)
        interpolated_timeframe_data_frames.append(build_dataframe(data_frame_data, workspace_sensors))
    return [InterpolatedSample(label=sample.label, data_frame=df) for df in interpolated_timeframe_data_frames]


def split_data_by_timeframe(sample: SampleFromWorkspace) -> Dict[Timeframe, Dict[str, List[DataPoint]]]:
    # Build key index for binary search
    sensor_timestamps = {}  # Sensor name -> List of all timestamps
    for i in range(len(sample.sensorDataPoints)):
        sensor_name = sample.sensorDataPoints[i].sensor
        datapoints: List[DataPoint] = sample.sensorDataPoints[i].dataPoints
        sensor_timestamps[sensor_name] = [datapoint.timestamp for datapoint in datapoints]

    data_by_timeframe = {}  # Timeframe -> (Sensor name -> Data points in this timeframe)
    for timeframe in sample.timeFrames:
        data_in_timeframe = {}  # Sensor name -> Data points in this timeframe
        for j in range(len(sample.sensorDataPoints)):
            sensor_name = sample.sensorDataPoints[j].sensor
            left = bisect.bisect_right(sensor_timestamps[sensor_name], timeframe.start)
            right = bisect.bisect_right(sensor_timestamps[sensor_name], timeframe.end)
            data_in_timeframe[sensor_name] = sample.sensorDataPoints[j].dataPoints[left:right]
        data_by_timeframe[timeframe] = data_in_timeframe
    return data_by_timeframe

# Interpolate data points from each sensor so that they all match the max.


def delta_of_sensors(sensors: List[Sensor]) -> float:
    max: int = -1
    for sensor in sensors:
        if sensor.sampling_rate > max:
            max = sensor.sampling_rate
    return 1000 / max


def build_dataframe(data_frame_data: Dict[str, List[List[float]]], workspace_sensors: Dict[str, Sensor]) -> DataFrame:
    data_point_count = len(next(iter(data_frame_data.values())))
    # Sanity check (each data point list per sensor must be of the same size)
    for data in data_frame_data.values():
        assert len(data) == data_point_count
    rows = [{} for __range__ in range(data_point_count)]
    for sensor_name, data in data_frame_data.items():
        sensor = workspace_sensors[sensor_name]
        for i in range(data_point_count):
            for component_index in range(len(sensor.components)):
                component = sensor.components[component_index]
                rows[i][component] = data[i][component_index]
    return DataFrame(rows)


def interpolate_sensor_data_points_in_timeframe(data_points: List[DataPoint], sensor: Sensor, timeframe: Timeframe, delta: float) -> List[List[float]]:
    target_len = (timeframe.end - timeframe.start) // delta

    # If there are no datapoints in the selected timeframe, we decided to give default values for that timeframe. (default: 0)
    if not data_points:
        return [[0 for __range__ in range(len(sensor.components))] for __range__ in range(target_len)]

    # If start and end timestamps of sample are not aligned with the given timeframe, add a default data point to the start and end of the sample.
    # (default: first and last data point of the sample)
    # This is necessary for the interpolation as a start value is needed to interpolate the beginning of the sample (and for the end)
    data_points.insert(0, DataPoint(data=data_points[0].data, timestamp=timeframe.start))
    data_points.append(DataPoint(data=data_points[-1].data, timestamp=timeframe.end))

    # Normalize the timestamps by removing the offset from each
    for datapoint in data_points:
        datapoint.timestamp -= timeframe.start

    result = []
    hi = 0
    for i in range(target_len):
        while i * delta >= data_points[hi].timestamp:
            hi += 1
        interpolation_percentage = (i * delta - data_points[hi - 1].timestamp) / \
            (data_points[hi].timestamp - data_points[hi - 1].timestamp)
        interpolated_datapoint = []
        for component_index in range(len(sensor.components)):
            difference = data_points[hi].data[component_index] - data_points[hi - 1].data[component_index]
            interpolated_value = data_points[hi - 1].data[component_index] + difference * interpolation_percentage
            interpolated_datapoint.append(interpolated_value)
        result.append(interpolated_datapoint)
    return result
