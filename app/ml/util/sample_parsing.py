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
    split_data_by_timeframe = split_data_by_timeframe(sample)
    interpolated_timeframe_data_frames = []
    delta = delta_of_sensors(workspace_sensors)
    for timeframe, timeframe_data in split_data_by_timeframe.items():
        data_frame_data: Dict[Sensor, List[List[float]]] = {}
        for sensor, sensor_data in timeframe_data.items():
            workspace_sensor = workspace_sensors[sensor]
            data_frame_data[workspace_sensor] = interpolate_sensor_data_points_in_timeframe(sensor_data, workspace_sensor, timeframe, delta)
        interpolated_timeframe_data_frames.append(build_dataframe(data_frame_data))
    return [InterpolatedSample(label=sample.label, data_frame=df) for df in interpolated_timeframe_data_frames]

# Interpolate data points from each sensor so that they all match the max
def delta_of_sensors(sensors: List[Sensor]):
    max: int = -1
    for sensor in sensors:
        if sensor.sampling_rate > max:
            max = sensor.sampling_rate
    return 1000 // max

def build_dataframe(data_frame_data: Dict[Sensor, List[List[float]]]) -> DataFrame:
    data_point_count = len(next(iter(data_frame_data.values())))
    # Sanity check (each data point list per sensor must be of the same size)
    for data in data_frame_data.values():
        assert len(data) == data_point_count
    rows = [{} for __range__ in range(data_point_count)]
    for sensor, data in data_frame_data.items():
        for i in range(data_point_count):
            for component_index in range(len(sensor.components)):
                component = sensor.components[component_index]
                rows[i][component] = data[i][component_index]
    return DataFrame(rows)
 
def split_data_by_timeframe(sample: SampleFromWorkspace) -> Dict[Timeframe, Dict[str, List[DataPoint]]]:
    # Build key index for binary search
    sensor_timestamps = {}
    for i in range(len(sample.sensorDataPoints)):
        sensorName = sample.sensorDataPoints[i].sensor
        datapoints: List[DataPoint] = sample.sensorDataPoints[i].dataPoints
        sensor_timestamps[sensorName] = [datapoint.timestamp for datapoint in datapoints]

    data_by_timeframe = {}
    for i in range(len(sample.timeFrames)):
        timeframe = sample.timeFrames[i]
        data_in_timeframe = {}
        for j in range(len(sample.sensorDataPoints)):
            sensorName = sample.sensorDataPoints[j].sensor
            left = bisect.bisect_left(sensor_timestamps[sensorName], timeframe.start)
            right = bisect.bisect_left(sensor_timestamps[sensorName], timeframe.end)
            data_in_timeframe[sensorName] += sample.sensorDataPoints[j].dataPoints[left:right]
        data_by_timeframe[timeframe] = data_in_timeframe
    return data_by_timeframe

def interpolate_sensor_data_points_in_timeframe(data_points: List[DataPoint], sensor: Sensor, timeframe: Timeframe, delta: int) -> List[List[float]]:
    target_len = (timeframe.end - timeframe.start) // delta

    # If there are no datapoints in the selected timeframe, we decided to give default values for that timeframe. (default: 0)
    if not data_points:
        return [[0 for __range__ in range(len(sensor.components))] for __range__ in range(target_len)]

    # We add these two data points to interpolate the section between the first sample and the start of the timeframe (analog for the end)
    data_points.insert(0, DataPoint(data=data_points[0].data, timestamp=timeframe.start))
    data_points.append(DataPoint(data=data_points[-1].data, timestamp=timeframe.end))
    
    # Normalize the timestamps by removing the offset from each
    for datapoint in data_points:
        datapoint.timestamp -= timeframe.start

    result = []
    hi = 0
    for i in range(target_len):
        while i * delta > data_points[hi].timestamp:
            hi += 1
        interpolation_percentage = (i * delta - data_points[hi - 1].timestamp) / (data_points[hi].timestamp - data_points[hi - 1].timestamp)
        interpolated_datapoint = []
        for component_index in range(len(sensor.components)):
            difference = data_points[hi].data[component_index] - data_points[hi - 1].data[component_index]
            interpolated_value = data_points[hi - 1].data[component_index] + difference * interpolation_percentage
            interpolated_datapoint.append(interpolated_value)
        result.append(interpolated_datapoint)
    return result
    
    
    
    