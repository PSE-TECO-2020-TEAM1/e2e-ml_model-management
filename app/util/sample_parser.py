import bisect
from typing import Dict, List
from pandas.core.frame import DataFrame
from app.models.workspace import DataPoint, SampleInJson, Sensor, Timeframe


class SampleParser():
    def __init__(self, sensors: List[Sensor]):
        self.sensors = sensors
        self.delta = 1000 / max(sensor.samplingRate for sensor in sensors)

    def parse_sample(self, sample: SampleInJson) -> List[DataFrame]:
        data_by_timeframe = self.split_data_by_timeframe(sample)
        dataframes_of_timeframes: List[DataFrame] = []
        for timeframe in data_by_timeframe.keys():
            dataframes_of_timeframes.append(self.parse_timeframe(data_by_timeframe[timeframe]))
        return dataframes_of_timeframes

    def split_data_by_timeframe(self, sample: SampleInJson) -> Dict[Timeframe, Dict[str, List[DataPoint]]]:
        # Build key index for binary search
        sensor_timestamps = {}
        for sensor in self.sensors:
            datapoints: List[DataPoint] = sample.sensorDataPoints[sensor.name]
            sensor_timestamps[sensor.name] = [datapoint.timestamp for datapoint in datapoints]

        data_by_timeframe = {}
        for timeframe in sample.timeframes:
            data_in_timeframe = {}
            for sensor in self.sensors:
                left = bisect.bisect_left(sensor_timestamps[sensor.name], timeframe.start)
                right = bisect.bisect_left(sensor_timestamps[sensor.name], timeframe.end)
                data_in_timeframe[sensor.name] = sample.sensorDataPoints[sensor.name][left:right]
            data_by_timeframe[timeframe] = data_in_timeframe
        return data_by_timeframe

    def parse_timeframe(self, timeframe: Timeframe, sensor_data_points: Dict[str, List[DataPoint]]) -> DataFrame:
        dataframe_data: List[Dict[str, float]] = []
        ideal_datapoint_count = (timeframe.end - timeframe.start) / self.delta
        for sensor in self.sensors:
            # Normalize the timestamps
            datapoints = sensor_data_points[sensor.name]
            for datapoint in datapoints:
                datapoint.timestamp -= timeframe.start
            interpolated = self.interpolate_sensor_datapoints(datapoints, len(sensor.dataFormat), ideal_datapoint_count)
            for datapoint_index in range(ideal_datapoint_count):
                for format_index in range(len(sensor.dataFormat)):
                    row = dataframe_data[datapoint_index]
                    row[sensor.name + "_" + sensor.dataFormat[format_index]] = interpolated[datapoint_index][format_index]
        return DataFrame(dataframe_data)

    def interpolate_sensor_datapoints(self, datapoints: List[DataPoint], sensor_format_len: int, target_len: int) -> List[List[float]]:
        result: List[List[float]] = []
        hi = 0
        i = 0
        while i * self.delta <= datapoints[0].timestamp:
            result.append(datapoints[0].data)
            i += 1
        for i in range(i, target_len):
            if i * self.delta >= datapoints[-1].timestamp:
                result.append(datapoints[-1].data)
                continue
            while i * self.delta > datapoints[hi].timestamp:
                hi += 1
            interpolation_percentage = (i * self.delta - datapoints[hi - 1]) / (datapoints[hi] - datapoints[hi - 1])
            interpolated_datapoint = []
            for format_index in range(sensor_format_len):
                difference = datapoints[hi].data[format_index] - datapoints[hi - 1].data[format_index]
                interpolated_value = datapoints[hi - 1].data[format_index] + difference * interpolation_percentage
                interpolated_datapoint.append(interpolated_value)
            result.append(interpolated_datapoint)
        return result
