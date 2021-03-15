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
        for timeframe in data_by_timeframe:
            print(timeframe)
            dataframe_of_timeframe = self.parse_timeframe(sample.timeFrames[timeframe], data_by_timeframe[timeframe])
            dataframes_of_timeframes.append(dataframe_of_timeframe)
        return dataframes_of_timeframes

    def split_data_by_timeframe(self, sample: SampleInJson) -> Dict[Timeframe, Dict[str, List[DataPoint]]]:
        # Build key index for binary search
        sensor_timestamps = {}
        for i in range(len(sample.sensorDataPoints)):
            sensorName = sample.sensorDataPoints[i].sensorName
            datapoints: List[DataPoint] = sample.sensorDataPoints[i].dataPoints
            sensor_timestamps[sensorName] = [datapoint.timestamp for datapoint in datapoints]

        data_by_timeframe = {}
        for i in range(len(sample.timeFrames)):
            timeframe = sample.timeFrames[i]
            data_in_timeframe = {}
            for j in range(len(sample.sensorDataPoints)):
                sensorName = sample.sensorDataPoints[j].sensorName
                left = bisect.bisect_left(sensor_timestamps[sensorName], timeframe.start)
                right = bisect.bisect_left(sensor_timestamps[sensorName], timeframe.end)
                data_in_timeframe[sensorName] = sample.sensorDataPoints[j].dataPoints[left:right]
            data_by_timeframe[i] = data_in_timeframe
        return data_by_timeframe

    def parse_timeframe(self, timeframe: Timeframe, sensor_data_points: Dict[str, List[DataPoint]]) -> DataFrame:
        target_len = int((timeframe.end - timeframe.start) / self.delta)
        print(self.delta)
        dataframe_data: List[Dict[str, float]] = [{} for __range__ in range(target_len)]
        for sensor in self.sensors:
            interpolated = self.interpolate_sensor_datapoints(sensor_data_points[sensor.name], len(sensor.dataFormat), timeframe.start, timeframe.end)
            for datapoint_index in range(target_len):                
                for format_index in range(len(sensor.dataFormat)):
                    key = sensor.name + "_" + sensor.dataFormat[format_index]
                    dataframe_data[datapoint_index][key] = interpolated[datapoint_index][format_index]
        return DataFrame(dataframe_data)

    def interpolate_sensor_datapoints(self, datapoints: List[DataPoint], sensor_format_len: int, start: int, end: int) -> List[List[float]]:
        # Normalize the timestamps
        for datapoint in datapoints:
            datapoint.timestamp -= start
            
        result: List[List[float]] = []
        if len(datapoints) == 0:
            return result
        hi = 0
        i = 0
        target_len = int((end - start) / self.delta)
        while i * self.delta <= datapoints[0].timestamp:
            result.append(datapoints[0].data)
            i += 1
        for i in range(i, target_len):
            if i * self.delta >= datapoints[-1].timestamp:
                result.append(datapoints[-1].data)
                continue
            while i * self.delta > datapoints[hi].timestamp:
                hi += 1
            interpolation_percentage = (i * self.delta - datapoints[hi - 1].timestamp) / (datapoints[hi].timestamp - datapoints[hi - 1].timestamp)
            interpolated_datapoint = []
            for format_index in range(sensor_format_len):
                difference = datapoints[hi].data[format_index] - datapoints[hi - 1].data[format_index]
                interpolated_value = datapoints[hi - 1].data[format_index] + difference * interpolation_percentage
                interpolated_datapoint.append(interpolated_value)
            result.append(interpolated_datapoint)
        return result
