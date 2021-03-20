import bisect
from typing import Dict, List
from numpy import invert
from pandas.core.frame import DataFrame
from app.models.workspace import DataPoint, SampleInJson, Sensor, Timeframe

class ParsedSample():
    def __init__(self, positive: List[DataFrame], negative: List[DataFrame]):
        self.positive = positive
        self.negative = negative

class SampleParser():
    def __init__(self, sensors: List[Sensor]):
        self.sensors = sensors
        self.delta = 1000 / max(sensor.samplingRate for sensor in sensors)

    def parse_sample(self, sample: SampleInJson) -> ParsedSample:
        data_by_timeframe = self.split_data_by_timeframe(sample)
        dataframes_of_positive: List[DataFrame] = []
        for i in data_by_timeframe:
            dataframe_of_timeframe = self.parse_timeframe(sample.timeFrames[i], data_by_timeframe[i])
            dataframes_of_positive.append(dataframe_of_timeframe)

        self.invert_timeframes(sample)
        data_by_timeframe = self.split_data_by_timeframe(sample)
        dataframes_of_negative: List[DataFrame] = []
        for i in data_by_timeframe:
            dataframe_of_timeframe = self.parse_timeframe(sample.timeFrames[i], data_by_timeframe[i])
            dataframes_of_negative.append(dataframe_of_timeframe)
        return ParsedSample(positive=dataframes_of_positive, negative=dataframes_of_negative)


    def invert_timeframes(self, sample: SampleInJson):
        timeframes = sample.timeFrames
        inverted_timeframes: List[Timeframe] = []
        # Add the beginning and the end
        if sample.start - timeframes[0].start > 0:
            inverted_timeframes.append(Timeframe(start=sample.start, end=timeframes[0].start))
        if sample.end - timeframes[-1].end > 0:
            inverted_timeframes.append(Timeframe(start=sample.end - timeframes[-1].end))
        # Handle rest
        for i in range(len(timeframes) - 1):
            between_two = Timeframe(start=timeframes[i].end, end=timeframes[i+1].start)
            inverted_timeframes.append(between_two)
        sample.timeFrames = inverted_timeframes

    def split_data_by_timeframe(self, sample: SampleInJson) -> Dict[int, Dict[str, List[DataPoint]]]:
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
