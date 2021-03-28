from app.util.sample_parser import SampleParser
import tsfresh
import pickle
import pymongo
from collections import deque
from multiprocessing import set_start_method
from pandas.core.indexes.base import Index
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.connection import Connection
from gridfs import GridFS
from pandas import DataFrame
from typing import Deque, Dict, List
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

from app.models.workspace import DataPoint, DataPointsPerSensor
from app.models.ml_model import MlModel
from app.models.mongo_model import OID
from app.config import get_settings
from app.util.ml_objects import IClassifier, IImputer, INormalizer


class PredictorEntry():
    def __init__(self, sensorDataPoints: List[DataPointsPerSensor], start: int, end: int):
        self.sensorDataPoints: Dict[str, List[DataPoint]] = {i.sensor: i.dataPoints for i in sensorDataPoints}
        self.start = start
        self.end = end

class PredictionResult():
    def __init__(self, labels: List[str], start: int, end: int):
        self.labels = labels
        self.start = start
        self.end = end

class Predictor():
    def __init__(self, model_id: OID):
        self.model_id = model_id

    def __init_objects(self):
        settings = get_settings()
        client = pymongo.MongoClient(settings.DATABASE_URI, settings.DATABASE_PORT)
        db = client[settings.DATABASE_NAME]
        fs = GridFS(db)
        model = MlModel(**db.ml_models.find_one({"_id": self.model_id}))

        self.window_size = model.windowSize
        self.sliding_step = model.slidingStep
        self.sorted_features = model.sortedFeatures
        self.column_order = model.columnOrder
        self.sensors = model.sensors
        self.label_code_to_label = model.labelCodeToLabel

        self.imputation_object: IImputer = pickle.loads(fs.get(model.imputerObject).read())
        self.normalizer_object: INormalizer = pickle.loads(fs.get(model.normalizerObject).read())
        self.classifier_object: IClassifier = pickle.loads(fs.get(model.classifierObject).read())
        client.close()

    # TODO rename
    def init_predictor_process(self, semaphore: SemaphoreType, predictor_end: Connection):
        # The child process can fork safely, even though it must be spawned by the parent
        set_start_method("fork", force=True)

        self.__init_objects()

        parser = SampleParser(self.sensors)
        last_window: Deque[Dict[str, float]] = deque()
        while True:
            while not semaphore.acquire():
                pass

            entry: PredictorEntry = predictor_end.recv()
            if not self.__valid_entry(entry.sensorDataPoints):
                # TODO write -1 to the pipe maybe? or we can even pass the error message since pipe objects are all strings
                continue
            for sensor in self.sensors:
                entry.sensorDataPoints[sensor.name] = parser.interpolate_sensor_datapoints(
                    entry.sensorDataPoints[sensor.name], len(sensor.dataFormat), entry.start, entry.end)
            data_windows: List[List[Dict]] = []
            datapoint_count = int((entry.end - entry.start) / parser.delta)
            while datapoint_count > 0:
                split_data_point: Dict[str, float] = {}
                for sensor in self.sensors:
                    for i in range(len(sensor.dataFormat)):
                        split_data_point[sensor.name + "_" + sensor.dataFormat[i]] = entry.sensorDataPoints[sensor.name][-datapoint_count][i]
                datapoint_count -= 1
                last_window.append(split_data_point)
                if len(last_window) == self.window_size:
                    data_windows.append(list(last_window))
                    for i in range(self.sliding_step):
                        last_window.popleft()
            if len(data_windows) == 0:
                continue
            predictions = self.__predict(data_windows)
            translated_predictions = []
            for prediction in predictions:
                translated = self.label_code_to_label[prediction]
                translated_predictions.append(translated)
            predictor_end.send(PredictionResult(labels=translated_predictions, start=entry.start, end=entry.end))

    def __valid_entry(self, data: Dict[str, List[DataPoint]]):
        for sensor in self.sensors:
            if sensor.name not in data:
                return False
            for data_point in data[sensor.name]:
                if len(data_point.data) != len(sensor.dataFormat):
                    return False
        return True

    def __predict(self, pipeline_data: List[List[Dict]]) -> List[int]:
        pipeline_data = self.__preprocess(pipeline_data)
        return self.classifier_object.predict(pipeline_data)

    def __preprocess(self, pipeline_data: List[List[Dict]]) -> DataFrame:  # List[List[DataPoint]]
        pipeline_data = self.__extract_features(pipeline_data)
        pipeline_data = self.__impute(pipeline_data)
        pipeline_data = self.__normalize(pipeline_data)
        return pipeline_data

    def __extract_features(self, pipeline_data: List[List[Dict]]) -> DataFrame:
        concatenated_data_windows: List[Dict] = []
        for data_window_index in range(len(pipeline_data)):
            for i in range(len(pipeline_data[data_window_index])):
                pipeline_data[data_window_index][i]["id"] = data_window_index
                concatenated_data_windows.append(pipeline_data[data_window_index][i])

        settings = {key: ComprehensiveFCParameters()[key] for key in self.sorted_features}
        extracted = tsfresh.extract_features(DataFrame(concatenated_data_windows), column_id="id",
                                             default_fc_parameters=settings, disable_progressbar=True)
        return extracted[Index(self.column_order)]

    def __impute(self, pipeline_data: DataFrame) -> DataFrame:
        return DataFrame(self.imputation_object.transform(pipeline_data), columns=pipeline_data.columns)

    def __normalize(self, pipeline_data: DataFrame) -> DataFrame:
        return DataFrame(self.normalizer_object.transform(pipeline_data), columns=pipeline_data.columns)
