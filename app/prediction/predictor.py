import tsfresh
import pickle
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.connection import Connection
from gridfs import GridFS
from pymongo.mongo_client import MongoClient
from pandas import DataFrame
from typing import Dict, List
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

from app.models.workspace import DataPoint
from app.models.mongo_model import OID
from app.config import get_settings
from app.util.training_parameters import Feature
from app.util.ml_objects import IClassifier, IImputer, INormalizer


class Predictor():
    def __init__(self, window_size: int, sliding_step: int, imputation_id: OID, normalizer_id: OID, classifier_id: OID, features: List[Feature], label_code_to_label: Dict[int, str]):
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.imputation_id = imputation_id
        self.normalizer_id = normalizer_id
        self.classifier_id = classifier_id
        self.features = features
        self.label_code_to_label = label_code_to_label

    def __init_objects(self):
        settings = get_settings()
        client = MongoClient(settings.client_uri, settings.client_port)
        db = client[settings.db_name]
        fs = GridFS(db)

        self.imputation_object: IImputer = pickle.loads(fs.get(self.imputation_id).read())
        self.normalizer_object: INormalizer = pickle.loads(fs.get(self.normalizer_id).read())
        self.classifier_object: IClassifier = pickle.loads(fs.get(self.classifier_id).read())
        client.close()

    #TODO rename
    def for_process(self, semaphore: SemaphoreType, predictor_end: Connection):
        self.__init_objects()
        while True:
            while not semaphore.acquire(): 
                print("Couldn't acquire semaphore :(") #TODO delet this
                pass
            # TODO implement the actual logic here
            data_window: Dict[str, List[DataPoint]] = predictor_end.recv()
            data_point_count = 4 #TODO replace 2 with the number of datapoints (maybe pass from the request via pipe?)
            dataframe_data = [{} for __range__ in range(data_point_count)]
            for sensor, sensor_data_points in data_window.items():
                sensor_data_point_index = 0
                for sensor_data_point in sensor_data_points:
                    sensor_data_point = sensor_data_point.values
                    dataframe_data[sensor_data_point_index].update({sensor + str(i): sensor_data_point[i] for i in range(len(sensor_data_point))})
                    sensor_data_point_index +=1
            prediction = self.__predict(DataFrame(dataframe_data))
            translated_prediction = self.label_code_to_label[str(prediction[0])]
            predictor_end.send(translated_prediction)

    def __predict(self, data_window: DataFrame) -> int:
        pipeline_data = self.__preprocess(data_window)
        return self.classifier_object.predict(pipeline_data)

    def __preprocess(self, pipeline_data: DataFrame) -> DataFrame:
        pipeline_data = self.__extract_features(pipeline_data)
        pipeline_data = self.__impute(pipeline_data)
        pipeline_data = self.__normalize(pipeline_data)
        return pipeline_data

    def __extract_features(self, data_window: DataFrame) -> DataFrame:
        settings = {key: ComprehensiveFCParameters()[key] for key in self.features}
        data_window["id"] = 0
        return tsfresh.extract_features(
            data_window, column_id="id", default_fc_parameters=settings, disable_progressbar=False)

    def __impute(self, data_window: DataFrame) -> DataFrame:
        return DataFrame(self.imputation_object.transform(data_window), columns=data_window.columns)

    def __normalize(self, data_window: DataFrame) -> DataFrame:
        return DataFrame(self.normalizer_object.transform(data_window), columns=data_window.columns)
