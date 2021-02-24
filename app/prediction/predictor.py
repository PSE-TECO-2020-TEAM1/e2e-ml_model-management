from pandas.core.indexes.base import Index
from app.models.ml_model import MlModel
import tsfresh
import pickle
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.connection import Connection
from gridfs import GridFS
from pymongo.mongo_client import MongoClient
from pandas import DataFrame
from typing import Dict, List
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

from app.models.mongo_model import OID
from app.config import get_settings
from app.util.ml_objects import IClassifier, IImputer, INormalizer


class PredictorEntry():
    def __init__(self, data_points: Dict[str, List[List[float]]], data_point_count: int):
        self.data_points = data_points
        self.data_point_count = data_point_count


class Predictor():
    def __init__(self, model_id: OID):
        self.model_id = model_id

    def __init_objects(self):
        settings = get_settings()
        client = MongoClient(settings.client_uri, settings.client_port)
        db = client[settings.db_name]
        fs = GridFS(db)
        model = MlModel(**db.ml_models.find_one({"_id": self.model_id}))

        self.window_size = model.windowSize
        self.sliding_step = model.slidingStep
        self.sorted_features = model.sortedFeatures
        self.column_order = model.columnOrder
        self.label_code_to_label = model.labelCodeToLabel

        self.imputation_object: IImputer = pickle.loads(fs.get(model.imputerObject).read())
        self.normalizer_object: INormalizer = pickle.loads(fs.get(model.normalizerObject).read())
        self.classifier_object: IClassifier = pickle.loads(fs.get(model.classifierObject).read())
        client.close()

    # TODO rename
    def for_process(self, semaphore: SemaphoreType, predictor_end: Connection):
        self.__init_objects()
        while True:

            while not semaphore.acquire():
                pass

            # TODO implement the actual queue logic here
            entry: PredictorEntry = predictor_end.recv()
            dataframe_data = [{} for __range__ in range(entry.data_point_count)]
            for sensor, sensor_data_points in entry.data_points.items():
                sensor_data_point_index = 0
                for sensor_data_point in sensor_data_points:
                    dataframe_data[sensor_data_point_index].update(
                        {sensor + str(i): sensor_data_point[i] for i in range(len(sensor_data_point))})
                    sensor_data_point_index += 1
            prediction = self.__predict(DataFrame(dataframe_data))
            translated_prediction = self.label_code_to_label[str(prediction[0])]
            predictor_end.send(translated_prediction)

    def __predict(self, pipeline_data: DataFrame) -> int:
        pipeline_data = self.__preprocess(pipeline_data)
        return self.classifier_object.predict(pipeline_data)

    def __preprocess(self, pipeline_data: DataFrame) -> DataFrame:
        pipeline_data = self.__extract_features(pipeline_data)
        pipeline_data = self.__impute(pipeline_data)
        pipeline_data = self.__normalize(pipeline_data)
        return pipeline_data

    def __extract_features(self, pipeline_data: DataFrame) -> DataFrame:
        settings = {key: ComprehensiveFCParameters()[key] for key in self.sorted_features}
        pipeline_data["id"] = 0
        # Don't spawn new processes, the prediction data is probably tiny and process spawn overhead is huge
        extracted = tsfresh.extract_features(pipeline_data, column_id="id",
                                             default_fc_parameters=settings, n_jobs=0, disable_progressbar=False)
        return extracted[Index(self.column_order)]

    def __impute(self, pipeline_data: DataFrame) -> DataFrame:
        return DataFrame(self.imputation_object.transform(pipeline_data), columns=pipeline_data.columns)

    def __normalize(self, pipeline_data: DataFrame) -> DataFrame:
        return DataFrame(self.normalizer_object.transform(pipeline_data), columns=pipeline_data.columns)
