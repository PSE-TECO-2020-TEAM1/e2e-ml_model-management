from os import pipe
from pandas import DataFrame, concat
import tsfresh
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from app.models.workspace import DataPoint
from app.util.training_parameters import Feature
from app.util.ml_objects import IClassifier, IImputer, INormalizer
from typing import Dict, List


class Predictor():
    def __init__(self, window_size: int, sliding_step: int, imputation_object: IImputer, normalizer_object: INormalizer, classifier_object: IClassifier, features: List[Feature], label_code_to_label: Dict[int, str]):
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.imputation_object = imputation_object
        self.normalizer_object = normalizer_object
        self.classifier_object = classifier_object
        self.features = sorted(features)
        self.label_code_to_label = label_code_to_label

    #TODO rename
    def for_process(self, pipe_write, cv):
        # 
        # while 1:
            # wait(cv)
            # condition check
            # self.__predict(data)
        pass

    def __predict(self):
        # data = from shared memory queue
        data_window: DataFrame
        pipeline_data = self.__preprocess(data_window)
        self.classifier_object.predict(pipeline_data)

    def __preprocess(self, pipeline_data: DataFrame) -> DataFrame:
        pipeline_data = self.__extract_features(pipeline_data)
        pipeline_data = self.__impute(pipeline_data)
        pipeline_data = self.__normalize(pipeline_data)
        return pipeline_data

    def __extract_features(self, data_window: DataFrame) -> DataFrame:
        settings = {key.value: ComprehensiveFCParameters()[key.value] for key in self.features}
        data_window["id"] = 0
        return tsfresh.extract_features(
            data_window, column_id="id", default_fc_parameters=settings, disable_progressbar=False)

    def __impute(self, data_window: DataFrame) -> DataFrame:
        return DataFrame(self.imputation_object.transform(data_window), columns=data_window.columns)

    def __normalize(self, data_window: DataFrame) -> DataFrame:
        return DataFrame(self.normalizer_object.transform(data_window), columns=data_window.columns)
