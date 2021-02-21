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

    def predict(self, sensor_data_points: Dict[str, List[DataPoint]]):
        data = self.__preprocess(sensor_data_points)
        # TODO

    def __preprocess(self, sensor_data_points: Dict[str, List[DataPoint]]) -> DataFrame:
        pipeline_data = sensor_data_points
        pipeline_data = self.__split_to_windows(pipeline_data)
        pipeline_data = self.__extract_features(pipeline_data)
        pipeline_data = self.__impute(pipeline_data)
        pipeline_data = self.__normalize(pipeline_data)
        return pipeline_data

    def __split_to_windows(self, sensor_data_points: Dict[str, List[DataPoint]]) -> List[DataFrame]:
        data_windows: List[DataFrame] = []
        for x in range(0, len(list(sensor_data_points.values())[0])):
            data_window: List[Dict[str, float]] = [{} for _range_ in range(self.window_size)]
            for sensor in sorted(sensor_data_points):
                data_points: List[DataPoint] = sensor_data_points[sensor]
                for i in range(0, self.window_size):
                        data_point_values = data_points[x+i].values
                        data_window[i].update({sensor+str(v): data_point_values[v]
                                               for v in range(len(data_point_values))})
            data_windows.append(DataFrame(data_window))
        return data_windows

    def __extract_features(self, data: List[DataFrame]) -> DataFrame:
        settings = {key.value: ComprehensiveFCParameters()[key.value] for key in self.features}
        extracted_features: List[DataFrame] = []
        for data_window in data:
            data_window["id"] = 0
            result = tsfresh.extract_features(
                data_window, column_id="id", default_fc_parameters=settings, disable_progressbar=True)
            result.drop(columns=["id"])
            extracted_features.append(result)
        return concat(extracted_features, axis=1)

    def __impute(self, data: DataFrame) -> DataFrame:
        return DataFrame(self.imputation_object.transform(data), columns=data.columns)

    def __normalize(self, data: DataFrame) -> DataFrame:
        return DataFrame(self.normalizer_object.transform(data), columns=data.columns)
