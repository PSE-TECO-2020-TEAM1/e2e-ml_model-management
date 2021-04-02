from app.models.domain.sensor import SensorComponent
from dataclasses import dataclass
from bson.objectid import ObjectId
from pandas.core.frame import DataFrame
from app.ml.training.parameters.features import Feature
from typing import Dict, List
import pickle


@dataclass
class SplitToWindowsData():
    data_windows_df_file_ID: ObjectId
    labels_of_data_windows_file_ID: ObjectId
    sensor_component_feature_df_file_IDs: Dict[SensorComponent, Dict[Feature, ObjectId]] = {}

    @staticmethod
    def serialize_data_windows_df(data_windows_df: DataFrame) -> bytes:
        return pickle.dumps(data_windows_df)

    @staticmethod
    def deserialize_data_windows_df(data_windows_df: bytes) -> DataFrame:
        return pickle.loads(data_windows_df)

    @staticmethod
    def serialize_labels_of_data_windows(labels_of_data_windows: List[str]) -> bytes:
        return pickle.dumps(labels_of_data_windows)

    @staticmethod
    def deserialize_labels_of_data_windows(labels_of_data_windows: bytes) -> List[str]:
        return pickle.loads(labels_of_data_windows)

    @staticmethod
    def serialize_sensor_component_feature_df(sensor_component_feature_df: DataFrame) -> bytes:
        return pickle.dumps(sensor_component_feature_df)

    @staticmethod
    def deserialize_sensor_component_feature_df(sensor_component_feature_df: bytes) -> DataFrame:
        return pickle.loads(sensor_component_feature_df)

    def get_all_file_IDs(self) -> List[ObjectId]:
        IDs = [self.labels_of_data_windows_file_ID, self.data_windows_df_file_ID]
        # TODO
        # return IDs + list(res.values() for res in self.sensor_component_feature_df_file_IDs.values())