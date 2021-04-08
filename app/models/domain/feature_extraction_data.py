from app.models.domain.sensor import SensorComponent
from dataclasses import dataclass, field
from bson.objectid import ObjectId
from pandas.core.frame import DataFrame
from app.ml.objects.feature import Feature
from typing import Dict, List
import pickle


@dataclass
class FeatureExtractionData():
    data_windows_df_file_ID: ObjectId
    labels_of_data_windows_file_ID: ObjectId
    sensor_component_feature_df_file_IDs: Dict[SensorComponent, Dict[Feature, ObjectId]] = field(default_factory=dict)

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

    def sensor_component_feature_in_cache(self, sensor_component: SensorComponent, feature: Feature) -> bool:
        if sensor_component not in self.sensor_component_feature_df_file_IDs.keys():
            return False
        return feature in self.sensor_component_feature_df_file_IDs[sensor_component].keys()

    def get_all_file_IDs(self) -> List[ObjectId]:
        IDs = [self.labels_of_data_windows_file_ID, self.data_windows_df_file_ID]
        for i in self.sensor_component_feature_df_file_IDs.values():
            IDs += list(i.values())
        return IDs
