from app.ml.util.sample_parsing import parse_samples_from_workspace
from app.models.domain.sample import InterpolatedSample
from app.models.domain.split_to_windows_data import SplitToWindowsData
from app.models.domain.sliding_window import SlidingWindow
from pandas.core.frame import DataFrame
from app.ml.objects.feature.enum import Feature
from app.models.domain.sensor import SensorComponent
from bson.objectid import ObjectId
from app.db.sync.file_repository import FileRepository
from typing import List
from app.models.domain.training_data_set import TrainingDataSet
from app.db.sync.workspace_repository import WorkspaceRepository
from app.workspace_management_api.data_source import ExternalDataSource
from app.ml.training.error.training_error import TrainingError
from pymongo.database import Database


class DataSetManager():
    def __init__(self, workspace_id: ObjectId, external_data_source: ExternalDataSource):
        self.workspace_id = workspace_id
        self.external_data_source = external_data_source

    def is_valid_cache_manager(self):
        if self.workspace_repository is None:
            raise TrainingError("Database is not configured")

    def set_db(self, db: Database):
        self.file_repository = FileRepository(db)
        self.workspace_repository = WorkspaceRepository(db)

    def update_training_data_set(self):
        self.is_valid_cache_manager()
        workspace = self.workspace_repository.get_workspace(self.workspace_id)
        last_modified = self.external_data_source.last_modified(self.workspace_id)
        if workspace.training_data_set.last_modified == last_modified:
            return
        raw_samples = self.external_data_source.fetch_samples(self.workspace_id)
        interpolated_samples = parse_samples_from_workspace(raw_samples)
        # Don't forget to delete the old files first
        for file_id in TrainingDataSet.get_all_file_IDs():
            self.file_repository.delete_file(file_id)
        file_id = self.file_repository.put_file(TrainingDataSet.serialize_sample_list(interpolated_samples))
        new_training_data_set = TrainingDataSet(last_modified=last_modified, sample_list_file_ID=file_id)
        self.workspace_repository.set_training_data_set(new_training_data_set)

    def get_sample_list(self) -> List[InterpolatedSample]:
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        sample_list_file = self.file_repository.get_file(training_data_set.sample_list_file_ID)
        return TrainingDataSet.deserialize_sample_list(sample_list_file)

    def get_labels_of_data_windows(self, sliding_window: SlidingWindow) -> List[str]:
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_id = training_data_set.split_to_windows_cache[sliding_window].labels_of_data_windows_file_ID
        return SplitToWindowsData.deserialize_labels_of_data_windows(self.file_repository.get_file(file_id))

    def is_cached_split_to_windows(self, sliding_window: SlidingWindow) -> bool:
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        return training_data_set.sliding_window_in_cache(sliding_window)

    def get_cached_split_to_windows(self, sliding_window: SlidingWindow) -> DataFrame:
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_ID = training_data_set.split_to_windows_cache[sliding_window].data_windows_df_file_ID
        return SplitToWindowsData.deserialize_data_windows_df(self.file_repository.get_file(file_ID))

    def add_split_to_windows(self, sliding_window: SlidingWindow, data_windows: DataFrame, labels_of_data_windows: List[str]):
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        data_windows_df_file_ID = self.file_repository.put_file(SplitToWindowsData.serialize_data_windows_df(data_windows))
        labels_of_data_windows_file_ID = self.file_repository.put_file(SplitToWindowsData.serialize_labels_of_data_windows(labels_of_data_windows))
        res = SplitToWindowsData(data_windows_df_file_ID=data_windows_df_file_ID, labels_of_data_windows_file_ID=labels_of_data_windows_file_ID)
        training_data_set.split_to_windows_cache[sliding_window] = res
        self.workspace_repository.set_training_data_set(self.workspace_id, training_data_set)

    def is_cached_sensor_component_feature(self, sliding_window: SlidingWindow, sensor_component: SensorComponent, feature: Feature) -> bool:
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        if training_data_set.sliding_window_in_cache(sliding_window):
            return False
        return training_data_set.split_to_windows_cache[sliding_window].sensor_component_feature_in_cache(sensor_component, feature)

    def get_cached_sensor_component_feature(self, sliding_window: SlidingWindow, sensor_component: SensorComponent, feature: Feature) -> DataFrame:
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_ID = training_data_set.split_to_windows_cache[sliding_window].sensor_component_feature_df_file_IDs[sensor_component][feature]
        return SplitToWindowsData.deserialize_sensor_component_feature_df(self.file_repository.get_file(file_ID))

    def add_sensor_component_feature(self, sliding_window: SlidingWindow, sensor_component: SensorComponent, feature: Feature, feature_df: DataFrame):
        self.is_valid_cache_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_IDs_dict = training_data_set.split_to_windows_cache[sliding_window].sensor_component_feature_df_file_IDs
        file_ID = self.file_repository.put_file(SplitToWindowsData.serialize_sensor_component_feature_df(feature_df))
        if sensor_component in file_IDs_dict.keys():
            file_IDs_dict[sensor_component][feature] = file_ID
        else:
            file_IDs_dict[sensor_component] = {feature: file_ID}
