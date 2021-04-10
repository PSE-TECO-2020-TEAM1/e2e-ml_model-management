from app.ml.training.training_state import TrainingState
from app.db.syncdb.ml_model_repository import MlModelRepository
from app.models.domain.ml_model import MlModel
from app.ml.util.sample_parsing import parse_samples_from_workspace
from app.models.domain.sample import InterpolatedSample
from app.models.domain.feature_extraction_data import FeatureExtractionData
from app.models.domain.sliding_window import SlidingWindow
from pandas.core.frame import DataFrame
from app.ml.objects.feature.enum import Feature
from app.models.domain.sensor import SensorComponent
from bson.objectid import ObjectId
from app.db.syncdb.file_repository import FileRepository
from typing import List
from app.models.domain.training_data_set import TrainingDataSet
from app.db.syncdb.workspace_repository import WorkspaceRepository
from app.workspace_management_api.data_source import ExternalDataSource
from app.ml.training.error.training_error import TrainingError
from pymongo.database import Database


class DataSetManager():
    def __init__(self, workspace_id: ObjectId, external_data_source: ExternalDataSource):
        self.workspace_id = workspace_id
        self.external_data_source = external_data_source

    def is_valid_data_set_manager(self):
        if not self.db_is_set:
            raise TrainingError("Database is not configured")

    def set_db(self, db: Database):
        self.db_is_set = True
        self.file_repository = FileRepository(db)
        self.workspace_repository = WorkspaceRepository(db)
        self.ml_model_repository = MlModelRepository(db)

    def update_training_data_set(self):
        self.is_valid_data_set_manager()
        workspace = self.workspace_repository.get_workspace(self.workspace_id)
        last_modified = self.external_data_source.last_modified(self.workspace_id)
        if workspace.training_data_set.last_modified == last_modified:
            return
        raw_samples = self.external_data_source.fetch_samples(self.workspace_id)
        interpolated_samples = parse_samples_from_workspace(raw_samples, workspace.sensors)
        # Don't forget to delete the old files first
        for file_id in workspace.training_data_set.get_all_file_IDs():
            self.file_repository.delete_file(file_id)
        file_id = self.file_repository.put_file(TrainingDataSet.serialize_sample_list(interpolated_samples))
        new_training_data_set = TrainingDataSet(last_modified=last_modified, sample_list_file_ID=file_id)
        self.workspace_repository.set_training_data_set(self.workspace_id, new_training_data_set)

    def get_sample_list(self) -> List[InterpolatedSample]:
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        sample_list_file = self.file_repository.get_file(training_data_set.sample_list_file_ID)
        return TrainingDataSet.deserialize_sample_list(sample_list_file)

    def is_cached_split_to_windows(self, sliding_window: SlidingWindow) -> bool:
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        return training_data_set.sliding_window_in_cache(sliding_window)

    def get_labels_of_data_windows(self, sliding_window: SlidingWindow) -> List[str]:
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_ID = training_data_set.feature_extraction_cache[str(sliding_window)].labels_of_data_windows_file_ID
        return FeatureExtractionData.deserialize_labels_of_data_windows(self.file_repository.get_file(file_ID))

    def get_cached_split_to_windows(self, sliding_window: SlidingWindow) -> DataFrame:
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_ID = training_data_set.feature_extraction_cache[str(sliding_window)].data_windows_df_file_ID
        return FeatureExtractionData.deserialize_data_windows_df(self.file_repository.get_file(file_ID))

    def add_split_to_windows(self, sliding_window: SlidingWindow, data_windows: DataFrame, labels_of_data_windows: List[str]):
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        data_windows_df_file_ID = self.file_repository.put_file(FeatureExtractionData.serialize_data_windows_df(data_windows))
        labels_of_data_windows_file_ID = self.file_repository.put_file(FeatureExtractionData.serialize_labels_of_data_windows(labels_of_data_windows))
        res = FeatureExtractionData(data_windows_df_file_ID=data_windows_df_file_ID, labels_of_data_windows_file_ID=labels_of_data_windows_file_ID)
        training_data_set.feature_extraction_cache[str(sliding_window)] = res
        self.workspace_repository.set_training_data_set(self.workspace_id, training_data_set)

    def is_cached_sensor_component_feature(self, sliding_window: SlidingWindow, sensor_component: SensorComponent, feature: Feature) -> bool:
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        if not training_data_set.sliding_window_in_cache(sliding_window):
            return False
        return training_data_set.feature_extraction_cache[str(sliding_window)].sensor_component_feature_in_cache(sensor_component, feature)

    def get_cached_sensor_component_feature(self, sliding_window: SlidingWindow, sensor_component: SensorComponent, feature: Feature) -> DataFrame:
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_ID = training_data_set.feature_extraction_cache[str(sliding_window)].sensor_component_feature_df_file_IDs[sensor_component][feature]
        return FeatureExtractionData.deserialize_sensor_component_feature_df(self.file_repository.get_file(file_ID))

    def add_sensor_component_feature(self, sliding_window: SlidingWindow, sensor_component: SensorComponent, feature: Feature, feature_df: DataFrame):
        self.is_valid_data_set_manager()
        training_data_set = self.workspace_repository.get_training_data_set(self.workspace_id)
        file_IDs_dict = training_data_set.feature_extraction_cache[str(sliding_window)].sensor_component_feature_df_file_IDs
        file_ID = self.file_repository.put_file(FeatureExtractionData.serialize_sensor_component_feature_df(feature_df))
        if sensor_component in file_IDs_dict.keys():
            file_IDs_dict[sensor_component][feature] = file_ID
        else:
            file_IDs_dict[sensor_component] = {feature: file_ID}
        self.workspace_repository.set_training_data_set(self.workspace_id, training_data_set)

    def set_training_state(self, training_state: TrainingState):
        self.is_valid_data_set_manager()
        self.workspace_repository.set_training_state(self.workspace_id, training_state)

    def save_model(self, config, label_performance_metrics, column_order, label_encoder, pipeline):
        self.is_valid_data_set_manager()
        label_encoder_object_file_ID = self.file_repository.put_file(MlModel.serialize_label_encoder(label_encoder))
        pipeline_object_file_ID = self.file_repository.put_file(MlModel.serialize_pipeline(pipeline))
        model_in_db = MlModel(None, config, label_performance_metrics, column_order, label_encoder_object_file_ID, pipeline_object_file_ID)
        model_id = self.ml_model_repository.add_ml_model(model_in_db)
        self.workspace_repository.add_ml_model_ref(self.workspace_id, model_id)
