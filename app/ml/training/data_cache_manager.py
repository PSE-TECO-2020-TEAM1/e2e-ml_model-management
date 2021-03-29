from app.ml.training.parameters.features import Feature
from typing import Dict, List
from app.models.domain.training_data_set import Sample
from pandas.core.arrays import boolean
from app.models.domain.split_to_windows_data import LabeledDataWindows
from dataclasses import dataclass
from app.models.domain.sliding_window import SlidingWindow
from app.db.sync.workspace_repository import WorkspaceRepository
from app.workspace_management_api.data_source import ExternalDataSource
from app.ml.training.error.training_error import TrainingError
from pymongo.database import Database
from app.models.domain.mongo_model import OID
from pandas import DataFrame


@dataclass
class SplitToWindowsLevel():
    sliding_window: SlidingWindow


@dataclass
class ExtractedFeatureLevel(SplitToWindowsLevel):
    features: List[Feature]


class DataCacheManager():
    def __init__(self, workspace_id: OID, external_data_source: ExternalDataSource):
        self.workspace_repository: WorkspaceRepository = None
        self.workspace_id = workspace_id
        self.external_data_source = external_data_source

    def is_valid_cache_manager(self):
        if self.workspace_repository is None:
            raise TrainingError("Database is not configured")

    def set_workspace_repository(self, workspace_repository: WorkspaceRepository):
        self.workspace_repository = workspace_repository

    def update_training_data_set(self):
        self.is_valid_cache_manager()
        workspace = self.workspace_repository.get_workspace_by_id(self.workspace_id)
        last_modified = self.external_data_source.last_modified(self.workspace_id)
        if workspace.training_data_set.last_modified == last_modified:
            return
        # TODO

    def get_sample_list(self) -> List[Sample]:
        self.is_valid_cache_manager()
        return self.workspace_repository.get_sample_list()

    def list_cached_labeled_data_windows(self) -> List[SlidingWindow]:
        self.is_valid_cache_manager()
        workspace = self.workspace_repository.get_workspace_by_id(self.workspace_id)
        sliding_windows = []
        for entry in workspace.training_data_set.split_to_windows_entries:
            sliding_windows.append(entry.sliding_window)
        return sliding_windows

    def get_cached_labeled_data_windows(self, parameter: SplitToWindowsLevel) -> LabeledDataWindows:
        self.is_valid_cache_manager()
        return self.workspace_repository.get_labeled_data_windows(parameter.sliding_window)

    def add_labeled_data_windows(self, labeled_data_windows: LabeledDataWindows):
        self.is_valid_cache_manager()
        self.workspace_repository.add_labeled_data_windows(labeled_data_windows)

    def list_cached_extracted_features(self, parameter: SplitToWindowsLevel) -> List[Feature]:
        self.is_valid_cache_manager()
        workspace = self.workspace_repository.get_workspace_by_id(self.workspace_id)
        extracted_features = []
        for split_to_windows_entry in workspace.training_data_set.split_to_windows_entries:
            if split_to_windows_entry.sliding_window == parameter.sliding_window:
                for extracted_feature_entry in split_to_windows_entry.extracted_feature_entries:
                    extracted_features.append(extracted_feature_entry.extracted_feature)
        return extracted_features

    def get_cached_extracted_feature_data_frames(self, parameter: ExtractedFeatureLevel) -> Dict[Feature, DataFrame]:
        self.is_valid_cache_manager()
        result = {}
        for feature in parameter.features:
            result[feature] = self.workspace_repository.get_extracted_feature(self.workspace_id, parameter.sliding_window, feature)
        return result

    def add_extracted_feature_data_frames(self, sliding_window: SlidingWindow, extracted_features: Dict[Feature, DataFrame]):
        self.is_valid_cache_manager()
        for feature in extracted_features:
            self.workspace_repository.add_extracted_feature(self.workspace_id, sliding_window, feature, extracted_features[feature])
