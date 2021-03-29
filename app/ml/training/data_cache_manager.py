from app.ml.training.parameters.features import Feature
from typing import List
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

    def set_database(self, db: Database):
        self.workspace_repository = WorkspaceRepository(db)

    def update_training_data_set(self):
        self.is_valid_cache_manager()
        workspace = self.workspace_repository.get_workspace_by_id(self.workspace_id)
        last_modified = self.external_data_source.last_modified(self.workspace_id)
        if workspace.training_data_set.last_modified == last_modified:
            return
        # TODO

    def get_training_data_set(self) -> List[Sample]:
        self.is_valid_cache_manager()


    def list_cached_split_to_windows(self, parameter: SplitToWindowsLevel) -> List[SlidingWindow]:
        self.is_valid_cache_manager()


    def get_cached_split_to_windows(self, parameter: SplitToWindowsLevel) -> LabeledDataWindows:
        self.is_valid_cache_manager()

    def list_cached_extracted_features(self, parameter: SplitToWindowsLevel) -> List[Feature]:
        self.is_valid_cache_manager()


    def get_cached_extracted_feature(self, parameter: ExtractedFeatureLevel) -> List[DataFrame]:
        self.is_valid_cache_manager()
