from app.models.domain.extracted_feature_data import ExtractedFeatureData
from pandas.core.frame import DataFrame
from app.ml.training.parameters.features import Feature
from app.models.domain.split_to_windows_data import LabeledDataWindows, SplitToWindowsData
from app.models.domain.sliding_window import SlidingWindow
from app.ml.util.data_preprocessing import split_data_to_windows
from typing import Dict, List

from app.models.domain.training_data_set import Sample, TrainingDataSet
from app.db.error.non_existant_error import NonExistentError
from app.models.domain.mongo_model import OID
from app.models.domain.workspace import Workspace
from gridfs import GridFS
from pymongo.database import Database
import datetime


class WorkspaceRepository():
    def __init__(self, db: Database):
        self.collection = db["workspaces"]
        self.fs = GridFS(db)

    def get_training_data_set(self, workspace_id: OID) -> TrainingDataSet:
        training_data_set = self.collection.find_one({"_id": workspace_id}, {"training_data_set": True})
        if training_data_set is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return TrainingDataSet(**training_data_set)

    def find_split_to_windows_data_in_training_data_set(training_data_set: TrainingDataSet, sliding_window: SlidingWindow) -> SplitToWindowsData:
        for entry in training_data_set.split_to_windows_entries:
            if entry.sliding_window == sliding_window:
                return entry
        raise NonExistentError("Labeled data windows with the given parameter does not exist")

    def find_extracted_feature_data_in_split_to_windows_data(split_to_windows_data: SplitToWindowsData, extracted_feature: Feature) -> ExtractedFeatureData:
        for entry in split_to_windows_data.extracted_feature_entries:
            if entry.extracted_feature == extracted_feature:
                return entry
        raise NonExistentError("Extracted feature with the given name does not exist")

    def get_workspace_by_id(self, workspace_id: OID) -> Workspace:
        workspace = self.collection.find_one({"_id": workspace_id})
        if workspace is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return Workspace(**workspace)

    def get_sample_list(self, workspace_id: OID) -> List[Sample]:
        training_data_set = self.get_training_data_set(workspace_id)
        return TrainingDataSet.deserialize(training_data_set.sample_list_file_ID)

    def replace_sample_list(self, workspace_id: OID, new_sample_list: List[Sample]):
        old_training_data_set = self.get_training_data_set(workspace_id)
        # Delete the old data saved in the file system if there is data at all
        if old_training_data_set.last_modified != 0:
            to_be_deleted_IDs = old_training_data_set.get_all_file_IDs()
            for id in to_be_deleted_IDs:
                self.fs.delete(id)
        serialized = TrainingDataSet.serialize(new_sample_list)
        file_id = self.fs.put(serialized)
        new_training_data_set = TrainingDataSet(last_modified=datetime.utcnow(), sample_list_file_ID=file_id)
        self.collection.update_one({"_id": workspace_id}, {"$set": {"training_data_set": new_training_data_set}})

    def get_labeled_data_windows(self, workspace_id: OID, sliding_window: SlidingWindow) -> LabeledDataWindows:
        training_data_set = self.get_training_data_set(workspace_id)
        split_to_windows_data = WorkspaceRepository.find_split_to_windows_data_in_training_data_set(training_data_set, sliding_window)
        labeled_data_windows = self.fs.get(split_to_windows_data.labeled_data_windows_file_ID)
        return SplitToWindowsData.deserialize(labeled_data_windows)

    def add_labeled_data_windows(self, workspace_id: OID, sliding_window: SlidingWindow, labeled_data_windows: LabeledDataWindows):
        training_data_set = self.get_training_data_set(workspace_id)
        serialized = SplitToWindowsData.serialize(labeled_data_windows)
        file_id = self.fs.put(serialized)
        new_entry = SplitToWindowsData(sliding_window=sliding_window, labeled_data_windows_file_ID=file_id)
        training_data_set.split_to_windows_entries.append(new_entry)
        self.collection.update_one({"_id": workspace_id}, {"$set": {"training_data_set": training_data_set.dict()}})

    def get_extracted_feature(self, workspace_id: OID, sliding_window: SlidingWindow, extracted_feature: Feature) -> DataFrame:
        training_data_set = self.get_training_data_set(workspace_id)
        split_to_windows_data = WorkspaceRepository.find_split_to_windows_data_in_training_data_set(training_data_set, sliding_window)
        extracted_feature_data = WorkspaceRepository.find_extracted_feature_data_in_split_to_windows_data(split_to_windows_data, extracted_feature)
        return self.fs.get(extracted_feature_data.extracted_feature_data_frame_file_ID)

    def add_extracted_feature(self, workspace_id: OID, sliding_window: SlidingWindow, extracted_feature: Feature, extracted_feature_data_frame: DataFrame):
        training_data_set = self.get_training_data_set(workspace_id)
        split_to_windows_data = WorkspaceRepository.find_split_to_windows_data_in_training_data_set(training_data_set, sliding_window)
        serialized = ExtractedFeatureData.serialize(extracted_feature_data_frame)
        file_id = self.fs.put(serialized)
        new_entry = ExtractedFeatureData(extracted_feature=extracted_feature, extracted_feature_data_frame_file_ID=file_id)
        split_to_windows_data.extracted_feature_entries.append(new_entry)
        self.collection.update_one({"_id": workspace_id}, {"$set": {"training_data_set": training_data_set.dict()}})
