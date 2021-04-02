from bson.objectid import ObjectId
from app.models.domain.split_to_windows_data import SplitToWindowsData
from app.models.domain.sliding_window import SlidingWindow
from typing import Dict, List
from pandas import DataFrame
from dataclasses import dataclass
import pickle

@dataclass
class InterpolatedSample():
    label: str
    data_frame: DataFrame
    
@dataclass
class TrainingDataSet():
    last_modified: int
    sample_list_file_ID: ObjectId
    split_to_windows_cache: Dict[SlidingWindow, SplitToWindowsData] = {}

    @staticmethod
    def serialize_sample_list(sample_list: List[InterpolatedSample]) -> bytes:
        return pickle.dumps(sample_list)

    @staticmethod
    def deserialize_sample_list(sample_list: bytes) -> List[InterpolatedSample]:
        return pickle.loads(sample_list)

    def get_all_file_IDs(self) -> List[ObjectId]:
        IDs = [self.sample_list_file_ID]
        for entry in self.split_to_windows_cache.values():
            IDs += entry.get_all_file_IDs()
        return IDs
