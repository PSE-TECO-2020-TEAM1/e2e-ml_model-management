from app.models.domain.sample import InterpolatedSample
from bson.objectid import ObjectId
from app.models.domain.feature_extraction_data import FeatureExtractionData
from app.models.domain.sliding_window import SlidingWindow
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pickle

@dataclass
class TrainingDataSet():
    last_modified: int = -1 # Guaranteed to be invalid
    sample_list_file_ID: Optional[ObjectId] = None
    # https://bugs.python.org/issue43141 I cannot use SlidingWindow as key because of a Python bug. Temporarily use strings as key
    feature_extraction_cache: Dict[str, FeatureExtractionData] = field(default_factory=dict)

    @staticmethod
    def serialize_sample_list(sample_list: List[InterpolatedSample]) -> bytes:
        return pickle.dumps(sample_list)

    @staticmethod
    def deserialize_sample_list(sample_list: bytes) -> List[InterpolatedSample]:
        return pickle.loads(sample_list)

    def sliding_window_in_cache(self, sliding_window: SlidingWindow) -> bool:
        return str(sliding_window) in self.feature_extraction_cache.keys()

    def get_all_file_IDs(self) -> List[ObjectId]:
        IDs = [self.sample_list_file_ID]
        for entry in self.feature_extraction_cache.values():
            IDs += entry.get_all_file_IDs()
        return IDs
