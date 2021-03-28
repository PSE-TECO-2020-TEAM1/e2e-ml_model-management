from pandas.core.frame import DataFrame
from app.models.domain.sliding_window import SlidingWindow
from app.models.domain.mongo_model import OID
from pydantic.fields import Field
from app.models.domain.extracted_feature_data import ExtractedFeatureData
from typing import List
import pickle
from pydantic import BaseModel

class LabeledDataWindows():
    labels: List[str]
    data_windows: List[DataFrame]

class SplitToWindowsData(BaseModel):
    sliding_window: SlidingWindow
    labeled_data_windows_file_ID: OID
    extracted_feature_entries: List[ExtractedFeatureData] = Field([])

    def serialize(labeled_data_windows: LabeledDataWindows) -> bytes:
        return pickle.dumps(labeled_data_windows)

    def deserialize(labeled_data_windows: bytes) -> LabeledDataWindows:
        return pickle.loads(labeled_data_windows)

    def get_all_file_IDs(self) -> List[OID]:
        return [self.labeled_data_windows_file_ID] + [entry.get_all_file_IDs() for entry in self.extracted_feature_entries]