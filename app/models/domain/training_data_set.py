from app.models.domain.split_to_windows_data import SplitToWindowsData
from typing import List
from app.models.domain.mongo_model import OID
from pydantic import BaseModel
from pydantic.fields import Field
from pandas import DataFrame
import pickle


class Sample(BaseModel):
    label: str
    data_frame: DataFrame


class TrainingDataSet(BaseModel):
    last_modified: int = Field(0)
    sample_list_file_ID: OID = Field(None)
    split_to_windows_entries: List[SplitToWindowsData] = Field([])

    def serialize(samples: List[Sample]) -> bytes:
        return pickle.dumps(samples)

    def deserialize(samples: bytes) -> List[Sample]:
        return pickle.loads(samples)

    def get_all_file_IDs(self) -> List[OID]:
        return [self.sample_list_file_ID] + [entry.get_all_file_IDs() for entry in self.split_to_windows_entries]
