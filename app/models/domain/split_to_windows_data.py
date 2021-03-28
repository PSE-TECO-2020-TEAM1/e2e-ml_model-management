from app.models.domain.mongo_model import OID
from pydantic.fields import Field
from app.models.domain.extracted_feature_data import ExtractedFeatureData
from typing import List
from pydantic import BaseModel

class SplitToWindowsData(BaseModel):
    window_size: int
    sliding_step: int
    split_to_windows_file_ID: OID
    extracted_feature_entries: List[ExtractedFeatureData] = Field([])