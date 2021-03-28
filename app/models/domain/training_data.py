from app.models.domain.split_to_windows_data import SplitToWindowsData
from typing import List
from app.models.domain.extracted_feature import ExtractedFeature
from app.models.domain.mongo_model import OID
from pydantic import BaseModel
from pydantic.fields import Field

class DataSet(BaseModel):
    last_modified: int = Field(0)
    raw_sensor_data_file_ID: OID = Field(None)
    split_to_windows_entries: List[SplitToWindowsData] = Field([])