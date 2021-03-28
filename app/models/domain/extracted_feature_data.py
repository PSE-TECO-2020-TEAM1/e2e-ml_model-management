import pickle
from typing import List

from pandas.core.frame import DataFrame
from app.models.domain.mongo_model import OID
from pydantic import BaseModel
from app.ml.training.parameters.features import Feature

class ExtractedFeatureData(BaseModel):
    extracted_feature: Feature
    extracted_feature_data_frame_file_ID: OID

    def serialize(extracted_feature_data_frame: DataFrame) -> bytes:
        return pickle.dumps(extracted_feature_data_frame)

    def deserialize(extracted_feature_data_frame: bytes) -> DataFrame:
        return pickle.loads(extracted_feature_data_frame)

    def get_all_file_IDs(self) -> List[OID]:
        return [self.data_frame_file_ID]