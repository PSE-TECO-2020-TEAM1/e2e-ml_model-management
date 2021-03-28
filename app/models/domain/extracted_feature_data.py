from app.models.domain.mongo_model import OID
from pydantic import BaseModel
from app.ml.training.parameters.features import Feature

class ExtractedFeatureData(BaseModel):
    extracted_feature: Feature
    extracted_feature_file_ID: OID