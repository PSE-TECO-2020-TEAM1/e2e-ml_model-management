from typing import Dict, List, Optional
from pydantic import Field

from app.models.mongo_model import OID, MongoModel
from app.util.training_parameters import Feature

class SlidingWindow(MongoModel):
    id: Optional[OID] = None
    data_windows: OID = Field(..., description="References to data windows (one SlidingWindowData document)")
    labels_of_data_windows: List[int]
    extracted_features: Dict[Feature, OID] = Field({}, description="Features extracted with this imputed data")

class ExtractedFeature(MongoModel):
    id: Optional[OID] = None
    data: OID = Field(..., description="References to the rows of extracted feature DataFrame (one ExtractedFeatureData document)")