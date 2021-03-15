from typing import Dict, List
from pydantic import Field

from app.models.mongo_model import OID, MongoModel
from app.util.training_parameters import Feature

class SlidingWindow(MongoModel):
    id: OID = Field(None, alias="_id")
    dataWindows: OID = Field(..., description="References to data windows (one SlidingWindowData document)")
    labelsOfDataWindows: List[str]
    extractedFeatures: Dict[Feature, OID] = Field({}, description="Features extracted with this imputed data")

class ExtractedFeature(MongoModel):
    id: OID = Field(None, alias="_id")
    data: OID = Field(..., description="References to the rows of extracted feature DataFrame (one ExtractedFeatureData document)")