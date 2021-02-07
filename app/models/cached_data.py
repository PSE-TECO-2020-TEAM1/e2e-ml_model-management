from typing import List, Optional
from pydantic import Field
from bson import Binary

from app.util.data_processing_parameters import Imputation, Feature
from app.models.mongo_model import OID, MongoModel

class SlidingWindow(MongoModel):
    _id: Optional[OID] = None
    window_size: int
    sliding_step: int
    data: Binary = Field(..., description="DataFrame")
    imputedData: List[OID] = Field([], description="Data imputed with this sliding window parameters")

class ImputedData(MongoModel):
    _id: Optional[OID] = None
    imputation: Imputation
    data: Binary = Field(..., description="DataFrame")
    imputerObject: Binary = Field(..., description="IImputer")
    extractedFeatures: List[OID] = Field([], description="Features extracted with this imputed data")

class ExtractedFeature(MongoModel):
    _id: Optional[OID] = None
    feature: Feature
    data: Binary = Field(..., description="DataFrame")