from typing import Dict, List, Optional
from pydantic import Field
from bson import Binary

from app.util.training_parameters import Imputation, Feature
from app.models.mongo_model import OID, MongoModel

#class ImputedSamples(MongoModel):
#    _id: Optional[OID] = None
#    samples: List[Binary] = Field(..., description="DataFrame")
#    imputerObject: Binary = Field(..., description="IImputer")
#    slidingWindows: List[OID] = Field([], description="Data imputed with this sliding window parameters")

class SlidingWindow(MongoModel):
    _id: Optional[OID] = None
    data_windows: List[Binary] = Field(..., description="DataFrame[]")
    labels_of_data_windows: List[int]
    extracted_features: Dict[Feature, OID] = Field([], description="Features extracted with this imputed data")

class ExtractedFeature(MongoModel):
    _id: Optional[OID] = None
    data: Binary = Field(..., description="DataFrame")