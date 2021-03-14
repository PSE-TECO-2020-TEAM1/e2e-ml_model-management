from typing import Any, Dict, List
from ConfigSpace import Configuration
from pydantic import validator
from pydantic.fields import Field

from app.models.mongo_model import OID, MongoModel
from app.models.workspace import DataPoint, Sensor
from app.util.classifier_config_spaces import config_spaces
from app.util.training_parameters import (Classifier, Feature, Imputation,
                                          Normalization)


class PostCreateWorkspaceReq(MongoModel):
    workspaceId: OID
    sensors: List[Sensor]

class PostTrainReq(MongoModel):
    modelName: str
    imputation: Imputation
    features: List[Feature]
    normalizer: Normalization
    classifier: Classifier
    windowSize: int = Field(..., ge=4)
    slidingStep: int = Field(..., ge=1)
    hyperparameters: Dict[str, Any]

    @validator("hyperparameters")
    def valid_hyperparameters(cls, v, values):
        # Check if v is legal configuration, raise an error if not
        for key, value in v.items():
            if type(value) == str and value.isnumeric():
                v[key] = int(value)
        config_space = config_spaces[values["classifier"]]
        Configuration(config_space, v)
        # Parse strings to actual types ( strings are required by the ConfigSpace module ¯\_(ツ)_/¯ )
        for key, value in v.items():
            if value == "None":
                v[key] = None
            elif value == "True":
                v[key] = True
            elif value == "False":
                v[key] = False
        return v

class PostSubmitDataReq(MongoModel):
    predictionId: str
    # We don't want pydantic to validate the data points, as the data is potentially huge and the validation blocks the event loop.
    # We validate it later in its own predictor process
    data: Dict[str, List[DataPoint]]
    start: int
    end: int
