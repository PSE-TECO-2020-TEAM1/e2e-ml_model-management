from typing import Any, Dict, List

from pydantic.fields import Field

from app.models.mongo_model import OID, MongoModel
from app.models.workspace import DataPoint, Sensor
from app.util.classifier_config_spaces import config_spaces
from app.util.training_parameters import (Classifier, Feature, Imputation,
                                          Normalization)
from ConfigSpace import Configuration
from pydantic import validator


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
        # Check if v is legal configuration, raise an error if so
        config_space = config_spaces[values["classifier"]]
        config = Configuration(config_space, v)
        # Parse strings to actual types ( strings are required by the ConfigSpace module ¯\_(ツ)_/¯ )
        for key, value in v.items():
            if value == "None":
                v[key] = None
            elif value == "True":
                v[key] = True
            elif value == "False":
                v[key] = False
        return config

class PostSubmitDataReq(MongoModel):
    predictionId: str
    dataPointCount: int #TODO we can also len first value in data. Should we also check if the values in data has the same len ??
    data: Dict[str, List[DataPoint]]
