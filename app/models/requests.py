from typing import Dict, List, Union

from app.models.mongo_model import OID, MongoModel
from app.models.workspace import Sensor
from app.util.classifier_config_spaces import get_config_space
from app.util.training_parameters import (Classifier, Feature, Imputation,
                                          Normalization)
from ConfigSpace import Configuration
from pydantic import BaseModel, validator


class WorkspaceReq(MongoModel):
    workspace_id: OID


class PostCreateWorkspaceReq(WorkspaceReq):
    sensors: List[Sensor]


class PostTrainReq(WorkspaceReq):
    model_name: str
    imputation: Imputation
    features: List[Feature]
    normalizer: Normalization
    classifier: Classifier
    hyperparameters: Dict[str, Union[str, float, int, bool]]

    @validator("hyperparameters")
    def valid_hyperparameters(cls, v, values):
        # Check if v is legal configuration, raise an error if so
        config_space = get_config_space(values["classifier"])
        config = Configuration(config_space, v)
        # Parse strings to actual types (strings are required by ConfigSpace module ¯\_(ツ)_/¯ )
        for key, value in v.items():
            if value == "None":
                v[key] = None
            elif value == "True":
                v[key] = True
            elif value == "False":
                v[key] = False
        return config

class GetPredictionConfig(BaseModel):
    predectionID: str


class SubmitDataWindow(BaseModel):
    predectionID: str
