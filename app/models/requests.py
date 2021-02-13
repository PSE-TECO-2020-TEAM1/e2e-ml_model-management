from app.util.classifier_config_spaces import get_config_space
from typing import List, Union
from pydantic import BaseModel, validator
from ConfigSpace import Configuration

from app.models.mongo_model import MongoModel, OID
from app.models.workspace import Sensor
from app.util.training_parameters import Classifier, Imputation, Feature, Normalizer 

class WorkspaceReq(MongoModel):
    workspaceId: OID

class PostCreateWorkspaceReq(WorkspaceReq):
    sensors: List[Sensor]

class PostTrainReq(WorkspaceReq):
    model_name: str
    imputation: Imputation
    features: List[Feature]
    normalizer: Normalizer
    classifier: Classifier
    hyperparameters: Configuration

    @validator("hyperparameters")
    def valid_hyperparameters(cls, v: Union[str, float, int], values):
        config_space = get_config_space(values["classifier"])
        config = Configuration(config_space, v)
        try:
            config.is_valid_configuration()
            return config
        except:
            raise ValueError("hyperparameters not valid")

class GetPredictionConfig(BaseModel):
    predectionID: str

class SubmitDataWindow(BaseModel):
    predectionID: str