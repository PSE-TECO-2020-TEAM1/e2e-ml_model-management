from typing import List
from app.models.mongo_model import MongoModel, OID
from app.models.workspace import Sensor
from pydantic import BaseModel

class WorkspaceReq(MongoModel):
    workspaceId: OID

class PostCreateWorkspaceReq(WorkspaceReq):
    sensors: List[Sensor]

############################################################################
#class GetModelsReq():
    #workspaceID: str    #no need ? 

class Hyperparameter():
    name: str
    value: int     # is value an int ? 

class Classifier():
    name: str
    Hyperparameters : list[Hyperparameter]

class TrainReq(BaseModel):
    model_name: str
    imputation: str
    features: list[str]
    normalizer: str
    classifier: Classifier

#class GetTrainingProgress():
  #  workspaceID: str 

#class GetModel():
  #  workspaceID: str  
  # modelID: str   

#class GetPredictionID():
   # workspaceID: str  
   # modelID: str   

class GetPredictionConfig(BaseModel):
    predectionID: str  

class SubmitDataWindow(BaseModel):
    predectionID: str      