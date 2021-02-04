from typing import Optional

class GetModelsReq():
    workspaceID: str    #no need ? 

class Hyperparameter():
    name: str
    value: int     # is value an int ? 

class Classifier():
    name: str
    Hyperparameters : list[Hyperparameter]
class TrainReq():
    model_name: str
    imputation: str
    features: list[str]
    normalizer: str
    classifier: Classifier

class GetTrainingProgress():
    workspaceID: str 

class GetModel():
    workspaceID: str  
    modelID: str   

class GetPredictionID():
    workspaceID: str  
    modelID: str   

class GetPredictionConfig():
    predectionID: str  

class SubmitDataWindow():
    predectionID: str      