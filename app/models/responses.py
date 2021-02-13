from typing import Optional
from pydantic import BaseModel 

class Hyperparameters():
    name: str
    format: str

class Classifiers():
    name: str
    Hyperparameters : list[Hyperparameters]

class GetParameterRes(BaseModel):
    imputation: list[str]
    features: list[str]
    normalizers: list[str]
    classifiers: Classifiers

class GetModelsRes(BaseModel):
    modelID: str
    name: str

class TrainingProgressRes(BaseModel):
    progress: float

class PerformanceMetrics():
    name: str
    score: int

class LabelPerformanceMetrics():
    label: str
    PerformanceMetrics: list[PerformanceMetrics]

class Parameters():
    imputation: str
    features: list[str]
    normalizer: str
    classifier: Classifiers

class ModelRes(BaseModel):
    labelPerformanceMetrics: list[LabelPerformanceMetrics]
    parameters: Parameters

class GetPredictionIdRes(BaseModel):
    predictionID: str

class PredictionConfig():
    name: list[str]
    samplingRate: int

class PredictionConfigRes(BaseModel):
    predictionConfig: PredictionConfig

class DataPoints():
    timeStamp: int
    data: list[int]

class SensorDataPoint():
    sensor: str
    dataPoints: list[DataPoints]

class submitDataWindowRes(BaseModel):
    start: int
    end: int
    sensortDataPionts: list[SensorDataPoint] 
