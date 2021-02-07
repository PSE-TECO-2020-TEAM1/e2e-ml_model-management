from typing import Optional
 

class Hyperparameter():
    name: str
    format: str

class Classifiers():
    name: str
    Hyperparameters : list[Hyperparameter]

class GetParameterRes():
    imputation: list[str]
    features: list[str]
    normalizers: list[str]
    classifiers: Classifiers

class GetModelsRes():
    modelID: str
    name: str

class TrainingProgress():
    Progress: int

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

class ModelRes():
    labelPerformanceMetrics: list[LabelPerformanceMetrics]
    parameters: Parameters

class PredictionIDRes():
    predictionID: str

class PredictionConfig():
    name: list[str]
    samplingRate: int

class PredictionConfigRes():
    predictionConfig: PredictionConfig

class DataPoints:
    timeStamp: int
    data: list[int]

class SensorDataPoint:
    sensor: str
    dataPoints: list[DataPoints]

class submitDataWindowRes():
    start: int
    end: int
    sensortDataPionts: list[SensorDataPoint] 
