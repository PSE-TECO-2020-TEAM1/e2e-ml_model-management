from app.ml.util.sample_parsing import parse_sample_in_predict
from app.workspace_management_api.sample_model import Timeframe
from app.ml.prediction.data_set_manager import DataSetManager
from collections import deque
from multiprocessing import set_start_method
from pandas.core.indexes.base import Index
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.connection import Connection
from pandas import DataFrame
from typing import Deque, Dict, List
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from dataclasses import dataclass


@dataclass
class PredictionResult():
    labels: List[str]
    timeframe: Timeframe

class Predictor():
    def __init__(self, data_set_manager: DataSetManager):
        self.data_set_manager = data_set_manager

    def init_predictor_process(self, semaphore: SemaphoreType, predictor_end: Connection):
        set_start_method("fork", force=True)
        workspace_sensors = self.data_set_manager.get_workspace_sensors()
        sliding_window = self.data_set_manager.get_sliding_window()
        label_encoder = self.data_set_manager.get_label_encoder()
        pipeline = self.data_set_manager.get_pipeline()
        prediction_df = DataFrame()
        while True:
            while not semaphore.acquire():
                pass
            data = predictor_end.recv()
            parse_sample_in_predict(data, workspace_sensors)
            predictor_end.send(PredictionResult(#TODO))