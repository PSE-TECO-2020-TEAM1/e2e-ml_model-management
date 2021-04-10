from typing import Callable

from pymongo.database import Database
from app.models.schemas.prediction_data import SampleInPredict
from pandas.core.indexes.base import Index
from app.ml.util.data_processing import extract_features, roll_data_frame
from app.ml.util.sample_parsing import parse_sensor_data_points_in_predict
from app.ml.prediction.data_set_manager import DataSetManager
from multiprocessing import set_start_method
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.connection import Connection
from pandas import DataFrame
import pandas as pd

class Predictor():
    def __init__(self, data_set_manager: DataSetManager, create_db: Callable[[], Database]):
        self.data_set_manager = data_set_manager
        self.create_db = create_db

    def init_predictor_process(self, semaphore: SemaphoreType, predictor_end: Connection):
        set_start_method("fork", force=True)
        self.data_set_manager.set_db(self.create_db())
        workspace_sensors = self.data_set_manager.get_workspace_sensors()
        sliding_window = self.data_set_manager.get_sliding_window()
        component_features = self.data_set_manager.get_component_features()
        column_index = self.data_set_manager.get_column_order()
        label_encoder = self.data_set_manager.get_label_encoder()
        pipeline = self.data_set_manager.get_pipeline()
        prediction_df = DataFrame()
        while True:
            while not semaphore.acquire():
                pass
            data: SampleInPredict = predictor_end.recv()
            parsed_df = parse_sensor_data_points_in_predict(data, workspace_sensors)
            prediction_df = pd.concat([prediction_df, parsed_df], ignore_index=True)
            if len(prediction_df.index) < sliding_window.window_size:
                # There are not enough data to do prediction
                continue
            data_windows = roll_data_frame(sliding_window, prediction_df)

            # We want to spare the leftover, so that we can count it in the window when the next sample arrives
            leftover = sliding_window.window_size - sliding_window.sliding_step
            prediction_df = prediction_df.iloc[len(prediction_df.index) - leftover:, :]
            all_feature_dfs = []
            for sensor_component, features in component_features.items():
                all_feature_dfs += list(extract_features(data_windows[[sensor_component, "id"]], features).values())
            x = pd.concat(all_feature_dfs, axis=1)[Index(column_index)]
            predictions = pipeline.predict(x)
            translated_labels = list(label_encoder.inverse_transform(predictions))
            predictor_end.send(translated_labels)