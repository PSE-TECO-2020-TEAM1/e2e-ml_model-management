from pandas.core.indexes.base import Index
from app.ml.util.data_processing import extract_features, roll_data_frame
from app.ml.util.sample_parsing import parse_sample_in_predict
from app.ml.prediction.data_set_manager import DataSetManager
from multiprocessing import set_start_method
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.connection import Connection
from pandas import DataFrame
import pandas as pd

class Predictor():
    def __init__(self, data_set_manager: DataSetManager):
        self.data_set_manager = data_set_manager

    def init_predictor_process(self, semaphore: SemaphoreType, predictor_end: Connection):
        set_start_method("fork", force=True)
        workspace_sensors = self.data_set_manager.get_workspace_sensors()
        feature_extraction_config = self.data_set_manager.get_feature_extraction_config()
        column_index = self.data_set_manager.get_column_order()
        label_encoder = self.data_set_manager.get_label_encoder()
        pipeline = self.data_set_manager.get_pipeline()
        prediction_df = DataFrame()
        while True:
            while not semaphore.acquire():
                pass
            data = predictor_end.recv()
            parsed_df = parse_sample_in_predict(data, workspace_sensors)
            prediction_df = pd.concat(prediction_df, parsed_df, ignore_index=True)
            if len(prediction_df.index) < feature_extraction_config.sliding_window.window_size:
                # There are not enough data to do prediction
                continue
            data_windows = roll_data_frame(feature_extraction_config.sliding_window, prediction_df)

            # We want to spare the leftover, so that we can count it in the window when the next sample arrives
            leftover = feature_extraction_config.sliding_window.window_size - feature_extraction_config.sliding_window.sliding_step
            prediction_df = prediction_df.iloc[len(prediction_df.index) - leftover:, :]

            all_feature_dfs = []
            for sensor_component, features in feature_extraction_config.sensor_component_features.items():
                all_feature_dfs.append(extract_features(data_windows[[sensor_component, "id"]], features).values())
            x = pd.concat(all_feature_dfs, axis=1, ignore_index=True)[Index(column_index)]
            predictions = pipeline.predict(x)
            translated_labels = label_encoder.inverse_transform(predictions)
            predictor_end.send(translated_labels)