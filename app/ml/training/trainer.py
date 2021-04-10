from app.ml.training.training_state import TrainingState
from app.ml.util.data_processing import calculate_classification_report, extract_features, split_to_data_windows
from pandas.core.frame import DataFrame
from app.ml.objects.feature.enum import Feature
from app.models.domain.sensor import SensorComponent
from multiprocessing import set_start_method
from app.ml.objects.pipeline_factory import make_pipeline_from_config
from typing import Callable, Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from pymongo.database import Database
from app.models.domain.training_config import TrainingConfig
from app.ml.training.data_set_manager import DataSetManager
from sklearn.model_selection import train_test_split
import pandas as pd


class Trainer():

    def __init__(self, training_config: TrainingConfig, data_set_manager: DataSetManager, create_db: Callable[[], Database]):
        self.training_config = training_config
        self.data_set_manager = data_set_manager
        self.create_db = create_db

    def setup(self):
        db = self.create_db()
        self.data_set_manager.set_db(db)
        self.data_set_manager.update_training_data_set()

    def train(self):
        set_start_method("fork", force=True)
        self.setup()
        try:
            #------------------------------------FEATURE_EXTRACTION-----------------------------------#
            self.data_set_manager.set_training_state(TrainingState.FEATURE_EXTRACTION)
            # Get the data frame ready for pipeline
            x, y = self.gather_features_and_labels()

            #--------------------------------------MODEL_TRAINING-------------------------------------#
            self.data_set_manager.set_training_state(TrainingState.MODEL_TRAINING)
            # We have to sort the columns correctly when we are predicting later so we save the order
            columns = x.columns.tolist()
            # Encode labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            # Train-test-split
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
            # Create the pipeline
            pipeline = make_pipeline_from_config(self.training_config.get_component_pipeline_configs(),
                                                 self.training_config.classifier, self.training_config.hyperparameters)
            # Fit the model
            pipeline.fit(x_train, y_train)

            #------------------------------------CLASSIFICATION_REPORT--------------------------------#
            self.data_set_manager.set_training_state(TrainingState.CLASSIFICATION_REPORT)
            # Calculate performance metrics
            test_prediction_result = pipeline.predict(x_test)
            performance_metrics = calculate_classification_report(test_prediction_result, y_test, label_encoder)
            self.data_set_manager.save_model(
                config=self.training_config,
                label_performance_metrics=performance_metrics,
                column_order=columns,
                label_encoder=label_encoder,
                pipeline=pipeline
            )
        finally:
            # We want to set the training state back to no active training no matter what
            # so that an internal error does not block the whole workspace
            self.data_set_manager.set_training_state(TrainingState.NO_ACTIVE_TRAINING)

    def gather_features_and_labels(self) -> Tuple[DataFrame, List[str]]:
        sliding_window = self.training_config.sliding_window
        result: List[DataFrame] = []  # data windows of one sensor component and one feature (only one column)
        to_be_calculated: Dict[SensorComponent, List[Feature]] = {}
        for sensor_component, features in self.training_config.get_component_features().items():
            uncached_features = []
            for feature in features:
                if self.data_set_manager.is_cached_sensor_component_feature(sliding_window, sensor_component, feature):
                    result.append(self.data_set_manager.get_cached_sensor_component_feature(sliding_window, sensor_component, feature))
                else:
                    uncached_features.append(feature)
            if uncached_features:
                to_be_calculated[sensor_component] = uncached_features
        if to_be_calculated:
            result += self.extract_sensor_component_features(to_be_calculated)
        labels = self.data_set_manager.get_labels_of_data_windows(self.training_config.sliding_window)
        return (pd.concat(result, axis=1), labels)

    def extract_sensor_component_features(self, sensor_component_features: Dict[SensorComponent, List[Feature]]) -> List[DataFrame]:
        sliding_window = self.training_config.sliding_window
        data_windows: DataFrame
        # Check if data windows are cached
        if self.data_set_manager.is_cached_split_to_windows(sliding_window):
            data_windows = self.data_set_manager.get_cached_split_to_windows(sliding_window)
        else:
            # Compute and add to cache otherwise along with the labels
            data_windows, labels_of_data_windows = split_to_data_windows(sliding_window, self.data_set_manager.get_sample_list())
            self.data_set_manager.add_split_to_windows(sliding_window, data_windows, labels_of_data_windows)
        result = []  # data windows of one sensor component and one feature (only one column)
        for sensor_component, features in sensor_component_features.items():
            features = extract_features(data_windows[[sensor_component, "id"]], features)
            for feature in features:
                self.data_set_manager.add_sensor_component_feature(sliding_window, sensor_component, feature, features[feature])
            result += list(features.values())
        return result
