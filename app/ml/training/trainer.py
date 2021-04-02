from app.ml.objects.pipeline_factory import make_pipeline_from_config
from app.ml.training.parameters.features import Feature
from app.models.domain.sensor import SensorComponent
from typing import Callable, Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder
from pandas.core.frame import DataFrame
from pymongo.database import Database
from app.models.domain.training_config import TrainingConfig
from app.db.sync.training_service import TrainingService
from sklearn.model_selection import train_test_split
import pandas as pd

class Trainer():

    def __init__(self, training_config: TrainingConfig, training_service: TrainingService, get_db: Callable[[], Database]):
        self.feature_extraction_config = training_config.feature_extraction_config
        self.pipeline_config = training_config.pipeline_config
        self.classifier = training_config.classifier
        self.hyperparameters = training_config.hyperparameters
        self.training_service = training_service
        self.get_db = get_db

    def setup(self):
        db = self.get_db()
        self.training_service.set_db(db)
        self.training_service.update_training_data_set()

    def train(self):
        self.setup()
        x, y = self.gather_features_and_labels()
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        pipeline = make_pipeline_from_config(self.pipeline_config, self.classifier, self.hyperparameters)
        pipeline.fit(x_train, y_train)


    def gather_features_and_labels(self) -> Tuple[DataFrame, List[str]]:
        result = []
        to_be_calculated: Dict[SensorComponent, List[Feature]] = {}
        for sensor_component, features in self.feature_extraction_config.sensor_component_features.items():
            uncached_features = []
            for feature in features:
                if self.training_service.is_cached_sensor_component_feature(sensor_component, feature):
                    result.append(self.training_service.get_sensor_component_feature(sensor_component, feature))
                else:
                    uncached_features.append(feature)
            to_be_calculated[sensor_component] = uncached_features
        if to_be_calculated:
            result += self.extract_sensor_component_features(to_be_calculated)
        labels = self.training_service.get_labels_of_data_windows(self.feature_extraction_config.sliding_window)
        return (pd.concat(result, axis=1), labels)

    def extract_sensor_component_features(self, sensor_component_features: Dict[SensorComponent, List[Feature]]) -> List[DataFrame]:
        sliding_window = self.feature_extraction_config.sliding_window
        data_windows: DataFrame
        # Check if data windows are cached
        if self.training_service.is_cached_split_to_windows(sliding_window):
            data_windows = self.training_service.get_cached_split_to_windows(sliding_window)
        else:
            # Compute and add to cache otherwise along with the labels
            data_windows, labels_of_data_windows = split_to_data_windows(sliding_window, self.training_service.get_sample_list())
            self.training_service.add_split_to_windows(sliding_window, data_windows, labels_of_data_windows)
        result = []
        for sensor_component, features in sensor_component_features.items():
            features = extract_features(data_windows[[sensor_component]], features)
            for feature in features:
                self.training_service.add_sensor_component_feature(sensor_component, feature, features[feature])
            result.append(features.values())
        return result