from app.util.ml_objects import IClassifier, INormalizer
from bson.binary import Binary
from app.models.cached_data import ExtractedFeature, SlidingWindow
from typing import Dict, List, Tuple
from ConfigSpace import Configuration
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.util.training_parameters import Classifier, Feature, Imputation, Normalizer
from app.models.mongo_model import OID
from app.models.workspace import DataPoint, Sample, WorkspaceData
from pandas import DataFrame
import pickle


class Trainer():

    def __init__(self, db: AsyncIOMotorDatabase, workspaceID: OID, model_name: str, imputation: Imputation, features: List[Feature],
                 normalizer: Normalizer, classifier: Classifier, window_size: int, sliding_step: int, hyperparameters: Configuration):
        # TODO wrapper ?
        self.db = db
        self.progress = 0  # percentage
        self.workspaceId = workspaceID
        self.model_name = model_name
        self.imputation = imputation
        self.features = features
        self.normalizer = normalizer
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.window_size = window_size
        self.sliding_step = sliding_step

    def train(self):
        # split to data windows
        sliding_windows_dict: Dict[Tuple[int, int], OID] = self.db.workspaces.find_one(
            {"_id": self.workspaceId}, {"workspace_data.sliding_windows": True}).sliding_windows
        sliding_window: SlidingWindow

        if (self.window_size, self.sliding_step) in sliding_windows_dict:
            data_windows_id = sliding_windows_dict[(
                self.window_size, self.sliding_step)]
            sliding_window = self.db.sliding_windows.find_one(
                {"_id": data_windows_id})
        else:
            samples: List[Sample] = self.db.workspaces.find_one(
                {"_id": self.workspaceId}, {"samples": True}).samples
            data_windows: List[DataFrame] = self.__splitToWindows(samples)
            data_windows_binary: List[bytes]
            for data_window in data_windows:
                data_windows_binary.append(pickle.dumps(data_window))
            sliding_window = SlidingWindow(data_windows_binary)
            self.db.sliding_windows.insert_one(sliding_window.dict())

        # extract features
        extracted_features_dict: Dict[Feature,
                                      OID] = sliding_window.extracted_features
        extracted_features: DataFrame = DataFrame()
        data: List[Binary] = sliding_window.data
        for feature in self.features:
            if feature in extracted_features_dict:
                extracted_feature: ExtractedFeature = self.db.extracted_features.find_one(
                    {"_id": extracted_features_dict[feature]})
                extracted_features.join(pickle.loads(
                    extracted_feature.data))  # Binary to bytes??
            else:
                extracted_data: DataFrame = self.__extractFeature(
                    data, feature)
                result = self.db.extracted_features.insert_one(
                    pickle.dumps(extracted_data))
                self.db.sliding_windows.update_one({"_id": sliding_window._id}, {
                                                   "$set": {"extracted_features." + feature.value: result.inserted_id}})
                extracted_features.join(extracted_data)

        # normalize

        normalizeTuple: Tuple[DataFrame, INormalizer] = self.__normalize(
            extracted_feature)
        normalizedData: DataFrame = normalizeTuple[0]
        normalizer_object: INormalizer = normalizeTuple[1]

        # train

        classifier_object: IClassifier = self.__train(normalizedData)

        # save ml_model

        pass

    # def __impute()
    #     pass

    def __splitToWindows(self, data: List[Sample]) -> List[DataFrame]:
        label_dict: Dict[str, int] = self.db.workspaces.find_one(
            {"_id": self.workspaceId}, {"label_dict": True}).label_dict
        data_windows: List[DataFrame] = []

        for sample in data:
            # Length of data points for each sensor is same, so we take the first one
            length_of_data_points = len(sample.sensor_data_points.values[0])
            # For each window in this sample, incremented by the sliding step value starting from 0
            for x in range(start=0, stop=length_of_data_points, step=self.sliding_step):
                #  [time][sensor e.g acc_x]
                data_window: List[List[int]] = [[]
                                                for x in range(self.window_size)]
                # For each sensor (sorted so that we always get the same column order)
                for sensor in sorted(sample.sensor_data_points):
                    # Data points of this sensor (time as index)
                    data_points: List[DataPoint] = sample.sensors_data_points[sensor]
                    # For each data point which is in this data window
                    for i in range(start=x, stop=x+self.window_size):
                        # Append the values to this data window (index of values is e.g x in acc_x)
                        data_window[i].append(x for x in data_points[i].values)
                # Append this data window as a DataFrame to the list of all data windows
                data_windows.append(DataFrame(data_window))

        return data_windows

    def __extractFeature(self, data: List[DataFrame], feature: Feature) -> DataFrame:
        pass

    def __normalize(self, data: DataFrame) -> Tuple[DataFrame, INormalizer]:
        pass

    def __train(self, data: DataFrame) -> IClassifier:
        pass
