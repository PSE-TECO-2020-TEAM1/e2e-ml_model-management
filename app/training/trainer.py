from app.util.ml_objects import IClassifier, INormalizer
from app.models.cached_data import ExtractedFeature, SlidingWindow
from typing import Dict, List, Tuple, Union
from ConfigSpace import Configuration
from motor.motor_asyncio import AsyncIOMotorDatabase
from pandas import DataFrame
import pickle

from app.util.training_parameters import Classifier, Feature, Imputation, Normalizer
from app.models.mongo_model import OID
from app.models.workspace import DataPoint, Sample, WorkspaceData


class Trainer():

    def __init__(self, db: AsyncIOMotorDatabase, workspaceID: OID, model_name: str, imputation: Imputation, features: List[Feature],
                 normalizer: Normalizer, classifier: Classifier, window_size: int, sliding_step: int, hyperparameters: Configuration):
        # TODO wrapper for config?
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
        # data split to windows
        pipeline_data = self.__get_data_split_to_windows()

        # extract features
        pipeline_data = self.__get_extracted_features(pipeline_data)

        # normalize
        normalizeTuple: Tuple[DataFrame, INormalizer] = self.__normalize(extracted_feature)
        normalizedData: DataFrame = normalizeTuple[0]
        normalizer_object: INormalizer = normalizeTuple[1]

        # train

        classifier_object: IClassifier = self.__train(normalizedData)

        # save ml_model

        pass

    def __get_data_split_to_windows(self) -> SlidingWindow:
        workspace_data: WorkspaceData = self.db.workspaces.find_one({"_id": self.workspaceId})
        # If already cached, retrieve from the database
        if (self.window_size, self.sliding_step) in workspace_data.sliding_windows:
            data_windows_id = workspace_data.sliding_windows[(self.window_size, self.sliding_step)]
            return self.db.sliding_windows.find_one({"_id": data_windows_id})
        # Split yourself and cache otherwise
        data_windows = self.__split_to_windows(workspace_data.samples, workspace_data.label_dict)
        data_windows_binary: List[bytes]
        for data_window in data_windows:
            data_windows_binary.append(pickle.dumps(data_window))
        sliding_window = SlidingWindow(data_windows_binary)
        self.db.sliding_windows.insert_one(sliding_window.dict())
        return sliding_window

    def __split_to_windows(self, workspace_data: List[Sample], label_dict: Dict[str, int]) -> List[DataFrame]:
        # TODO labeldict ????? shouldve use here
        data_windows: List[DataFrame] = []

        for sample in workspace_data:
            # Length of data points for each sensor is same, so we take the first one
            length_of_data_points = len(sample.sensor_data_points.values[0])
            # For each window in this sample, incremented by the sliding step value starting from 0
            for x in range(start=0, stop=length_of_data_points, step=self.sliding_step):
                #  [time][sensor e.g acc_x]
                data_window: List[List[int]] = [[] for x in range(self.window_size)]
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

    #TODO we are kinda rekt here
    def __get_extracted_features(self, sliding_window: SlidingWindow) -> DataFrame:
        extracted_features_dict = sliding_window.extracted_features
        extracted_features: List[Union[Feature, DataFrame]] = []
        have_non_cached_feature: bool = False
        # Sorted so that we always get the same column order
        for feature in sorted(self.features):
            if feature in extracted_features_dict:
                extracted_feature_doc: ExtractedFeature = self.db.extracted_features.find_one({"_id": extracted_features_dict[feature]})
                extracted_features.append(pickle.loads(extracted_feature_doc.data))  # Binary to bytes??
            else:
                have_non_cached_feature = True
                extracted_features.append(Feature)
        # Load the data_windows only if we don't have all the features already
        if have_non_cached_feature:
            data_windows = [pickle.loads(data_window) for data_window in sliding_window.data_windows]
            for i in range(len(extracted_features)):
                if isinstance(extracted_features[i], Feature):
                    extracted_feature = self.__extractFeature(data_windows, extracted_features[i])
                    # Cache the extracted feature for future use
                    result = self.db.extracted_features.insert_one(pickle.dumps(extracted_feature))
                    self.db.sliding_windows.update_one({"_id": sliding_window._id}, {"$set": {"extracted_features." + feature.value: result.inserted_id}})
        # Join them together
        result_dataframe = DataFrame()
        for extracted_feature in extracted_features:
            result_dataframe.join(extracted_feature)
        return result_dataframe


    def __extractFeature(self, data: List[DataFrame], feature: Feature) -> DataFrame:
        pass

    def __normalize(self, data: DataFrame) -> Tuple[DataFrame, INormalizer]:
        pass

    def __train(self, data: DataFrame) -> IClassifier:
        pass
