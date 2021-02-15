from app.training.factory import get_normalizer, get_classifier, get_imputer
from app.util.ml_objects import IClassifier, INormalizer
from app.models.cached_data import ExtractedFeature, SlidingWindow
from typing import Dict, List, Tuple, Union
from ConfigSpace import Configuration
from motor.motor_asyncio import AsyncIOMotorDatabase
from pandas import DataFrame, concat
from tsfresh.feature_extraction import ComprehensiveFCParameters
import tsfresh
import pickle

from app.util.training_parameters import Classifier, Feature, Imputation, Normalizer
from app.models.mongo_model import OID
from app.models.workspace import DataPoint, Sample, Workspace


class Trainer():

    def __init__(self, db: AsyncIOMotorDatabase, workspace_id: OID, model_name: str, window_size: int, sliding_step: int, features: List[Feature], imputation: Imputation, 
                 normalizer: Normalizer, classifier: Classifier, hyperparameters: Configuration):
        # TODO wrapper for config?
        self.db = db
        self.progress = 0  # percentage
        self.workspace_id = workspace_id
        self.model_name = model_name
        self.imputation = imputation
        self.features = features
        self.normalizer = normalizer
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.window_size = window_size
        self.sliding_step = sliding_step

    async def train(self):
        # data split to windows
        pipeline_data = await self.__get_data_split_to_windows()

        # extract features
        pipeline_data = await self.__get_extracted_features(pipeline_data)

        # normalize
        normalizeTuple: Tuple[DataFrame,
                              INormalizer] = self.__normalize(pipeline_data)
        normalized_data: DataFrame = normalizeTuple[0]
        normalizer_object: INormalizer = normalizeTuple[1]
        print(normalized_data)

        # # train
        # classifier_object: IClassifier = self.__train(normalizedData)

        # # save ml_model

    async def __get_data_split_to_windows(self) -> SlidingWindow:
        workspace_document = await self.db.workspaces.find_one({"_id": self.workspace_id})
        workspace = Workspace(**workspace_document,
                              id=workspace_document["_id"])
        workspace_data = workspace.workspace_data
        # If already cached, retrieve from the database
        if str(self.window_size) + "_" + str(self.sliding_step) in workspace_data.sliding_windows:
            data_windows_id = workspace_data.sliding_windows[str(self.window_size) + "_" + str(self.sliding_step)]
            sliding_window_document = await self.db.sliding_windows.find_one({"_id": data_windows_id})
            return SlidingWindow(**sliding_window_document, id=sliding_window_document["_id"])
        # Split yourself and cache otherwise
        (data_windows, labels_of_data_windows) = self.__split_to_windows(
            workspace_data.samples, workspace_data.label_dict)
        data_windows_binary: List[bytes] = []
        for data_window in data_windows:
            data_windows_binary.append(pickle.dumps(data_window))
        sliding_window = SlidingWindow(data_windows=data_windows_binary, labels_of_data_windows=labels_of_data_windows)
        result = await self.db.sliding_windows.insert_one(sliding_window.dict(exclude={"id"}))
        sliding_window.id = result.inserted_id
        await self.db.workspaces.update_one({"_id": workspace.id}, {"$set": {"workspace_data.sliding_windows." + str(self.window_size) + "_" + str(self.sliding_step): result.inserted_id}})
        return sliding_window

    def __split_to_windows(self, workspace_data: List[Sample], label_dict: Dict[str, int]) -> Tuple[List[DataFrame], List[int]]:
        data_windows: List[DataFrame] = []
        labels_of_data_windows: List[int] = []
        for sample in workspace_data:
            # Length of data points for each sensor is same, so we take the first one
            length_of_data_points = len(list(sample.sensor_data_points.values())[0])
            # For each window in this sample, incremented by the sliding step value starting from 0
            for x in range(0, length_of_data_points, self.sliding_step):
                #  [time][sensor e.g acc_x]
                data_window: List[Dict[str, float]] = [{} for _range_ in range(self.window_size)]
                # For each sensor (sorted so that we always get the same column order)
                for sensor in sorted(sample.sensor_data_points):
                    # Data points of this sensor (time as index)
                    data_points: List[DataPoint] = sample.sensor_data_points[sensor]
                    # For each data point which is in this data window
                    for i in range(0, self.window_size):
                        # Append the values to this data window (index of values is e.g x in acc_x)
                        data_point_values = data_points[x+i].values
                        data_window[i].update({sensor+str(v): data_point_values[v]
                                               for v in range(len(data_point_values))})
                # Append this data window as a DataFrame to the list of all data windows
                data_windows.append(DataFrame(data_window))
                labels_of_data_windows.append(label_dict[sample.label])

        #TODO label 0 for the windows not in timeframes
        return (data_windows, labels_of_data_windows)

    async def __get_extracted_features(self, sliding_window: SlidingWindow) -> DataFrame:
        extracted_features_dict = sliding_window.extracted_features
        cached_extracted_features: Dict[Feature, DataFrame] = {}
        not_cached_features: List[Feature] = []
        # Sorted so that we always get the same column order
        for feature in self.features:
            if feature in extracted_features_dict:
                extracted_feature_doc = await self.db.extracted_features.find_one({"_id": extracted_features_dict[feature]})
                cached_extracted_features[feature] = pickle.loads(extracted_feature_doc["data"])  # Binary to bytes??
            else:
                not_cached_features.append(feature)

        newly_extracted_features: Dict[Feature, DataFrame]

        if not_cached_features:
            data_windows = [pickle.loads(data_window) for data_window in sliding_window.data_windows]
            newly_extracted_features = self.__extractFeatures(data_windows, not_cached_features)
            for feature in newly_extracted_features:
                extracted_feature = ExtractedFeature(data=pickle.dumps(newly_extracted_features[feature]))
                result = await self.db.extracted_features.insert_one(extracted_feature.dict(exclude={"id"}))
                await self.db.sliding_windows.update_one({"_id": sliding_window.id},
                                                         {"$set": {"extracted_features." + feature.value: result.inserted_id}})

        # Join them together
        sorted_dataframes: List[DataFrame] = []
        for feature in sorted(self.features):
            to_concat: DataFrame = None
            if feature in cached_extracted_features:
                to_concat = cached_extracted_features[feature]
            else:
                to_concat = newly_extracted_features[feature]
            sorted_dataframes.append(to_concat)
        return concat(sorted_dataframes, axis=1)

    def __extractFeatures(self, data_windows: List[DataFrame], features: List[Feature]) -> Dict[Feature, DataFrame]:
        # TODO parallelize maybe xd
        settings = {key.value: ComprehensiveFCParameters()[key.value] for key in features}
        newly_extracted_features: Dict[Feature, List[Dict[str, float]]] = {
            f: [{} for _range_ in range(len(data_windows))] for f in features}
        for data_window_index in range(len(data_windows)):
            data_window = data_windows[data_window_index]
            data_window["id"] = 0
            curr_extracted = tsfresh.extract_features(
                data_window, column_id="id", default_fc_parameters=settings, disable_progressbar=True).to_dict(orient="records")[0]
            extracted_keys = list(curr_extracted.keys())
            for key_index in range(len(extracted_keys)):
                curr_key = extracted_keys[key_index]
                newly_extracted_features[features[key_index %
                                                  len(features)]][data_window_index][curr_key] = curr_extracted[curr_key]

        for feature, data in newly_extracted_features.items():
            newly_extracted_features[feature] = DataFrame(data)
        return newly_extracted_features

    def __normalize(self, data: DataFrame) -> Tuple[DataFrame, INormalizer]:
        normalizer_object: INormalizer = get_normalizer(self.normalizer)
        normalizer_object.fit(data)
        normalized_data = DataFrame(normalizer_object.transform(data), columns=data.columnns)
        return (normalized_data, normalizer_object)

    def __train(self, data: DataFrame) -> IClassifier:
        pass
