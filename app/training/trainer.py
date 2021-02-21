import pickle
from typing import Any, Dict, List, Tuple
import tsfresh
from pandas import DataFrame, concat
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.metrics import classification_report
from pymongo import MongoClient

from app.models.ml_model import ML_Model, PerformanceMetrics, PerformanceMetricsPerLabel
from app.models.cached_data import ExtractedFeature, SlidingWindow
from app.models.mongo_model import OID
from app.models.workspace import DataPoint, Sample, Workspace
from app.training.factory import get_classifier, get_imputer, get_normalizer
from app.util.ml_objects import IClassifier, IImputer, INormalizer
from app.util.training_parameters import (Classifier, Feature, Imputation,
                                          Normalization)


class Trainer():

    def __init__(self, workspace_id: OID, model_name: str, window_size: int, sliding_step: int, features: List[Feature], imputation: Imputation,
                 normalizer: Normalization, classifier: Classifier, hyperparameters: Dict[str, Any]):
        # TODO wrapper for config?
        self.client = None
        self.db = None

        self.progress: int = 0  # percentage
        self.workspace_id = workspace_id
        self.model_name = model_name
        self.imputation = imputation
        self.features = features
        self.normalizer = normalizer
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.window_size = window_size
        self.sliding_step = sliding_step


    def train(self):
        #TODO database client
        self.client = MongoClient("mongodb://0.0.0.0/", 27017)
        self.db = self.client["test"]

        workspace_document = self.db.workspaces.find_one({"_id": self.workspace_id})
        workspace = Workspace(**workspace_document, id=workspace_document["_id"])

        # data split to windows
        pipeline_data = self.__get_data_split_to_windows(workspace)
        labels_of_data_windows = pipeline_data.labels_of_data_windows

        # extract features
        pipeline_data = self.__get_extracted_features(pipeline_data)

        # train-test split
        train_data = pipeline_data.iloc[:int(pipeline_data.shape[0]*0.8)]
        train_labels = labels_of_data_windows[:int(len(labels_of_data_windows)*0.8)]
        test_data = pipeline_data.iloc[int(pipeline_data.shape[0]*0.8):]
        test_labels = labels_of_data_windows[int(len(labels_of_data_windows)*0.8):]

        del pipeline_data
        del labels_of_data_windows

        # impute
        (train_data, test_data, imputer_object) = self.__impute(train_data, test_data)

        # normalize
        (train_data, test_data, normalizer_object) = self.__normalize(train_data, test_data)

        # train
        classifier_object = self.__train(train_data, train_labels)

        # performance metrics
        performance_metrics = self.__get_performance_metrics(
            classifier_object, test_data, test_labels, workspace.workspace_data.label_code_to_label)

        # save ml_model
        ml_model = ML_Model(name=self.model_name, workspace_id=self.workspace_id, window_size=self.window_size, sliding_step=self.sliding_step, label_performance_metrics=performance_metrics,
                            imputer_object=pickle.dumps(imputer_object), normalizer_object=pickle.dumps(normalizer_object), classifier_object=pickle.dumps(classifier_object), hyperparameters=self.hyperparameters)

        result = self.db.ml_models.insert_one(ml_model.dict(exclude={"id"}))
        self.db.workspaces.update_one({"_id": workspace.id}, {"$push": {"ml_models": result.inserted_id}})


    def __get_data_split_to_windows(self, workspace: Workspace) -> SlidingWindow:
        #TODO handle timeframes
        workspace_data = workspace.workspace_data
        # If already cached, retrieve from the database
        if str(self.window_size) + "_" + str(self.sliding_step) in workspace_data.sliding_windows:
            data_windows_id = workspace_data.sliding_windows[str(self.window_size) + "_" + str(self.sliding_step)]
            sliding_window_document = self.db.sliding_windows.find_one({"_id": data_windows_id})
            return SlidingWindow(**sliding_window_document, id=sliding_window_document["_id"])
        # Split yourself and cache otherwise
        (data_windows, labels_of_data_windows) = self.__split_to_windows(
            workspace_data.samples, workspace_data.label_to_label_code)
        data_windows_binary: List[bytes] = []
        for data_window in data_windows:
            data_windows_binary.append(pickle.dumps(data_window))
        sliding_window = SlidingWindow(data_windows=data_windows_binary, labels_of_data_windows=labels_of_data_windows)
        result = self.db.sliding_windows.insert_one(sliding_window.dict(exclude={"id"}))
        sliding_window.id = result.inserted_id
        self.db.workspaces.update_one({"_id": workspace.id}, {
            "$set": {"workspace_data.sliding_windows." + str(self.window_size) + "_" + str(self.sliding_step): result.inserted_id}})
        return sliding_window

    def __split_to_windows(self, samples: List[Sample], label_to_label_code: Dict[str, int]) -> Tuple[List[DataFrame], List[int]]:
        data_windows: List[DataFrame] = []
        labels_of_data_windows: List[int] = []
        for sample in samples:
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
                labels_of_data_windows.append(label_to_label_code[sample.label])

        # TODO label 0 for the windows not in timeframes
        return (data_windows, labels_of_data_windows)

    def __get_extracted_features(self, sliding_window: SlidingWindow) -> DataFrame:
        extracted_features_dict = sliding_window.extracted_features
        cached_extracted_features: Dict[Feature, DataFrame] = {}
        not_cached_features: List[Feature] = []
        # Sorted so that we always get the same column order
        for feature in self.features:
            if feature in extracted_features_dict:
                extracted_feature_doc = self.db.extracted_features.find_one(
                    {"_id": extracted_features_dict[feature]})
                cached_extracted_features[feature] = pickle.loads(extracted_feature_doc["data"])  # Binary to bytes??
            else:
                not_cached_features.append(feature)

        newly_extracted_features: Dict[Feature, DataFrame]

        if not_cached_features:
            data_windows = [pickle.loads(data_window) for data_window in sliding_window.data_windows]
            newly_extracted_features = self.__extractFeatures(data_windows, not_cached_features)
            for feature in newly_extracted_features:
                extracted_feature = ExtractedFeature(data=pickle.dumps(newly_extracted_features[feature]))
                result = self.db.extracted_features.insert_one(extracted_feature.dict(exclude={"id"}))
                self.db.sliding_windows.update_one({"_id": sliding_window.id},
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

    def __impute(self, train_data: DataFrame, test_data: DataFrame) -> Tuple[DataFrame, DataFrame, IImputer]:
        imputer_object: IImputer = get_imputer(self.imputation)
        imputer_object.fit(train_data)
        imputed_train_data = DataFrame(imputer_object.transform(train_data), columns=train_data.columns)
        imputed_test_data = DataFrame(imputer_object.transform(test_data), columns=test_data.columns)
        return (imputed_train_data, imputed_test_data, imputer_object)

    def __normalize(self, train_data: DataFrame, test_data: DataFrame) -> Tuple[DataFrame, DataFrame, INormalizer]:
        normalizer_object: INormalizer = get_normalizer(self.normalizer)
        normalizer_object.fit(train_data)
        normalized_train_data = DataFrame(normalizer_object.transform(train_data), columns=train_data.columns)
        normalized_test_data = DataFrame(normalizer_object.transform(test_data), columns=test_data.columns)
        return (normalized_train_data, normalized_test_data, normalizer_object)

    def __train(self, data: DataFrame, labels_of_data_windows: List[int]) -> IClassifier:
        classifier_object = get_classifier(self.classifier, self.hyperparameters)
        classifier_object.fit(data, labels_of_data_windows)
        return classifier_object

    def __get_performance_metrics(self, classifier_object: IClassifier, test_data: DataFrame, test_labels: List[int], label_code_to_label: Dict[int, str]) -> PerformanceMetricsPerLabel:
        prediction = classifier_object.predict(test_data)
        result = {}
        print(prediction)
        for label_code, performance_metric in classification_report(test_labels, prediction, output_dict=True).items():
            if label_code in [str(label_code) for label_code in label_code_to_label]:
                result[label_code_to_label[int(label_code)]] = PerformanceMetrics(metrics=performance_metric)
        return PerformanceMetricsPerLabel(metrics_of_labels=result)
