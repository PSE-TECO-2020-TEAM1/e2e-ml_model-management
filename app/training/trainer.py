from os import pipe
import tsfresh
import pickle
from gridfs import GridFS
from typing import Any, Dict, List, Tuple
from pymongo.database import Database
from pandas import DataFrame
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from pymongo import MongoClient

from app.config import get_settings
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
        self.client: MongoClient
        self.db: Database
        self.fs: GridFS
        self.settings = get_settings()

        self.progress: int = 0  # percentage
        self.workspace_id = workspace_id
        self.model_name = model_name
        self.imputation = imputation
        self.sorted_features = sorted(features)
        self.normalizer = normalizer
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.window_size = window_size
        self.sliding_step = sliding_step

    def train(self):
        print("--start--")
        self.client = MongoClient(self.settings.client_uri, self.settings.client_port)
        self.db = self.client[self.settings.db_name]
        self.fs = GridFS(self.db)

        workspace_document = self.db.workspaces.find_one({"_id": self.workspace_id})
        workspace = Workspace(**workspace_document, id=workspace_document["_id"])

        print("start split to windows")

        # data split to windows
        pipeline_data = self.__get_data_split_to_windows(workspace)
        labels_of_data_windows = pipeline_data.labels_of_data_windows

        print("start feature extraction")

        # extract features
        pipeline_data = self.__get_extracted_features(pipeline_data)

        # train-test split
        pipeline_data, labels_of_data_windows = shuffle(pipeline_data, labels_of_data_windows)
        pipeline_data.reset_index(inplace=True, drop=True)

        train_data = pipeline_data.iloc[:int(pipeline_data.shape[0]*0.8)]
        train_labels = labels_of_data_windows[:int(len(labels_of_data_windows)*0.8)]
        test_data = pipeline_data.iloc[int(pipeline_data.shape[0]*0.8):]
        test_labels = labels_of_data_windows[int(len(labels_of_data_windows)*0.8):]

        del pipeline_data
        del labels_of_data_windows

        print("start imputation")

        # impute
        (train_data, test_data, imputer_object) = self.__impute(train_data, test_data)
        
        print("start normalize")

        # normalize
        (train_data, test_data, normalizer_object) = self.__normalize(train_data, test_data)
        print("start train")

        # train
        classifier_object = self.__train(train_data, train_labels)

        print("start performance metrics")

        # performance metrics
        performance_metrics = self.__get_performance_metrics(
            classifier_object, test_data, test_labels, workspace.workspace_data.label_code_to_label)

        print("Performance metrics:\n")
        print(performance_metrics)
        print("start saving ml model")

        # save ml_model
        imputer_object_db_id = self.fs.put(pickle.dumps(imputer_object))
        normalizer_object_db_id = self.fs.put(pickle.dumps(normalizer_object))
        classifier_object_db_id = self.fs.put(pickle.dumps(classifier_object))
        ml_model = ML_Model(name=self.model_name, workspace_id=self.workspace_id, windowSize=self.window_size, slidingStep=self.sliding_step,
                            features=self.sorted_features, imputation=self.imputation, imputer_object=imputer_object_db_id, normalization=self.normalizer, normalizer_object=normalizer_object_db_id, classifier=self.classifier, classifier_object=classifier_object_db_id, hyperparameters=self.hyperparameters, labelPerformanceMetrics=performance_metrics)

        result = self.db.ml_models.insert_one(ml_model.dict(exclude={"id"}))
        self.db.workspaces.update_one({"_id": workspace.id}, {
                                      "$push": {"ml_models": result.inserted_id}, "$set": {"progress": -1}})

        # close db connection
        self.client.close()

        print("--end--")

    def __get_data_split_to_windows(self, workspace: Workspace) -> SlidingWindow:
        workspace_data = workspace.workspace_data
        # If already cached, retrieve from the database
        if str(self.window_size) + "_" + str(self.sliding_step) in workspace_data.sliding_windows:
            data_windows_id = workspace_data.sliding_windows[str(self.window_size) + "_" + str(self.sliding_step)]
            sliding_window_document = self.db.sliding_windows.find_one({"_id": data_windows_id})
            return SlidingWindow(**sliding_window_document, id=sliding_window_document["_id"])
        # Split yourself and cache otherwise

        sample_docs = list(self.db.samples.find({"_id": {"$in": workspace_data.samples}}))

        samples = [Sample(**sample_doc, id=sample_doc["_id"]) for sample_doc in sample_docs]

        (data_windows, labels_of_data_windows) = self.__split_to_windows(
            samples, workspace_data.label_to_label_code)

        inserted_id = self.fs.put(pickle.dumps(data_windows))
        sliding_window = SlidingWindow(data_windows=inserted_id, labels_of_data_windows=labels_of_data_windows)

        result = self.db.sliding_windows.insert_one(sliding_window.dict(exclude={"id"}))
        sliding_window.id = result.inserted_id
        self.db.workspaces.update_one({"_id": workspace.id}, {
            "$set": {"workspace_data.sliding_windows." + str(self.window_size) + "_" + str(self.sliding_step): result.inserted_id}})
        return sliding_window

    def __split_to_windows(self, samples: List[Sample], label_to_label_code: Dict[str, int]) -> Tuple[List[List[Dict]], List[int]]:
        # TODO CONFIRM WE USE INDICES INSTEAD OF TIMESTAMPS FOR TIMEFRAMES
        data_windows: List[List[Dict]] = []
        labels_of_data_windows: List[int] = []
        for sample in samples:
            timeframe_iterator = iter(sample.timeframes)
            current_timeframe = next(timeframe_iterator)

            # TODO FOR ÖMER: [1,2],[3,4] is not legal timeframes, use [1,4] instead

            # Length of data points for each sensor is same, so we take the first one
            length_of_data_points = len(list(sample.sensor_data_points.values())[0])
            # For each window in this sample, incremented by the sliding step value starting from 0
            for window_offset in range(0, length_of_data_points - self.window_size + 1, self.sliding_step):
                while window_offset > current_timeframe.end:
                    try:
                        current_timeframe = next(timeframe_iterator)
                    except:
                        break
                data_window_in_timeframe = window_offset >= current_timeframe.start and window_offset + \
                    self.window_size - 1 <= current_timeframe.end

                #  [time][sensor e.g acc_x]
                data_window: List[Dict[str, float]] = [{} for _range_ in range(self.window_size)]
                # For each sensor (sorted so that we always get the same column order)
                for sensor in sorted(sample.sensor_data_points):
                    # Data points of this sensor (time as index)
                    data_points: List[DataPoint] = sample.sensor_data_points[sensor]
                    # For each data point which is in this data window
                    for datapoint_offset in range(0, self.window_size):
                        # Append the values to this data window (index of values is e.g x in acc_x)
                        data_point_values = data_points[window_offset+datapoint_offset].values
                        data_window[datapoint_offset].update({sensor+str(v): data_point_values[v]
                                                              for v in range(len(data_point_values))})
                # Append this data window as a DataFrame to the list of all data windows
                data_windows.append(data_window)
                labels_of_data_windows.append(label_to_label_code[sample.label] if data_window_in_timeframe else 0)

        return (data_windows, labels_of_data_windows)

    def __get_extracted_features(self, sliding_window: SlidingWindow) -> DataFrame:
        extracted_features_dict = sliding_window.extracted_features
        cached_extracted_features: Dict[Feature, List[Dict[str, float]]] = {}
        not_cached_features: List[Feature] = []
        # Sorted so that we always get the same column order
        for feature in self.sorted_features:
            if feature in extracted_features_dict:
                extracted_feature_doc = self.db.extracted_features.find_one(
                    {"_id": extracted_features_dict[feature]})
                cached_extracted_features[feature] = pickle.loads(self.fs.get(extracted_feature_doc["data"]).read())
            else:
                not_cached_features.append(feature)

        newly_extracted_features: Dict[Feature, List[Dict[str, float]]]

        if not_cached_features:
            # We have to load from the database for the case where sliding windows are cached
            data_windows = pickle.loads(self.fs.get(sliding_window.data_windows).read())
            newly_extracted_features = self.__extractFeatures(data_windows, not_cached_features)
            for feature in newly_extracted_features:
                inserted_id = self.fs.put(pickle.dumps(newly_extracted_features[feature]))
                extracted_feature = ExtractedFeature(data=inserted_id)
                result = self.db.extracted_features.insert_one(extracted_feature.dict(exclude={"id"}))
                self.db.sliding_windows.update_one({"_id": sliding_window.id},
                                                   {"$set": {"extracted_features." + feature.value: result.inserted_id}})

        # Join them together
        number_of_data_windows: int = None
        dataframes: List[Dict] = None
        for feature in self.sorted_features:
            dict_to_use = cached_extracted_features if feature in cached_extracted_features else newly_extracted_features
            if number_of_data_windows is None:
                number_of_data_windows = len(dict_to_use[feature])
                dataframes = [{} for __range__ in range(number_of_data_windows)]
            for i in range(number_of_data_windows):
                dataframes[i].update(dict_to_use[feature][i])

        return DataFrame(dataframes)

    def __extractFeatures(self, data_windows: List[List[Dict]], features: List[Feature]) -> Dict[Feature, List[Dict[str, float]]]:
        # TODO parallelize maybe xd
        settings = {key.value: ComprehensiveFCParameters()[key.value] for key in features}
        newly_extracted_features: Dict[Feature, List[Dict[str, float]]] = {
            f: [{} for _range_ in range(len(data_windows))] for f in features}

        concatenated_data_windows: List[Dict] = []

        for data_window_index in range(len(data_windows)):
            for i in range(len(data_windows[data_window_index])):
                data_windows[data_window_index][i]["id"] = data_window_index
                concatenated_data_windows.append(data_windows[data_window_index][i])

        extracted = tsfresh.extract_features(
            DataFrame(concatenated_data_windows), column_id="id", default_fc_parameters=settings, disable_progressbar=False).to_dict(orient="records")

        for data_window_index in range(len(data_windows)):
            extracted_from_curr_window = extracted[data_window_index]
            curr_extracted_keys = list(extracted_from_curr_window.keys())
            for key_index in range(len(curr_extracted_keys)):
                curr_key = curr_extracted_keys[key_index]
                feature = features[key_index % len(features)]
                newly_extracted_features[feature][data_window_index][curr_key] = extracted_from_curr_window[curr_key]

        for feature, data in newly_extracted_features.items():
            newly_extracted_features[feature] = data
        return newly_extracted_features

    # TODO better way to perform pipeline steps from sklearn library? (e.g. impute, normalize, ...)

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
        for label_code, performance_metric in classification_report(test_labels, prediction, output_dict=True).items():
            if label_code in [str(label_code) for label_code in label_code_to_label]:
                result[label_code_to_label[int(label_code)]] = PerformanceMetrics(metrics=performance_metric)
        return PerformanceMetricsPerLabel(metrics_of_labels=result)
