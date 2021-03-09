import json
import tsfresh
import pickle
import requests
from gridfs import GridFS
from typing import Any, Dict, List, Tuple
from pymongo.database import Database
from pandas import DataFrame
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from pymongo import MongoClient
from multiprocessing import set_start_method
from fastapi.exceptions import HTTPException
from starlette import status

from app.config import get_settings
from app.models.ml_model import MlModel, PerformanceMetrics, PerformanceMetricsPerLabel
from app.models.cached_data import ExtractedFeature, SlidingWindow
from app.models.mongo_model import OID
from app.models.workspace import Sample, SampleInJson, Workspace, WorkspaceData
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
        self.workspace: Workspace
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

    def __update_workspace_samples(self):
        last_modified: int = requests.get(url="/api/workspaces/" + str(self.workspace_id) +
                                          "/samples", params={"onlyDate": True})
        if last_modified == self.workspace.workspaceData.last_modified:
            return

        labels: List[str] = [label["name"]
                             for label in json.loads(requests.get(url="").json())]  # TODO complete api endpoint
        label_to_label_code: Dict[str, str] = {labels[i]: str(i+1) for i in range(len(labels))}
        label_code_to_label: Dict[str, str] = {str(i+1): labels[i] for i in range(len(labels))}

        # TODO complete api endpoint
        samples = [SampleInJson(sample) for sample in json.loads(requests.get(url="").json())]
        # Validate the received samples
        # TODO instead of raising add a field (enum?) to the workspace that represents the training state
        for sample in samples:
            for sensor in self.workspace.sensors:
                if sensor.name not in sample.sensorDataPoints:
                    # TODO set training state of workspace: "Sensors of the sample " + str(sample.id) + " does not match the workspace sensors"
                    exit(-1)
                if len(sample.sensorDataPoints[sensor.name]) != sample.dataPointCount:
                    # TODO set training state of workspace: "Number of data points of sensor " + sensor.name + " from sample " + str(sample.id) + " does not match the claimed data point count"
                    exit(-1)
                for data_point in sample.sensorDataPoints:
                    if len(data_point) != len(sensor.dataFormat):
                        # TODO set training state of workspace: "At least one data point of " + sensor.name + " from sample " + str(sample.id) + " is malformed"
                        exit(-1)
        # Validated so save the samples to the database
        new_samples: List[Sample] = []
        for sample in samples:
            result = self.fs.put(pickle.dumps(sample.sensorDataPoints))
            sample_to_insert = Sample(label=sample.label, timeframes=sample.timeframes,
                                      dataPointCount=sample.dataPointCount, sensorDataPoints=result).dict()
            new_samples.append(sample_to_insert)
        new_data = WorkspaceData(samples=new_samples, labelToLabelCode=label_to_label_code,
                                 labelCodeToLabel=label_code_to_label)
        self.db.workspaces.update_one({"_id": self.workspace_id}, {"$set": {"workspaceData": new_data.dict()}})
        self.workspace.workspaceData = new_data

    def train(self):
        # The child process can fork safely, even though it must be spawned by the parent
        set_start_method("fork", force=True)

        print("--start--")
        self.client = MongoClient(self.settings.client_uri, self.settings.client_port)
        self.db = self.client[self.settings.db_name]
        self.fs = GridFS(self.db)
        self.workspace = Workspace(**self.db.workspaces.find_one({"_id": self.workspace_id}))
        # TODO: put back in self.__update_workspace_samples()

        print("start split to windows")

        # data split to windows
        pipeline_data = self.__get_data_split_to_windows()
        labels_of_data_windows = pipeline_data.labelsOfDataWindows

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

        print(train_data)
        # normalize
        (train_data, test_data, normalizer_object) = self.__normalize(train_data, test_data)
        print("start train")

        # train
        classifier_object = self.__train(train_data, train_labels)

        print("start performance metrics")

        # performance metrics
        performance_metrics = self.__get_performance_metrics(classifier_object, test_data, test_labels)

        print("start saving ml model")

        # save ml_model
        imputer_object_db_id = self.fs.put(pickle.dumps(imputer_object))
        normalizer_object_db_id = self.fs.put(pickle.dumps(normalizer_object))
        classifier_object_db_id = self.fs.put(pickle.dumps(classifier_object))

        ml_model = MlModel(name=self.model_name, workspaceId=self.workspace_id, windowSize=self.window_size, slidingStep=self.sliding_step,
                           sortedFeatures=self.sorted_features, imputation=self.imputation, imputerObject=imputer_object_db_id,
                           normalization=self.normalizer, normalizerObject=normalizer_object_db_id, classifier=self.classifier,
                           classifierObject=classifier_object_db_id, hyperparameters=self.hyperparameters, labelPerformanceMetrics=performance_metrics,
                           columnOrder=train_data.columns.tolist(), sensors=self.workspace.sensors, labelCodeToLabel=self.workspace.workspaceData.labelCodeToLabel)

        result = self.db.ml_models.insert_one(ml_model.dict(exclude_none=True))
        self.db.workspaces.update_one({"_id": self.workspace_id}, {
                                      "$push": {"mlModels": result.inserted_id}, "$set": {"progress": -1}})

        print("--end--")
        exit(0) # Clean up automatically

    def __get_data_split_to_windows(self) -> SlidingWindow:
        # If already cached, retrieve from the database
        if str(self.window_size) + "_" + str(self.sliding_step) in self.workspace.workspaceData.slidingWindows:
            data_windows_id = self.workspace.workspaceData.slidingWindows[str(
                self.window_size) + "_" + str(self.sliding_step)]
            return SlidingWindow(**self.db.sliding_windows.find_one({"_id": data_windows_id}))

        # Otherwise split yourself and cache
        (data_windows, labels_of_data_windows) = self.__split_to_windows()

        inserted_id = self.fs.put(pickle.dumps(data_windows))
        sliding_window = SlidingWindow(dataWindows=inserted_id, labelsOfDataWindows=labels_of_data_windows)

        result = self.db.sliding_windows.insert_one(sliding_window.dict(exclude_none=True))
        sliding_window.id = result.inserted_id
        self.db.workspaces.update_one({"_id": self.workspace_id}, {
            "$set": {"workspaceData.slidingWindows." + str(self.window_size) + "_" + str(self.sliding_step): result.inserted_id}})
        return sliding_window

    def __split_to_windows(self) -> Tuple[List[List[Dict]], List[str]]:
        # TODO CONFIRM WE USE INDICES INSTEAD OF TIMESTAMPS FOR TIMEFRAMES
        data_windows: List[List[Dict]] = []
        labels_of_data_windows: List[str] = []
        for sample in self.workspace.workspaceData.samples:
            timeframe_iterator = iter(sample.timeframes)
            current_timeframe = next(timeframe_iterator)

            # TODO FOR ÖMER: [1,2],[3,4] is not legal timeframes, use [1,4] instead
            sensor_data_points: Dict[str, List[List[float]]] = pickle.loads(self.fs.get(sample.sensorDataPoints).read())
            # For each window in this sample, incremented by the sliding step value starting from 0
            for window_offset in range(0, sample.dataPointCount - self.window_size + 1, self.sliding_step):
                while window_offset > current_timeframe.end:
                    try:
                        current_timeframe = next(timeframe_iterator)
                    except:
                        break
                data_window_in_timeframe = window_offset >= current_timeframe.start and window_offset + \
                    self.window_size - 1 <= current_timeframe.end

                # [data_point_index][sensor_component: value]
                data_window: List[Dict[str, float]] = [{} for _range_ in range(self.window_size)]

                for sensor in self.workspace.sensors:
                    # Data points of this sensor (time as index)
                    data_points = sensor_data_points[sensor.name]
                    # For each data point which is in this data window
                    for datapoint_offset in range(0, self.window_size):
                        # Append the values to this data window (index of values is e.g x in acc_x)
                        data_point = data_points[window_offset+datapoint_offset]
                        data_window[datapoint_offset].update({sensor.name + "_" + sensor.dataFormat[v]: data_point[v]
                                                              for v in range(len(sensor.dataFormat))})
                # Append this data window as a DataFrame to the list of all data windows
                data_windows.append(data_window)
                labels_of_data_windows.append(
                    self.workspace.workspaceData.labelToLabelCode[sample.label] if data_window_in_timeframe else "0")

        return (data_windows, labels_of_data_windows)

    def __get_extracted_features(self, sliding_window: SlidingWindow) -> DataFrame:
        extracted_features_dict = sliding_window.extractedFeatures
        cached_extracted_features: Dict[Feature, List[Dict[str, float]]] = {}
        not_cached_features: List[Feature] = []
        # Sorted so that we always get the same column order
        for feature in self.sorted_features:
            if feature in extracted_features_dict:
                extracted_feature = ExtractedFeature(**self.db.extracted_features.find_one(
                    {"_id": extracted_features_dict[feature]}))
                cached_extracted_features[feature] = pickle.loads(self.fs.get(extracted_feature.data).read())
            else:
                not_cached_features.append(feature)

        newly_extracted_features: Dict[Feature, List[Dict[str, float]]]

        if not_cached_features:
            # We have to load from the database for the case where sliding windows are cached
            data_windows: List[List[Dict]] = pickle.loads(self.fs.get(sliding_window.dataWindows).read())
            newly_extracted_features = self.__extractFeatures(data_windows, not_cached_features)
            for feature in newly_extracted_features:
                inserted_id = self.fs.put(pickle.dumps(newly_extracted_features[feature]))
                extracted_feature = ExtractedFeature(data=inserted_id)
                result = self.db.extracted_features.insert_one(extracted_feature.dict(exclude_none=True))
                self.db.sliding_windows.update_one({"_id": sliding_window.id},
                                                   {"$set": {"extractedFeatures." + feature.value: result.inserted_id}})

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
        settings = {key: ComprehensiveFCParameters()[key] for key in features}
        newly_extracted_features: Dict[Feature, List[Dict[str, float]]] = {
            f: [{} for _range_ in range(len(data_windows))] for f in features}

        concatenated_data_windows: List[Dict] = []

        for data_window_index in range(len(data_windows)):
            for i in range(len(data_windows[data_window_index])):
                data_windows[data_window_index][i]["id"] = data_window_index
                concatenated_data_windows.append(data_windows[data_window_index][i])

        extracted = tsfresh.extract_features(
            DataFrame(concatenated_data_windows), column_id="id", default_fc_parameters=settings, pivot=False, disable_progressbar=False)

        for i in range(len(extracted)):
            feature: Feature = features[i % len(features)]
            data_window_index = extracted[i][0]
            label = extracted[i][1]
            value = extracted[i][2]
            newly_extracted_features[feature][data_window_index][label] = value

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

    def __get_performance_metrics(self, classifier_object: IClassifier, test_data: DataFrame, test_labels: List[int]) -> PerformanceMetricsPerLabel:
        prediction = classifier_object.predict(test_data)
        result = {}
        print(classification_report(test_labels, prediction, output_dict=True))
        for label_code, performance_metric in classification_report(test_labels, prediction, output_dict=True).items():
            if label_code in self.workspace.workspaceData.labelCodeToLabel:
                result[self.workspace.workspaceData.labelCodeToLabel[label_code]
                       ] = PerformanceMetrics(metrics=performance_metric)
            elif label_code == "0":
                result["Other"] = PerformanceMetrics(metrics=performance_metric)

        return PerformanceMetricsPerLabel(metrics_of_labels=result)
