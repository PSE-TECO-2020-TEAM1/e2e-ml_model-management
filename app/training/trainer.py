import jwt
import pandas
import tsfresh
import pickle
import requests
from datetime import datetime, timedelta
from gridfs import GridFS
from typing import Any, Dict, List, Tuple
from pymongo.database import Database
from pandas import DataFrame
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from pymongo import MongoClient
from multiprocessing import set_start_method

from app.config import get_settings
from app.models.ml_model import Hyperparameter, MlModel, PerformanceMetrics, SingleMetric
from app.models.cached_data import ExtractedFeature, SlidingWindow
from app.models.mongo_model import OID
from app.models.workspace import Sample, SampleInJson, Workspace, WorkspaceData
from app.training.factory import get_classifier, get_imputer, get_normalizer
from app.util.sample_parser import SampleParser
from app.util.ml_objects import IClassifier, IImputer, INormalizer
from app.util.training_parameters import Classifier, Feature, Imputation, Normalization


class Trainer():

    def __init__(self, workspace_id: OID, model_name: str, window_size: int, sliding_step: int, features: List[Feature], imputation: Imputation,
                 normalizer: Normalization, classifier: Classifier, hyperparameters: Dict[str, Any]):
        # TODO wrapper for config?
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
        token = jwt.encode({"userId": str(self.workspace.userId), "exp": datetime.utcnow() +
                            timedelta(minutes=10)}, key=self.settings.AUTH_SECRET, algorithm="HS256")
        auth_header = {"Authorization": "Bearer " + token}
        url = self.settings.WORKSPACE_MANAGEMENT_IP_PORT + \
            "/api/workspaces/"+str(self.workspace_id)+"/samples?onlyDate=true"
        last_modified: int = requests.get(url=url, headers=auth_header).json()
        if (self.workspace.workspaceData is not None) and (last_modified == self.workspace.workspaceData.lastModified):
            return
        url = self.settings.WORKSPACE_MANAGEMENT_IP_PORT+"/api/workspaces/"+str(self.workspace_id)+"/labels"
        label_res = requests.get(url=url, headers=auth_header).json()

        labels: List[str] = [label["name"] for label in label_res]
        label_to_label_code: Dict[str, str] = {labels[i]: str(i+1) for i in range(len(labels))}
        label_to_label_code["Other"] = "0"
        label_code_to_label: Dict[str, str] = {str(i+1): labels[i] for i in range(len(labels))}
        label_code_to_label["0"] = "Other"

        url = self.settings.WORKSPACE_MANAGEMENT_IP_PORT+ "/api/workspaces/" + str(self.workspace_id) + "/samples?showDataPoints=true"
        received_samples = requests.get(url=url, headers=auth_header).json()
        for sample in received_samples:
            for sensor_type in sample["sensorDataPoints"]:
                sensor_type["sensor"] = sensor_type.pop("sensorName")

        samples = [SampleInJson(**sample) for sample in received_samples]
        parser = SampleParser(sensors=self.workspace.sensors)
        new_samples: List[Sample] = []
        for sample in samples:
            parsed = parser.parse_sample(sample)
            for dataframe in parsed.positive:
                result = self.fs.put(pickle.dumps(dataframe))
                sample_to_insert = Sample(label=sample.label, sensorDataPoints=result).dict()
                new_samples.append(sample_to_insert)
            for dataframe in parsed.negative:
                result = self.fs.put(pickle.dumps(dataframe))
                sample_to_insert = Sample(label="Other", sensorDataPoints=result).dict()
                new_samples.append(sample_to_insert)

        new_data = WorkspaceData(samples=new_samples, lastModified=last_modified, labelToLabelCode=label_to_label_code,
                                 labelCodeToLabel=label_code_to_label)
        self.db.workspaces.update_one({"_id": self.workspace_id}, {"$set": {"workspaceData": new_data.dict()}})
        self.workspace.workspaceData = new_data

    def train(self):
        # The child process can fork safely, even though it must be spawned by the parent
        set_start_method("fork", force=True)

        print("--start--")
        self.client = MongoClient(self.settings.DATABASE_URI, self.settings.DATABASE_PORT)
        self.db = self.client[self.settings.DATABASE_NAME]
        self.fs = GridFS(self.db)
        self.workspace = Workspace(**self.db.workspaces.find_one({"_id": self.workspace_id}))
        self.__update_workspace_samples()

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

        list_of_parameters = []
        for name, value in self.hyperparameters.items():
            list_of_parameters.append(Hyperparameter(name=name, value=str(value)))

        ml_model = MlModel(name=self.model_name, workspaceId=self.workspace_id, windowSize=self.window_size, slidingStep=self.sliding_step,
                           sortedFeatures=self.sorted_features, imputation=self.imputation, imputerObject=imputer_object_db_id,
                           normalization=self.normalizer, normalizerObject=normalizer_object_db_id, classifier=self.classifier,
                           classifierObject=classifier_object_db_id, hyperparameters=list_of_parameters, labelPerformanceMetrics=performance_metrics,
                           columnOrder=train_data.columns.tolist(), sensors=self.workspace.sensors, labelCodeToLabel=self.workspace.workspaceData.labelCodeToLabel)

        result = self.db.ml_models.insert_one(ml_model.dict(exclude_none=True))
        self.db.workspaces.update_one({"_id": self.workspace_id}, {
                                      "$push": {"mlModels": result.inserted_id}, "$set": {"progress": -1}})

        print("--end--")
        exit(0)  # Clean up automatically

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

    def __split_to_windows(self) -> Tuple[List[DataFrame], List[str]]:
        data_windows: List[DataFrame] = []
        labels_of_data_windows: List[str] = []
        for sample in self.workspace.workspaceData.samples:
            sensor_data_points: DataFrame = pickle.loads(self.fs.get(sample.sensorDataPoints).read())
            # For each window in this sample, incremented by the sliding step value starting from 0
            for window_offset in range(0, len(sensor_data_points.index) - self.window_size + 1, self.sliding_step):
                data_windows.append(sensor_data_points.iloc[window_offset:window_offset + self.window_size])
                label = self.workspace.workspaceData.labelToLabelCode[sample.label]
                labels_of_data_windows.append(label)
        return (data_windows, labels_of_data_windows)

    def __get_extracted_features(self, sliding_window: SlidingWindow) -> DataFrame:
        extracted_features_dict = sliding_window.extractedFeatures
        cached_extracted_features: Dict[Feature, DataFrame] = {}
        not_cached_features: List[Feature] = []
        # Sorted so that we always get the same column order
        for feature in self.sorted_features:
            if feature in extracted_features_dict:
                extracted_feature = ExtractedFeature(**self.db.extracted_features.find_one(
                    {"_id": extracted_features_dict[feature]}))
                cached_extracted_features[feature] = pickle.loads(self.fs.get(extracted_feature.data).read())
            else:
                not_cached_features.append(feature)

        newly_extracted_features: Dict[Feature, DataFrame]
        if not_cached_features:
            # We have to load from the database for the case where data windows are cached
            data_windows: List[DataFrame] = pickle.loads(self.fs.get(sliding_window.dataWindows).read())
            newly_extracted_features = self.__extractFeatures(data_windows, not_cached_features)
            for feature in newly_extracted_features:
                inserted_id = self.fs.put(pickle.dumps(newly_extracted_features[feature]))
                extracted_feature = ExtractedFeature(data=inserted_id)
                result = self.db.extracted_features.insert_one(extracted_feature.dict(exclude_none=True))
                self.db.sliding_windows.update_one({"_id": sliding_window.id},
                                                   {"$set": {"extractedFeatures." + feature.value: result.inserted_id}})

        # Join them together
        dataframes = []
        for feature in self.sorted_features:
            dict_to_use = cached_extracted_features if feature in cached_extracted_features else newly_extracted_features
            dataframes.append(dict_to_use[feature])
        return pandas.concat(dataframes, axis=1)

    def __extractFeatures(self, data_windows: List[DataFrame], features: List[Feature]) -> Dict[Feature, DataFrame]:
        newly_extracted_features: Dict[Feature, List[Dict[str, float]]] = {
            f: [{} for _range_ in range(len(data_windows))] for f in features}

        settings = {key: ComprehensiveFCParameters()[key] for key in features}
        for data_window_index in range(len(data_windows)):
            data_windows[data_window_index]["id"] = data_window_index
        data_windows = pandas.concat(data_windows)
        extracted = tsfresh.extract_features(data_windows, column_id="id",
                                             default_fc_parameters=settings, pivot=False, disable_progressbar=False)
        # Split by columns features
        for i in range(len(extracted)):
            feature: Feature = features[i % len(features)]
            data_window_index = extracted[i][0]
            label = extracted[i][1]
            value = extracted[i][2]
            newly_extracted_features[feature][data_window_index][label] = value
        # Convert to DataFrame
        for feature in newly_extracted_features:
            newly_extracted_features[feature] = DataFrame(newly_extracted_features[feature])
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

    def __get_performance_metrics(self, classifier_object: IClassifier, test_data: DataFrame, test_labels: List[int]) -> List[PerformanceMetrics]:
        prediction = classifier_object.predict(test_data)
        result = []
        for label_code, performance_metric in classification_report(test_labels, prediction, output_dict=True).items():
            # Last element is always a float that represents what???
            if type(performance_metric) is not dict:
                continue
            metrics = []
            for name, score in performance_metric.items():
                metrics.append(SingleMetric(name=name, score=score))
            if label_code in self.workspace.workspaceData.labelCodeToLabel:            
                label = self.workspace.workspaceData.labelCodeToLabel[label_code]
                result.append(PerformanceMetrics(label=label, metrics=metrics))

        return result
