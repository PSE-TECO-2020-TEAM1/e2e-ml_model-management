from app.ml.objects.feature.enum import Feature
from typing import List
from bson.objectid import ObjectId
from pandas.core.frame import DataFrame
from app.models.domain.feature_extraction_data import FeatureExtractionData


def get_feature_extraction_data_stub_5_1():
    return FeatureExtractionData(
        data_windows_df_file_ID=ObjectId("60707242b377f2b04ebf6737"),
        labels_of_data_windows_file_ID=ObjectId("6070724cdd135f6e692c0959"),
        sensor_component_feature_df_file_IDs={
            "x_Accelerometer": {Feature.MINIMUM: ObjectId("6070727ed1a8cbba14ea120f"), Feature.MAXIMUM: ObjectId("6070728ce5eed4a716c39858")},
            "y_Accelerometer": {Feature.MINIMUM: ObjectId("607072b8c5d9f62dc8af39c5"), Feature.MAXIMUM: ObjectId("607072bd0cfbe6257fd2ccaa")},
            "z_Accelerometer": {Feature.MINIMUM: ObjectId("607072c3c456e5966ed54877"), Feature.MAXIMUM: ObjectId("607072c761d523b4852bc4fb")},
            "x_Gyroscope": {Feature.MINIMUM: ObjectId("607072ccc9439e3d45c48a31"), Feature.MAXIMUM: ObjectId("607072d1d7184b370db7a7f0")},
            "y_Gyroscope": {Feature.MINIMUM: ObjectId("607072d5638d1b75b59e8990"), Feature.MAXIMUM: ObjectId("607072dae852a74abbef32ca")},
            "z_Gyroscope": {Feature.MINIMUM: ObjectId("607072dfcbc9bd19451f0be2"), Feature.MAXIMUM: ObjectId("607072e37e4784db534231bf")},
        }
    )


def get_data_windows_df_5_1():
    return DataFrame([
        # sample_stub_1
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        # sample_stub_2
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4}
    ])


def get_labels_of_data_windows_5_1():
    return ["Shake", "Shake", "Rotate", "Rotate"]


def get_sensor_component_feature_dfs_5_1():
    return {
        "x_Accelerometer": {
            Feature.MINIMUM: DataFrame([{"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}]),
            Feature.MAXIMUM: DataFrame([{"x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}, {
                "x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}])
        },
        "y_Accelerometer": {
            Feature.MINIMUM: DataFrame([{"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}]),
            Feature.MAXIMUM: DataFrame([{"x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}, {
                "x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}])
        },
        "z_Accelerometer": {
            Feature.MINIMUM: DataFrame([{"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}]),
            Feature.MAXIMUM: DataFrame([{"x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}, {
                "x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}])
        },
        "x_Gyroscope": {
            Feature.MINIMUM: DataFrame([{"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}]),
            Feature.MAXIMUM: DataFrame([{"x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}, {
                "x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}])
        },
        "y_Gyroscope": {
            Feature.MINIMUM: DataFrame([{"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}]),
            Feature.MAXIMUM: DataFrame([{"x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}, {
                "x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}])
        },
        "z_Gyroscope": {
            Feature.MINIMUM: DataFrame([{"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}, {"x_Accelerometer__minimum": 1.0}]),
            Feature.MAXIMUM: DataFrame([{"x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}, {
                "x_Accelerometer__maximum": 1.0}, {"x_Accelerometer__maximum": 1.0}])
        }
    }


def get_feature_extraction_data_stub_4_2():
    return FeatureExtractionData(
        data_windows_df_file_ID=ObjectId("6070730112e067c7e6bf65df"),
        labels_of_data_windows_file_ID=ObjectId("6070730644135e99d11e07e3"),
        sensor_component_feature_df_file_IDs={
            "x_Accelerometer": {Feature.MEAN: ObjectId("6070730a3cb3407c3cee5088"), Feature.MEDIAN: ObjectId("6070730f960d158336e60381")},
            "y_Accelerometer": {Feature.MEAN: ObjectId("60707313d126b4fc5b5aad21"), Feature.MEDIAN: ObjectId("6070731781da40c314d7fa59")},
            "z_Accelerometer": {Feature.MEAN: ObjectId("6070731c86cd07184cb380ec"), Feature.MEDIAN: ObjectId("607073213b96af39b7103af7")},
            "x_Gyroscope": {Feature.MEAN: ObjectId("60707326750a110c0518ff57"), Feature.MEDIAN: ObjectId("6070732a0464a34f7737b1c4")},
            "y_Gyroscope": {Feature.MEAN: ObjectId("6070732e6c7d7545d2c3a6a5"), Feature.MEDIAN: ObjectId("6070733273f8f9217b07af9d")},
            "z_Gyroscope": {Feature.MEAN: ObjectId("6070733ab4862c8a4ea5591f"), Feature.MEDIAN: ObjectId("6070733e37039f4f3f0a2bf1")},
        }
    )


def get_data_windows_df_4_2():
    return DataFrame([
        # sample_stub_1
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 1},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 2},
        # sample_stub_2
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 3},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
        {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
         "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0, "id": 4},
    ])


def get_labels_of_data_windows_4_2():
    return ["Shake", "Shake", "Rotate", "Rotate"]


def get_sensor_component_feature_dfs_4_2():
    return {
        "x_Accelerometer": {
            Feature.MEAN: DataFrame([{"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}]),
            Feature.MEDIAN: DataFrame([{"x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}, {
                "x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}])
        },
        "y_Accelerometer": {
            Feature.MEAN: DataFrame([{"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}]),
            Feature.MEDIAN: DataFrame([{"x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}, {
                "x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}])
        },
        "z_Accelerometer": {
            Feature.MEAN: DataFrame([{"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}]),
            Feature.MEDIAN: DataFrame([{"x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}, {
                "x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}])
        },
        "x_Gyroscope": {
            Feature.MEAN: DataFrame([{"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}]),
            Feature.MEDIAN: DataFrame([{"x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}, {
                "x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}])
        },
        "y_Gyroscope": {
            Feature.MEAN: DataFrame([{"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}]),
            Feature.MEDIAN: DataFrame([{"x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}, {
                "x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}])
        },
        "z_Gyroscope": {
            Feature.MEAN: DataFrame([{"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}]),
            Feature.MEDIAN: DataFrame([{"x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}, {
                "x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}])
        }
    }
