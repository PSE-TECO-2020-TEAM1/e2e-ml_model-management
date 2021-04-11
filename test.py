from app.ml.objects.classification.enum import Classifier
from app.ml.objects.normalization.enum import Normalization
from app.ml.objects.imputation.enum import Imputation
from app.models.domain.sliding_window import SlidingWindow
from app.ml.util.data_processing import calculate_classification_report, extract_features, split_to_data_windows
from pandas.core.frame import DataFrame
from app.ml.objects.feature.enum import Feature
from multiprocessing import set_start_method
from app.ml.objects.pipeline_factory import make_pipeline_from_config
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
from app.models.domain.training_config import PerComponentConfig, PipelineConfig, TrainingConfig
from sklearn.model_selection import train_test_split
import pandas as pd

from app.ml.objects.classification.classifier_config_spaces.util import config_spaces


def train(training_config: TrainingConfig):
    set_start_method("fork", force=True)
    #------------------------------------FEATURE_EXTRACTION-----------------------------------#
    # Get the data frame ready for pipeline
    x, y = gather_features_and_labels()
    print(x)

    #--------------------------------------MODEL_TRAINING-------------------------------------#
    # We have to sort the columns correctly when we are predicting later so we save the order
    columns = x.columns.tolist()
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Train-test-split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    # Create the pipeline
    pipeline = make_pipeline_from_config(training_config.get_component_pipeline_configs(),
                                         training_config.classifier, training_config.hyperparameters)
    # Fit the model
    pipeline.fit(x_train, y_train)

    #------------------------------------CLASSIFICATION_REPORT--------------------------------#
    # Calculate performance metrics
    test_prediction_result = pipeline.predict(x_test)
    performance_metrics = calculate_classification_report(test_prediction_result, y_test, label_encoder)
    return (performance_metrics, columns, label_encoder, pipeline)


sensor_component_feature_dfs_4_2 = {
    "x_Accelerometer": {
        Feature.MEAN: DataFrame([{"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}, {"x_Accelerometer__mean": 1.0}]),
        Feature.MEDIAN: DataFrame([{"x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}, {
            "x_Accelerometer__median": 1.0}, {"x_Accelerometer__median": 1.0}])
    },
    "y_Accelerometer": {
        Feature.MEAN: DataFrame([{"y_Accelerometer__mean": 1.0}, {"y_Accelerometer__mean": 1.0}, {"y_Accelerometer__mean": 1.0}, {"y_Accelerometer__mean": 1.0}]),
        Feature.MEDIAN: DataFrame([{"y_Accelerometer__median": 1.0}, {"y_Accelerometer__median": 1.0}, {
            "y_Accelerometer__median": 1.0}, {"y_Accelerometer__median": 1.0}])
    },
    "z_Accelerometer": {
        Feature.MEAN: DataFrame([{"z_Accelerometer__mean": 1.0}, {"z_Accelerometer__mean": 1.0}, {"z_Accelerometer__mean": 1.0}, {"z_Accelerometer__mean": 1.0}]),
        Feature.MEDIAN: DataFrame([{"z_Accelerometer__median": 1.0}, {"z_Accelerometer__median": 1.0}, {
            "z_Accelerometer__median": 1.0}, {"z_Accelerometer__median": 1.0}])
    },
    "x_Gyroscope": {
        Feature.MEAN: DataFrame([{"x_Gyroscope__mean": 1.0}, {"x_Gyroscope__mean": 1.0}, {"x_Gyroscope__mean": 1.0}, {"x_Gyroscope__mean": 1.0}]),
        Feature.MEDIAN: DataFrame([{"x_Gyroscope__median": 1.0}, {"x_Gyroscope__median": 1.0}, {
            "x_Gyroscope__median": 1.0}, {"x_Gyroscope__median": 1.0}])
    },
    "y_Gyroscope": {
        Feature.MEAN: DataFrame([{"y_Gyroscope__mean": 1.0}, {"y_Gyroscope__mean": 1.0}, {"y_Gyroscope__mean": 1.0}, {"y_Gyroscope__mean": 1.0}]),
        Feature.MEDIAN: DataFrame([{"y_Gyroscope__median": 1.0}, {"y_Gyroscope__median": 1.0}, {
            "y_Gyroscope__median": 1.0}, {"y_Gyroscope__median": 1.0}])
    },
    "z_Gyroscope": {
        Feature.MEAN: DataFrame([{"z_Gyroscope__mean": 1.0}, {"z_Gyroscope__mean": 1.0}, {"z_Gyroscope__mean": 1.0}, {"z_Gyroscope__mean": 1.0}]),
        Feature.MEDIAN: DataFrame([{"z_Gyroscope__median": 1.0}, {"z_Gyroscope__median": 1.0}, {
            "z_Gyroscope__median": 1.0}, {"z_Gyroscope__median": 1.0}])
    }
}


def gather_features_and_labels() -> Tuple[DataFrame, List[str]]:
    result: List[DataFrame] = []
    for sensor_component in sensor_component_feature_dfs_4_2:
        for feature in sensor_component_feature_dfs_4_2[sensor_component]:
            result.append(sensor_component_feature_dfs_4_2[sensor_component][feature])
    labels = ["Shake", "Shake", "Rotate", "Rotate"]
    return (pd.concat(result, axis=1), labels)


rfc_default_hyperparameters = config_spaces[Classifier.RANDOM_FOREST_CLASSIFIER].get_default_configuration(
).get_dictionary()
for name, value in rfc_default_hyperparameters.items():
    if value == "None":
        rfc_default_hyperparameters[name] = None
    elif value == "True":
        rfc_default_hyperparameters[name] = True
    elif value == "False":
        rfc_default_hyperparameters[name] = False


(performance_metrics, columns, label_encoder, pipeline) = train(TrainingConfig(
    model_name="Model_4_2",
    sliding_window=SlidingWindow(window_size=4, sliding_step=2),
    perComponentConfigs={
        "x_Accelerometer": PerComponentConfig(
            features=[Feature.MINIMUM, Feature.MAXIMUM],
            pipeline_config=PipelineConfig(imputation=Imputation.MEAN_IMPUTATION,
                                           normalization=Normalization.MIN_MAX_SCALER)
        ),
        "y_Accelerometer": PerComponentConfig(
            features=[Feature.MINIMUM, Feature.MAXIMUM],
            pipeline_config=PipelineConfig(imputation=Imputation.LINEAR_INTERPOLATION,
                                           normalization=Normalization.NORMALIZER)
        ),
        "z_Accelerometer": PerComponentConfig(
            features=[Feature.MINIMUM, Feature.MAXIMUM],
            pipeline_config=PipelineConfig(imputation=Imputation.ZERO_INTERPOLATION,
                                           normalization=Normalization.STANDARD_SCALER)
        ),
        "x_Gyroscope": PerComponentConfig(
            features=[Feature.MINIMUM, Feature.MAXIMUM],
            pipeline_config=PipelineConfig(imputation=Imputation.MEAN_IMPUTATION,
                                           normalization=Normalization.MIN_MAX_SCALER)
        ),
        "y_Gyroscope": PerComponentConfig(
            features=[Feature.MINIMUM, Feature.MAXIMUM],
            pipeline_config=PipelineConfig(imputation=Imputation.LINEAR_INTERPOLATION,
                                           normalization=Normalization.NORMALIZER)
        ),
        "z_Gyroscope": PerComponentConfig(
            features=[Feature.MINIMUM, Feature.MAXIMUM],
            pipeline_config=PipelineConfig(imputation=Imputation.ZERO_INTERPOLATION,
                                           normalization=Normalization.STANDARD_SCALER)
        )
    },
    classifier=Classifier.RANDOM_FOREST_CLASSIFIER,
    hyperparameters=rfc_default_hyperparameters
))

print(performance_metrics)
