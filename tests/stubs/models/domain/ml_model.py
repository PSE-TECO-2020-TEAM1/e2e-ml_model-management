from app.ml.objects.classification.factory import get_classifier
from app.ml.objects.column_transfomer import PandasColumnTransformer
from sklearn.pipeline import Pipeline
from tests.stubs.models.domain.feature_extraction_data import get_labels_of_data_windows_4_2, get_labels_of_data_windows_5_1
from app.models.domain.performance_metrics import PerformanceMetrics, SingleMetric
from app.ml.objects.classification.enum import Classifier
from app.ml.objects.feature.enum import Feature
from app.ml.objects.normalization.enum import Normalization
from app.ml.objects.imputation.enum import Imputation
from app.models.domain.sliding_window import SlidingWindow
from app.models.domain.training_config import PerComponentConfig, PipelineConfig, TrainingConfig
from app.models.domain.ml_model import MlModel
from app.ml.objects.classification.classifier_config_spaces.util import config_spaces
from sklearn.preprocessing import LabelEncoder
from app.ml.objects.imputation.factory import get_imputer
from app.ml.objects.normalization.factory import get_normalizer
from sklearn.compose import make_column_selector

MATCH_REST_REGEX = ".*"


def get_rfc_default_hyperparameters():
    rfc_default_hyperparameters = config_spaces[Classifier.RANDOM_FOREST_CLASSIFIER].get_default_configuration(
    ).get_dictionary()
    for name, value in rfc_default_hyperparameters.items():
        if value == "None":
            rfc_default_hyperparameters[name] = None
        elif value == "True":
            rfc_default_hyperparameters[name] = True
        elif value == "False":
            rfc_default_hyperparameters[name] = False
    return rfc_default_hyperparameters


def get_training_config_stub_5_1():
    return TrainingConfig(
        model_name="Model_5_1",
        sliding_window=SlidingWindow(window_size=5, sliding_step=1),
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
        hyperparameters=get_rfc_default_hyperparameters()
    )


def get_label_performance_metrics_stub_5_1():
    return [PerformanceMetrics(label='Rotate', metrics=[SingleMetric(name='precision', score=0.0), SingleMetric(name='recall', score=0.0), SingleMetric(name='f1-score', score=0.0), SingleMetric(
            name='support', score=1.0)]), PerformanceMetrics(label='Shake', metrics=[SingleMetric(name='precision', score=0.0), SingleMetric(name='recall', score=0.0), SingleMetric(name='f1-score', score=0.0), SingleMetric(name='support', score=0.0)])]


def get_column_order_stub_5_1():
    return ['x_Accelerometer__minimum', 'x_Accelerometer__maximum', 'y_Accelerometer__minimum', 'y_Accelerometer__maximum', 'z_Accelerometer__minimum',
            'z_Accelerometer__maximum', 'x_Gyroscope__minimum', 'x_Gyroscope__maximum', 'y_Gyroscope__minimum', 'y_Gyroscope__maximum', 'z_Gyroscope__minimum', 'z_Gyroscope__maximum']


def get_label_encoder_stub_5_1():
    return LabelEncoder().fit(get_labels_of_data_windows_5_1())


def get_pipeline_stub_5_1():
    return Pipeline(steps=[
        ("imputation", PandasColumnTransformer([
            ("x_Accelerometer", get_imputer(Imputation.MEAN_IMPUTATION),
             make_column_selector(pattern="x_Accelerometer" + MATCH_REST_REGEX)),
            ("y_Accelerometer", get_imputer(Imputation.LINEAR_INTERPOLATION),
             make_column_selector(pattern="y_Accelerometer" + MATCH_REST_REGEX)),
            ("z_Accelerometer", get_imputer(Imputation.ZERO_INTERPOLATION),
             make_column_selector(pattern="z_Accelerometer" + MATCH_REST_REGEX)),
            ("x_Gyroscope", get_imputer(Imputation.MEAN_IMPUTATION),
             make_column_selector(pattern="x_Gyroscope" + MATCH_REST_REGEX)),
            ("y_Gyroscope", get_imputer(Imputation.LINEAR_INTERPOLATION),
             make_column_selector(pattern="y_Gyroscope" + MATCH_REST_REGEX)),
            ("z_Gyroscope", get_imputer(Imputation.ZERO_INTERPOLATION),
             make_column_selector(pattern="z_Gyroscope" + MATCH_REST_REGEX)),
        ])),
        ("normalization", PandasColumnTransformer([
            ("x_Accelerometer", get_normalizer(Normalization.MIN_MAX_SCALER),
             make_column_selector(pattern="x_Accelerometer" + MATCH_REST_REGEX)),
            ("y_Accelerometer", get_normalizer(Normalization.NORMALIZER),
             make_column_selector(pattern="y_Accelerometer" + MATCH_REST_REGEX)),
            ("z_Accelerometer", get_normalizer(Normalization.STANDARD_SCALER),
             make_column_selector(pattern="z_Accelerometer" + MATCH_REST_REGEX)),
            ("x_Gyroscope", get_normalizer(Normalization.MIN_MAX_SCALER),
             make_column_selector(pattern="x_Gyroscope" + MATCH_REST_REGEX)),
            ("y_Gyroscope", get_normalizer(Normalization.NORMALIZER),
             make_column_selector(pattern="y_Gyroscope" + MATCH_REST_REGEX)),
            ("z_Gyroscope", get_normalizer(Normalization.STANDARD_SCALER),
             make_column_selector(pattern="z_Gyroscope" + MATCH_REST_REGEX))
        ])),
        ("classification", get_classifier(Classifier.RANDOM_FOREST_CLASSIFIER, get_rfc_default_hyperparameters()))
    ])


def get_ml_model_stub_5_1() -> MlModel:
    return MlModel(
        _id="60736013b961d239c76711a3",
        config=get_training_config_stub_5_1(),
        label_performance_metrics=get_label_performance_metrics_stub_5_1(),
        column_order=get_column_order_stub_5_1(),
        label_encoder_object_file_ID="6073604f8741014a0cd09780",
        pipeline_object_file_ID="607360560b3beb16ac03de77"
    )


def get_training_config_stub_4_2():
    return TrainingConfig(
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
        hyperparameters=get_rfc_default_hyperparameters()
    )


def get_label_performance_metrics_stub_4_2():
    return [PerformanceMetrics(label='Rotate', metrics=[SingleMetric(name='precision', score=0.0), SingleMetric(name='recall', score=0.0), SingleMetric(name='f1-score', score=0.0), SingleMetric(name='support', score=1.0)]), PerformanceMetrics(label='Shake', metrics=[SingleMetric(name='precision', score=0.0), SingleMetric(name='recall', score=0.0), SingleMetric(name='f1-score', score=0.0), SingleMetric(name='support', score=0.0)])]


def get_label_encoder_stub_4_2():
    return LabelEncoder().fit(get_labels_of_data_windows_4_2())


def get_column_order_stub_4_2():
    return ['x_Accelerometer__minimum', 'x_Accelerometer__maximum', 'y_Accelerometer__minimum', 'y_Accelerometer__maximum', 'z_Accelerometer__minimum',
            'z_Accelerometer__maximum', 'x_Gyroscope__minimum', 'x_Gyroscope__maximum', 'y_Gyroscope__minimum', 'y_Gyroscope__maximum', 'z_Gyroscope__minimum', 'z_Gyroscope__maximum']


def get_pipeline_stub_4_2():
    return Pipeline(steps=[
        ("imputation", PandasColumnTransformer([
            ("x_Accelerometer", get_imputer(Imputation.MEAN_IMPUTATION),
             make_column_selector(pattern="x_Accelerometer" + MATCH_REST_REGEX)),
            ("y_Accelerometer", get_imputer(Imputation.LINEAR_INTERPOLATION),
             make_column_selector(pattern="y_Accelerometer" + MATCH_REST_REGEX)),
            ("z_Accelerometer", get_imputer(Imputation.ZERO_INTERPOLATION),
             make_column_selector(pattern="z_Accelerometer" + MATCH_REST_REGEX)),
            ("x_Gyroscope", get_imputer(Imputation.MEAN_IMPUTATION),
             make_column_selector(pattern="x_Gyroscope" + MATCH_REST_REGEX)),
            ("y_Gyroscope", get_imputer(Imputation.LINEAR_INTERPOLATION),
             make_column_selector(pattern="y_Gyroscope" + MATCH_REST_REGEX)),
            ("z_Gyroscope", get_imputer(Imputation.ZERO_INTERPOLATION),
             make_column_selector(pattern="z_Gyroscope" + MATCH_REST_REGEX)),
        ])),
        ("normalization", PandasColumnTransformer([
            ("x_Accelerometer", get_normalizer(Normalization.MIN_MAX_SCALER),
             make_column_selector(pattern="x_Accelerometer" + MATCH_REST_REGEX)),
            ("y_Accelerometer", get_normalizer(Normalization.NORMALIZER),
             make_column_selector(pattern="y_Accelerometer" + MATCH_REST_REGEX)),
            ("z_Accelerometer", get_normalizer(Normalization.STANDARD_SCALER),
             make_column_selector(pattern="z_Accelerometer" + MATCH_REST_REGEX)),
            ("x_Gyroscope", get_normalizer(Normalization.MIN_MAX_SCALER),
             make_column_selector(pattern="x_Gyroscope" + MATCH_REST_REGEX)),
            ("y_Gyroscope", get_normalizer(Normalization.NORMALIZER),
             make_column_selector(pattern="y_Gyroscope" + MATCH_REST_REGEX)),
            ("z_Gyroscope", get_normalizer(Normalization.STANDARD_SCALER),
             make_column_selector(pattern="z_Gyroscope" + MATCH_REST_REGEX))
        ])),
        ("classification", get_classifier(Classifier.RANDOM_FOREST_CLASSIFIER, get_rfc_default_hyperparameters()))
    ])


def get_ml_model_stub_4_2() -> MlModel:
    return MlModel(
        _id="607367362d98418cae5a1522",
        config=get_training_config_stub_4_2(),
        label_performance_metrics=get_label_performance_metrics_stub_4_2(),
        column_order=get_column_order_stub_4_2(),
        label_encoder_object_file_ID="60736744c795e80494dbc3e9",
        pipeline_object_file_ID="60736749a4c260179e9be00d"
    )
