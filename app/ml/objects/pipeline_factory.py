from app.ml.objects.classification import Classifier
from app.ml.objects.classification.factory import get_classifier
from app.ml.objects.imputation import Imputation
from app.ml.objects.imputation.factory import get_imputer
from app.ml.objects.normalization import Normalization
from app.ml.objects.normalization.factory import get_normalizer
from sklearn.compose import make_column_selector
from app.ml.objects.column_transfomer import PandasColumnTransformer
from app.models.domain.sensor import SensorComponent
from typing import Dict
from app.models.domain.training_config import PipelineConfig
from sklearn.pipeline import Pipeline

MATCH_REST_REGEX = ".*"

def make_pipeline_from_config(config: Dict[SensorComponent, PipelineConfig], classifier: Classifier, hyperparameters) -> Pipeline:
    steps = []
    imputations = {}
    normalizations = {}
    for sensor_component in config.keys():
        imputations[sensor_component] = config[sensor_component].imputation
        normalizations[sensor_component] = config[sensor_component].normalization
    steps.append(("imputation", make_imputer_column_transformer(imputations)))
    steps.append(("normalization", make_normalizer_column_transformer(normalizations)))
    steps.append(("classification", get_classifier(classifier, hyperparameters)))
    return Pipeline(steps=steps)

def make_imputer_column_transformer(imputations: Dict[SensorComponent, Imputation]):
    transformers = []
    for sensor_component, imputation in imputations.items():
        selector = make_column_selector(pattern=sensor_component + MATCH_REST_REGEX)
        transformers.append((sensor_component, get_imputer(imputation), selector))
    return PandasColumnTransformer(transformers)

def make_normalizer_column_transformer(normalizations: Dict[SensorComponent, Normalization]):
    transformers = []
    for sensor_component, normalization in normalizations.items():
        selector = make_column_selector(pattern=sensor_component + MATCH_REST_REGEX)
        transformers.append((sensor_component, get_normalizer(normalization), selector))
    return PandasColumnTransformer(transformers)