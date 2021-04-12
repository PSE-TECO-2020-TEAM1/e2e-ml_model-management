from typing import Any, Tuple

from app.ml.objects.classification.enum import Classifier
from app.ml.objects.classification.classifier_config_spaces.util import config_spaces, get_conditions, get_hyperparameters, validate_and_parse_hyperparameters
import pytest


@pytest.fixture
def classifier():
    return Classifier.RANDOM_FOREST_CLASSIFIER

@pytest.fixture
def rfc_hyperparameters():
    return {
        "criterion": {"type": "CategoricalHyperparameter", "choices": ("gini", "entropy"), "default_value": "gini"},
        "max_features": {"type": "UniformFloatHyperparameter", "lower": 0., "upper": 1., "default_value": 0.5},
        "max_depth": {"type": "Constant", "value": "None"},
        "min_samples_split": {"type": "UniformIntegerHyperparameter", "lower": 2, "upper": 20, "default_value": 2},
        "min_samples_leaf": {"type": "UniformIntegerHyperparameter", "lower": 1, "upper": 20, "default_value": 1},
        "min_weight_fraction_leaf": {"type": "Constant", "value": 0.},
        "max_leaf_nodes": {"type": "Constant", "value": "None"},
        "min_impurity_decrease": {"type": "Constant", "value": 0.0},
        "bootstrap": {"type": "CategoricalHyperparameter", "choices": ("True", "False"), "default_value": "True"}
    }


@pytest.fixture
def hyperparameters_and_values(classifier):
    return config_spaces[classifier].get_default_configuration().get_dictionary()


@pytest.fixture
def one_hyperparameter_and_value(hyperparameters_and_values) -> Tuple[str, Any]:
    return list(hyperparameters_and_values.items())[0]


def test_validate_and_parse_hyperparameters_with_valid_configuration(classifier, hyperparameters_and_values):
    print(hyperparameters_and_values)
    validate_and_parse_hyperparameters(classifier, hyperparameters_and_values)


def test_validate_and_parse_hyperparameters_with_invalid_classifier(hyperparameters_and_values):
    with pytest.raises(ValueError, match="Invalid classifier name"):
        validate_and_parse_hyperparameters("Elite Classifier", hyperparameters_and_values)


def test_validate_and_parse_hyperparameters_with_invalid_hyperparameters(classifier, hyperparameters_and_values, one_hyperparameter_and_value):
    hyperparameters_and_values[one_hyperparameter_and_value[0]] = "This can't be a valid value"
    with pytest.raises(ValueError):
        validate_and_parse_hyperparameters(classifier, hyperparameters_and_values)

    hyperparameters_and_values[one_hyperparameter_and_value[0]] = one_hyperparameter_and_value[1]
    hyperparameters_and_values["This can't be a hyperparameter"] = 0
    with pytest.raises(ValueError):
        validate_and_parse_hyperparameters(classifier, hyperparameters_and_values)

def test_get_hyperparameters(classifier, rfc_hyperparameters):
    assert get_hyperparameters(classifier) == rfc_hyperparameters

def test_get_conditions(classifier):
    assert get_conditions(classifier) == []
    # TODO test with a classifier with conditions