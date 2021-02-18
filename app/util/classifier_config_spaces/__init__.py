from typing import Dict, List, Tuple, Union

from app.util.classifier_config_spaces.kneighbours_classifier import cs as kneighbors
from app.util.classifier_config_spaces.mlp_classifier import cs as mlp
from app.util.classifier_config_spaces.random_forest_classifier import cs as random_forest
from app.util.classifier_config_spaces.svc_classifier import cs as svc
from app.util.training_parameters import Classifier
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter, Constant

config_spaces: Dict[Classifier, ConfigurationSpace] = {
    Classifier.KNEIGHBORS_CLASSIFIER: kneighbors,
    Classifier.RANDOM_FOREST_CLASSIFIER: random_forest,
    Classifier.SVC_CLASSIFIER: svc,
    Classifier.MLP_CLASSIFIER: mlp,
}


def get_config_space(classifier: Classifier) -> ConfigurationSpace:
    return config_spaces[classifier]

# List[(classifier, hyperparameters, conditions)]


def get_classifiers_with_hyperparameters() -> List[Tuple[Classifier, Dict[str, Dict[str, any]], List[str]]]:
    res = []
    for classifier in Classifier:
        res.append((classifier, get_hyperparameters(classifier), get_conditions(classifier)))
    return res


def get_hyperparameters(classifier: Classifier) -> Dict[str, Dict[str, Union[any]]]:
    res = {}
    hyperparameters = config_spaces[classifier].get_hyperparameters_dict()
    for name, hyperparameter in hyperparameters.items():
        if isinstance(hyperparameter, CategoricalHyperparameter):
            res[name] = {"choices": hyperparameter.choices, "default_value": hyperparameter.default_value}
        elif isinstance(hyperparameter, Constant):
            res[name] = {"value": hyperparameter.value}
        else:
            res[name] = {"lower": hyperparameter.lower, "upper": hyperparameter.upper,
                         "default_value": hyperparameter.default_value}

    pass


def get_conditions(classifier: Classifier) -> List[str]:
    return [str(condition) for condition in config_spaces[classifier].get_conditions()]
