from typing import Any, Dict, List, Tuple

from app.util.classifier_config_spaces.kneighbours_classifier import cs as kneighbors
from app.util.classifier_config_spaces.mlp_classifier import cs as mlp
from app.util.classifier_config_spaces.random_forest_classifier import cs as random_forest
from app.util.classifier_config_spaces.svc_classifier import cs as svc
from app.util.training_parameters import Classifier
from app.models.responses import ClassifierSelection


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter, Constant

config_spaces: Dict[Classifier, ConfigurationSpace] = {
    Classifier.KNEIGHBORS_CLASSIFIER: kneighbors,
    Classifier.RANDOM_FOREST_CLASSIFIER: random_forest,
    Classifier.SVC_CLASSIFIER: svc,
    Classifier.MLP_CLASSIFIER: mlp,
}


def get_classifiers_with_hyperparameters() -> List[Classifier]:
    # return value: List[(classifier, hyperparameters, conditions)]
    res = []
    for classifier in Classifier:
        res.append(ClassifierSelection(classifier=classifier, hyperparameters=__get_hyperparameters(
            classifier), conditions=__get_conditions(classifier)))
    return res


def __get_hyperparameters(classifier: Classifier) -> Dict[str, Dict[str, Any]]:
    res = {}
    hyperparameters = config_spaces[classifier].get_hyperparameters_dict()
    for name, hyperparameter in hyperparameters.items():
        if isinstance(hyperparameter, CategoricalHyperparameter):
            res[name] = {"choices": hyperparameter.choices, "default_value": hyperparameter.default_value}
        elif isinstance(hyperparameter, Constant):
            res[name] = {"value": hyperparameter.value}
        elif isinstance(hyperparameter, UniformIntegerHyperparameter) or isinstance(hyperparameter, UniformFloatHyperparameter):
            res[name] = {"lower": hyperparameter.lower, "upper": hyperparameter.upper,
                         "default_value": hyperparameter.default_value}
        else:
            raise NotImplementedError(type(hyperparameter).__name__ + ' is not supported')
        
        res[name]["type"] = type(hyperparameter).__name__
    return res


def __get_conditions(classifier: Classifier) -> List[str]:
    return [str(condition) for condition in config_spaces[classifier].get_conditions()]
