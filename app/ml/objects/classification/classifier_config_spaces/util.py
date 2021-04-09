from typing import Any, Dict, List
from ConfigSpace import Configuration
from app.ml.objects.classification import Classifier
from app.ml.objects.classification.classifier_config_spaces import (kneighbours_classifier,
                                                                    mlp_classifier,
                                                                    random_forest_classifier,
                                                                    svc_classifier)
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter, Constant        

# Config spaces are all from https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification

config_spaces = {
    Classifier.KNEIGHBORS_CLASSIFIER: kneighbours_classifier.cs,
    Classifier.RANDOM_FOREST_CLASSIFIER: random_forest_classifier.cs,
    Classifier.SVC_CLASSIFIER: svc_classifier.cs,
    Classifier.MLP_CLASSIFIER: mlp_classifier.cs,
}

def validate_hyperparameters(classifier: Classifier, hyperparameters: Dict[str, Any]):
    # Raises an error iff not valid
    Configuration(config_spaces[classifier], values=hyperparameters)

def get_hyperparameters(classifier: Classifier) -> Dict[str, Any]:
    res = {}
    hyperparameters = config_spaces[classifier].get_hyperparameters_dict()
    for name, hyperparameter in hyperparameters.items():
        if isinstance(hyperparameter, CategoricalHyperparameter):
            res[name] = {"choices": hyperparameter.choices, "default_value": hyperparameter.default_value}
        elif isinstance(hyperparameter, Constant):
            res[name] = {"value": hyperparameter.value}
        elif isinstance(hyperparameter, UniformIntegerHyperparameter) or isinstance(hyperparameter, UniformFloatHyperparameter):
            res[name] = {"lower": hyperparameter.lower, "upper": hyperparameter.upper, "default_value": hyperparameter.default_value}
        else:
            raise NotImplementedError(type(hyperparameter).__name__ + ' is not supported')
        res[name]["type"] = type(hyperparameter).__name__
    return res

def get_conditions(classifier: Classifier) -> List[str]:
    return [str(condition) for condition in config_spaces[classifier].get_conditions()]