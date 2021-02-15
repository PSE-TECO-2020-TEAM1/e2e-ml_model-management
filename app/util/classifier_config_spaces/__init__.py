from typing import Dict

from app.util.classifier_config_spaces.kneighbours_classifier import \
    cs as kneighbors
from app.util.classifier_config_spaces.mlp_classifier import cs as mlp
from app.util.classifier_config_spaces.random_forest_classifier import \
    cs as random_forest
from app.util.classifier_config_spaces.svc_classifier import cs as svc
from app.util.training_parameters import Classifier
from ConfigSpace.configuration_space import ConfigurationSpace

config_spaces: Dict[Classifier, ConfigurationSpace] = {
    Classifier.KNEIGHBORS_CLASSIFIER: kneighbors,
    Classifier.RANDOM_FOREST_CLASSIFIER: random_forest,
    Classifier.SVC_CLASSIFIER: svc,
    Classifier.MLP_CLASSIFIER: mlp,
}


def get_config_space(classifier: Classifier) -> ConfigurationSpace:
    return config_spaces[classifier]
