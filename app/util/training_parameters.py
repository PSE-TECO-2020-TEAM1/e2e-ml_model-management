from enum import Enum

class Classifier(dict, Enum):
    MLP_CLASSIFIER = "MLP_CLASSIFIER"
    SV_CLASSIFIER = "SV_CLASSIFIER"
    RANDOM_FOREST_CLASSIFIER = "SV_CLASSIFIER"
    KNEIGHBORS_CLASSIFIER = "KNEIGHBORS_CLASSIFIER"

#TODO hyperparameters enum for each classifier