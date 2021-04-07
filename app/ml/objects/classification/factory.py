from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from app.ml.objects.classification import Classifier


classifier_factory_dict = {
    Classifier.MLP_CLASSIFIER: lambda hyperparameters: MLPClassifier(**hyperparameters),
    Classifier.SVC_CLASSIFIER: lambda hyperparameters: SVC(**hyperparameters),
    Classifier.RANDOM_FOREST_CLASSIFIER: lambda hyperparameters: RandomForestClassifier(**hyperparameters),
    Classifier.KNEIGHBORS_CLASSIFIER: lambda hyperparameters: KNeighborsClassifier(**hyperparameters)
}

def get_classifier(classifier: Classifier, hyperparameters: Dict[str, Any]):
    return classifier_factory_dict[classifier](hyperparameters)