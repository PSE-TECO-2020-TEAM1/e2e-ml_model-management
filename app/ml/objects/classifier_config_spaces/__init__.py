from app.ml.training.parameters.classifiers import Classifier
from app.ml.objects.classifier_config_spaces import (kneighbours_classifier,
                                                     mlp_classifier,
                                                     random_forest_classifier,
                                                     svc_classifier)

# Config spaces are all from https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification

config_spaces = {
    Classifier.KNEIGHBORS_CLASSIFIER: kneighbours_classifier.cs,
    Classifier.RANDOM_FOREST_CLASSIFIER: random_forest_classifier.cs,
    Classifier.SVC_CLASSIFIER: svc_classifier.cs,
    Classifier.MLP_CLASSIFIER: mlp_classifier.cs,
}
