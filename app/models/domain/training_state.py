from dataclasses import dataclass
from enum import Enum

@dataclass
class TrainingState(str, Enum):
    NO_ACTIVE_TRAINING = "no active training",
    FEATURE_EXTRACTION = "feature extraction",
    MODEL_FITTING = "model fitting",
    MODEL_TRAINING = "model training",
    CLASSIFICATION_REPORT = "classification report"
    # TODO add error states
    