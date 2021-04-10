from enum import Enum

class TrainingState(str, Enum):
    NO_ACTIVE_TRAINING = "NO_ACTIVE_TRAINING",
    TRAINING_INITIATED = "TRAINING_INITIATED"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION",
    MODEL_TRAINING = "MODEL_TRAINING",
    CLASSIFICATION_REPORT = "CLASSIFICATION_REPORT"
    # TODO add error states
    