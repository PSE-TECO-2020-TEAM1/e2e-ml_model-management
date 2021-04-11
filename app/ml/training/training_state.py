from enum import Enum


class TrainingState(str, Enum):
    # Default state
    NO_TRAINING_YET = "NO_TRAINING_YET",
    # In progress training states
    TRAINING_INITIATED = "TRAINING_INITIATED"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION",
    MODEL_TRAINING = "MODEL_TRAINING",
    CLASSIFICATION_REPORT = "CLASSIFICATION_REPORT"
    # Successful state
    TRAINING_SUCCESSFUL = "TRAINING_SUCCESSFUL"
    # Error states
    WORKSPACE_MANAGEMENT_CONNECTION_ERROR = "WORKSPACE_MANAGEMENT_CONNECTION_ERROR"
    NO_SAMPLE_ERROR = "NO_SAMPLE_ERROR"
    TRAINING_ERROR = "TRAINING_ERROR"

    def is_in_progress_training_state(self):
        return self in _in_progress_training_states

    def is_error_state(self):
        return self in _error_states


_in_progress_training_states = [TrainingState.TRAINING_INITIATED,
                    TrainingState.FEATURE_EXTRACTION,
                    TrainingState.MODEL_TRAINING,
                    TrainingState.CLASSIFICATION_REPORT]

_error_states = [TrainingState.WORKSPACE_MANAGEMENT_CONNECTION_ERROR,
                 TrainingState.NO_SAMPLE_ERROR,
                 TrainingState.TRAINING_ERROR]
