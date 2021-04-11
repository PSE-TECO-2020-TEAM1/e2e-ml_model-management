from typing import Optional
from pydantic import BaseModel
from app.ml.training.training_state import TrainingState

class TrainingStateInResponse(BaseModel):
    state: TrainingState
    error: Optional[str]