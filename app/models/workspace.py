from pydantic import BaseModel
from typing import Dict, Optional

class Workspace(BaseModel):
    _id: Optional[int] = None
    userId: int
    predictionIds: Dict[str, int] = {}