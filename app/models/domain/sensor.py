from typing import List
from pydantic import BaseModel

class Sensor(BaseModel):
    name: str
    sampling_rate: int
    data_format: List[str]