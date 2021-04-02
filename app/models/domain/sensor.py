from app.ml.training.parameters.features import Feature
from dataclasses import dataclass
from typing import List

class SensorComponent(str):
    pass


@dataclass
class Sensor():
    name: str
    sampling_rate: int
    components: List[SensorComponent]