from dataclasses import dataclass
from typing import List

class SensorComponent(str):
    pass

def make_sensor_component(sensor_name: str, component: str) -> SensorComponent:
    return component + "_" + sensor_name

@dataclass(frozen=True)
class Sensor():
    sampling_rate: int
    components: List[SensorComponent]