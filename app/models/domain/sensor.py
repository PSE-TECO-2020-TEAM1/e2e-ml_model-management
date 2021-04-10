from dataclasses import dataclass
from typing import List, Tuple

class SensorComponent(str):
    pass

def make_sensor_component(sensor_name: str, component: str) -> SensorComponent:
    return component + "_" + sensor_name

def reverse_sensor_component(sensor_component: SensorComponent) -> Tuple[str, str]:
    temp = sensor_component.split("_")
    return (temp[0], temp[1])

@dataclass
class Sensor():
    sampling_rate: int
    components: List[SensorComponent]