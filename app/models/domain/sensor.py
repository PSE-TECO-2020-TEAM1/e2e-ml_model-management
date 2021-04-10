from dataclasses import dataclass
from typing import List, Tuple

class SensorComponent(str):
    pass

def make_sensor_component(sensor_name: str, component: str) -> SensorComponent:
    return component + "_" + sensor_name

def reverse_sensor_component(sensor_component: SensorComponent) -> Tuple[str, str]:
    """
    Returns: (sensor, component)
    """
    temp = sensor_component.split("_")
    return (temp[1], temp[0])

@dataclass
class Sensor():
    sampling_rate: int
    components: List[SensorComponent]