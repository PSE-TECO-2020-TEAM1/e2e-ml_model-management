from dataclasses import dataclass
from typing import Dict, List
from bson.objectid import ObjectId
from app.models.domain.sensor import Sensor

@dataclass
class MlModel():
    sensors: List[Sensor]
    column_order: List[str]
    label_code_to_label: Dict[str, str]