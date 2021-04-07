from dataclasses import dataclass
from pandas import DataFrame

@dataclass
class InterpolatedSample():
    label: str
    data_frame: DataFrame
