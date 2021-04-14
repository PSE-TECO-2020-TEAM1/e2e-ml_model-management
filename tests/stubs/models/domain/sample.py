from typing import Dict, List

from pandas.core.frame import DataFrame
from app.models.domain.sensor import SensorComponent
from app.models.domain.sample import InterpolatedSample
from app.core.config import LABEL_OUTSIDE_OF_TIMEFRAMES

# TODO replace values with better ones
sample_stub_1: List[Dict[SensorComponent, float]] = [
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0}
]

# TODO replace values with better ones
sample_stub_2: List[Dict[SensorComponent, float]] = [
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0}
]

sample_stub_3: List[Dict[SensorComponent, float]] = [
    {"x_Accelerometer": 0.0, "y_Accelerometer": 0.0, "z_Accelerometer": 0.0,
     "x_Gyroscope": 0.0, "y_Gyroscope": 0.0, "z_Gyroscope": 0.0}
]

sample_stub_4: List[Dict[SensorComponent, float]] = [
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0}
]

sample_stub_5: List[Dict[SensorComponent, float]] = [
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0},
    {"x_Accelerometer": 1.0, "y_Accelerometer": 1.0, "z_Accelerometer": 1.0,
        "x_Gyroscope": 1.0, "y_Gyroscope": 1.0, "z_Gyroscope": 1.0}
]


def get_interpolated_sample_stub_1():
    return InterpolatedSample(
        label="Shake",
        data_frame=DataFrame(sample_stub_1)
    )


def get_interpolated_sample_stub_2():
    return InterpolatedSample(
        label="Rotate",
        data_frame=DataFrame(sample_stub_2)
    )

def get_interpolated_sample_stub_3():
    return InterpolatedSample(
        label=LABEL_OUTSIDE_OF_TIMEFRAMES,
        data_frame=DataFrame(sample_stub_3)
    )

def get_interpolated_sample_stub_4():
    return InterpolatedSample(
        label=LABEL_OUTSIDE_OF_TIMEFRAMES,
        data_frame=DataFrame(sample_stub_4)
    )

def get_interpolated_sample_stub_5():
    return InterpolatedSample(
        label=LABEL_OUTSIDE_OF_TIMEFRAMES,
        data_frame=DataFrame(sample_stub_5)
    )
