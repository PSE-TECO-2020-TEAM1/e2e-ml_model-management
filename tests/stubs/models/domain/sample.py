from typing import Dict, List

from pandas.core.frame import DataFrame
from app.models.domain.sensor import SensorComponent
from app.models.domain.sample import InterpolatedSample

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
