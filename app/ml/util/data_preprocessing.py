from app.ml.training.parameters.features import Feature
from app.models.domain.training_data_set import InterpolatedSample
from pandas.core.frame import DataFrame
from typing import Dict, List, Tuple
from app.models.domain.sliding_window import SlidingWindow
import tsfresh
from tsfresh.feature_extraction import ComprehensiveFCParameters
import pandas


def split_to_data_windows(sliding_window: SlidingWindow, samples: List[InterpolatedSample]) -> Tuple[DataFrame, List[str]]:
    labels: List[str] = []
    data_windows: List[DataFrame] = []
    id_counter = 0
    (window_size, sliding_step) = (sliding_window.window_size, sliding_window.sliding_step)
    for sample in samples:
        sensor_data_points = sample.data_frame
        # For each window in this sample, increment by the sliding step value starting from 0
        for window_offset in range(0, len(sensor_data_points.index) - window_size + 1, sliding_step):
            data_window = sensor_data_points.iloc[window_offset:window_offset + window_size]
            data_window["id"] = id_counter
            id_counter += 1
            labels.append(sample.label)
    result = pandas.concat(data_windows)
    return (result, labels)


def extract_features(data_windows: DataFrame, features: List[Feature]) -> Dict[Feature, DataFrame]:
    settings = {key: ComprehensiveFCParameters()[key] for key in features}
    extracted = DataFrame, tsfresh.extract_features(data_windows, column_id="id", default_fc_parameters=settings)
    result = {}
    for feature_index in range(len(features)):
        feature = features[feature_index]
        result[feature] = extracted.iloc[:, [feature_index]]
    return result