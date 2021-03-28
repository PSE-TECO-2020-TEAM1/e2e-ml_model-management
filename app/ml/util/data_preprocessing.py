from app.ml.training.parameters.features import Feature
from app.models.domain.training_data_set import Sample
from pandas.core.frame import DataFrame
from app.models.domain.split_to_windows_data import LabeledDataWindows
from typing import Dict, List
from app.models.domain.sliding_window import SlidingWindow
import tsfresh
from tsfresh.feature_extraction import ComprehensiveFCParameters
import pandas


def split_to_data_windows(sliding_window: SlidingWindow, samples: List[Sample]) -> LabeledDataWindows:
    labels: List[str] = []
    data_windows: List[DataFrame] = []
    (window_size, sliding_step) = (sliding_window.window_size, sliding_window.sliding_step)
    for sample in samples:
        sensor_data_points = sample.data_frame
        # For each window in this sample, increment by the sliding step value starting from 0
        for window_offset in range(0, len(sensor_data_points.index) - window_size + 1, sliding_step):
            data_windows.append(sensor_data_points.iloc[window_offset:window_offset + window_size])
            labels.append(sample.label)
    return LabeledDataWindows(labels=labels, data_windows=data_windows)


def extract_features(features: List[Feature], data_windows: List[DataFrame]) -> Dict[Feature, DataFrame]:
    window_count = len(data_windows)
    newly_extracted_features: Dict[Feature, List[Dict[str, float]]] = {
        feature: [{} for _range_ in range(window_count)] for feature in features
    }
    settings = {key: ComprehensiveFCParameters()[key] for key in features}
    for data_window_index in range(window_count):
        data_windows[data_window_index]["id"] = data_window_index
    data_windows = pandas.concat(data_windows)
    extracted = tsfresh.extract_features(data_windows, column_id="id", default_fc_parameters=settings, pivot=False)
    # Split by columns features
    for i in range(len(extracted)):
        feature: Feature = features[i % len(features)]
        data_window_index = extracted[i][0]
        label = extracted[i][1]
        value = extracted[i][2]
        newly_extracted_features[feature][data_window_index][label] = value
    # Convert to DataFrame
    for feature in newly_extracted_features:
        newly_extracted_features[feature] = DataFrame(newly_extracted_features[feature])
    return newly_extracted_features
