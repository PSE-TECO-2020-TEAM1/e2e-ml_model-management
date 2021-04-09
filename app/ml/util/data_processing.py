from app.models.domain.performance_metrics import PerformanceMetrics, SingleMetric
from app.ml.objects.feature import Feature
from app.models.domain.sample import InterpolatedSample
from pandas.core.frame import DataFrame
from typing import Dict, List, Tuple
from app.models.domain.sliding_window import SlidingWindow
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tsfresh
import pandas


def split_to_data_windows(sliding_window: SlidingWindow, samples: List[InterpolatedSample]) -> Tuple[DataFrame, List[str]]:
    labels_of_data_windows: List[str] = []
    data_windows: List[DataFrame] = []
    id_counter = 0
    (window_size, sliding_step) = (sliding_window.window_size, sliding_window.sliding_step)
    for sample in samples:
        sensor_data_points = sample.data_frame # List[Dict[SensorComponent, float]]
        # For each window in this sample, increment by the sliding step value starting from 0
        for window_offset in range(0, len(sensor_data_points.index) - window_size + 1, sliding_step):
            data_window = sensor_data_points.iloc[window_offset:window_offset + window_size, :].copy()
            data_window["id"] = id_counter
            id_counter += 1
            data_windows.append(data_window)
            labels_of_data_windows.append(sample.label)
    result = pandas.concat(data_windows, ignore_index=True)
    return (result, labels_of_data_windows)


# Extracts features of data windows of one sensor component
def extract_features(data_windows: DataFrame, features: List[Feature]) -> Dict[Feature, DataFrame]:
    settings = {key: ComprehensiveFCParameters()[key] for key in [str(feature.value).lower() for feature in features]}
    extracted: DataFrame = tsfresh.extract_features(data_windows, column_id="id", default_fc_parameters=settings)
    result = {}
    for feature_index in range(len(features)):
        feature = features[feature_index]
        result[feature] = extracted.iloc[:, [feature_index]]
    return result

def calculate_classification_report(test_labels: List[int], prediction_result: List[int], encoder: LabelEncoder) -> List[PerformanceMetrics]:
    report = classification_report(test_labels, prediction_result, output_dict=True)
    result: List[PerformanceMetrics] = []
    for encoded_label, performance_metrics in report.items():
        if (not encoded_label.isnumeric()):
            continue
        metrics = []
        for name, score in performance_metrics.items():
            metrics.append(SingleMetric(name=name, score=score))
        label = encoder.inverse_transform([int(encoded_label)])[0]
        result.append(PerformanceMetrics(label=label, metrics=metrics))
    return result
