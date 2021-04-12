from app.ml.util.data_processing import split_to_data_windows
from tests.stubs.models.domain.feature_extraction_data import get_data_windows_df_4_2, get_data_windows_df_5_1, get_labels_of_data_windows_4_2, get_labels_of_data_windows_5_1
from app.models.domain.sliding_window import SlidingWindow
import pytest
from tests.stubs.models.domain.sample import get_interpolated_sample_stub_1, get_interpolated_sample_stub_2

@pytest.fixture
def sliding_window_5_1():
    return SlidingWindow(window_size=5, sliding_step=1)

@pytest.fixture
def sliding_window_4_2():
    return SlidingWindow(window_size=4, sliding_step=2)

@pytest.fixture
def interpolated_sample_list():
    return [get_interpolated_sample_stub_1(), get_interpolated_sample_stub_2()]

@pytest.fixture
def data_windows_df_5_1():
    return get_data_windows_df_5_1()

@pytest.fixture
def data_windows_df_4_2():
    return get_data_windows_df_4_2()

@pytest.fixture
def labels_of_data_windows_5_1():
    return get_labels_of_data_windows_5_1()

@pytest.fixture
def labels_of_data_windows_4_2():
    return get_labels_of_data_windows_4_2()

def test_split_to_data_windows(sliding_window_5_1, sliding_window_4_2, interpolated_sample_list, data_windows_df_5_1, data_windows_df_4_2, labels_of_data_windows_5_1, labels_of_data_windows_4_2):
    assert split_to_data_windows(sliding_window_5_1, interpolated_sample_list) == (data_windows_df_5_1, labels_of_data_windows_5_1)
    assert split_to_data_windows(sliding_window_4_2, interpolated_sample_list, )

