from app.models.domain.training_data_set import TrainingDataSet

from tests.stubs.models.domain.feature_extraction_data import feature_extraction_data_stub_5_1, feature_extraction_data_stub_4_2

training_data_set_stub = TrainingDataSet(
    last_modified=1617981582,
    sample_list_file_ID="607070acc7559b9ccb3335fc",
    feature_extraction_cache={"5_1": feature_extraction_data_stub_5_1, "4_2": feature_extraction_data_stub_4_2}
)