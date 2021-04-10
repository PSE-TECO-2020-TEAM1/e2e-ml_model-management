from bson.objectid import ObjectId
from app.models.domain.training_data_set import TrainingDataSet

from tests.stubs.models.domain.feature_extraction_data import get_feature_extraction_data_stub_5_1, get_feature_extraction_data_stub_4_2


def get_training_data_set_stub():
    return TrainingDataSet(
        last_modified=1617981582111,
        sample_list_file_ID=ObjectId("607070acc7559b9ccb3335fc"),
        feature_extraction_cache={"5_1": get_feature_extraction_data_stub_5_1(), "4_2": get_feature_extraction_data_stub_4_2()}
    )
