from tests.stubs.models.domain.sample import get_interpolated_sample_stub_1, get_interpolated_sample_stub_2
from tests.stubs.models.domain.feature_extraction_data import (get_feature_extraction_data_stub_5_1, get_data_windows_df_5_1, get_labels_of_data_windows_5_1,
                                                               get_sensor_component_feature_dfs_5_1, get_feature_extraction_data_stub_4_2, get_data_windows_df_4_2, get_labels_of_data_windows_4_2, get_sensor_component_feature_dfs_4_2)
from tests.stubs.models.domain.training_data_set import get_training_data_set_stub

from app.db.error.non_existent_error import NonExistentError

from typing import Dict
from bson.objectid import ObjectId
import random_object_id
import pickle


class FileRepositoryStub():
    def __init__(self, init: Dict[ObjectId, bytes]):
        self.files = init

    def get_file(self, id: ObjectId) -> bytes:
        if id not in self.files:
            raise NonExistentError("The request file with the id " + str(id) + " does not exist")

        return self.files[id]

    def put_file(self, file: bytes) -> ObjectId:
        id = ObjectId(random_object_id.generate())
        self.files[id] = file
        return id

    def delete_file(self, id: ObjectId):
        self.files.pop(id)

    def replace_file(self, old_id: ObjectId, new_file: bytes) -> ObjectId:
        self.delete_file(old_id)
        return self.put_file(new_file)
