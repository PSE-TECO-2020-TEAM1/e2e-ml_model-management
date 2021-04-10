from tests.stubs.models.domain.sample import interpolated_sample_stub_1, interpolated_sample_stub_2

from typing import Dict
from bson.objectid import ObjectId
import random_object_id
import pickle

class FileRepositoryStub():
    def __init__(self, db):
        self.files: Dict[ObjectId, bytes] = {}
        self.__insert_stubs__()

    def __insert_stubs__(self):
        sample_list = [interpolated_sample_stub_1, interpolated_sample_stub_2]
        self.files[ObjectId("607070acc7559b9ccb3335fc")] = pickle.dumps(sample_list)

    def get_file(self, id: ObjectId) -> bytes:
        if id not in self.files:
            # TODO raise error
            pass
        
        return self.files[id]

    def put_file(self, file: bytes) -> ObjectId:
        id: ObjectId(random_object_id.generate())
        self.files[id] = file
        return id

    def delete_file(self, id: ObjectId):
        self.files.pop(id)

    def replace_file(self, old_id: ObjectId, new_file: bytes) -> ObjectId:
        self.delete_file(old_id)
        return self.put_file(new_file)