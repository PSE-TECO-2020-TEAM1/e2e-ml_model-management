from app.models.domain.ml_model import MlModel

from typing import Dict
from bson.objectid import ObjectId
import random_object_id


class MlModelRepositoryStub():
    def __init__(self, db):
        self.models: Dict[ObjectId, MlModel] = {}
        self.__insert_stubs__()

    def __insert_stubs__(self):
        # TODO add stub models
        pass

    def get_ml_model(self, id: ObjectId) -> MlModel:
        if id not in self.models:
            # TODO raise error
            pass
        
        return self.models[id]

    def add_ml_model(self, ml_model: MlModel) -> ObjectId:
        id = ObjectId(random_object_id.generate())
        self.models[id] = ml_model
        return id
