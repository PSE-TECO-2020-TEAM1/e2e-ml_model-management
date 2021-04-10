from app.models.domain.ml_model import MlModel
from app.db.error.non_existent_error import NonExistentError

from typing import Dict
from bson.objectid import ObjectId
import random_object_id


class MlModelRepositoryStub():
    def __init__(self, init: Dict[ObjectId, MlModel]):
        self.models = init

    def get_ml_model(self, id: ObjectId) -> MlModel:
        if id not in self.models:
            raise NonExistentError("Workspace with the given id does not exist")
            pass
        
        return self.models[id]

    def add_ml_model(self, ml_model: MlModel) -> ObjectId:
        id = ObjectId(random_object_id.generate())
        self.models[id] = ml_model
        return id
