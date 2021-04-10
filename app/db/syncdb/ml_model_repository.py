from app.models.domain.sensor import SensorComponent
from app.db.error.non_existent_error import NonExistentError
from dataclasses import asdict
from app.models.domain.ml_model import MlModel
from pymongo.database import Database
from app.core.config import ML_MODEL_COLLECTION_NAME
from bson.objectid import ObjectId
import dacite


class MlModelRepository():
    def __init__(self, db: Database):
        self.collection = db[ML_MODEL_COLLECTION_NAME]

    def get_ml_model(self, ml_model_id: ObjectId) -> MlModel:
        ml_model = self.collection.find_one({"_id": ml_model_id})
        if ml_model is None:
            raise NonExistentError("Workspace with the given id does not exist")
        return dacite.from_dict(data_class=MlModel, data=ml_model, config=dacite.Config(cast=[SensorComponent]))

    def add_ml_model(self, ml_model: MlModel) -> ObjectId:
        ml_model_dict = ml_model.dict_for_db_insertion()
        return self.collection.insert_one(ml_model_dict).inserted_id
