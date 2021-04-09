from app.models.domain.sensor import SensorComponent
from app.ml.objects.feature.enum import Feature
from app.db.error.non_existant_error import NonExistentError
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
        self.collection.insert_one(asdict(ml_model))