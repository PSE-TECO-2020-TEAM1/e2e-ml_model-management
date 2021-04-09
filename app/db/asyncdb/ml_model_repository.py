from typing import Any
from app.models.domain.prediction_id import PredictionID
from app.core.config import ML_MODEL_COLLECTION_NAME
from app.db.error.non_existent_error import NonExistentError
from app.models.domain.sensor import SensorComponent
from app.models.domain.ml_model import MlModel
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson.objectid import ObjectId
import dacite

class MlModelRepository():
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[ML_MODEL_COLLECTION_NAME]

    async def get_ml_model(self, ml_model_id: ObjectId) -> MlModel:
        ml_model = await self.collection.find_one({"_id": ml_model_id})
        if ml_model is None:
            raise NonExistentError("ML Model with the given id does not exist")
        return dacite.from_dict(data_class=MlModel, data=ml_model, config=dacite.Config(cast=[SensorComponent]))

    async def add_prediction_id(self, ml_model_id: ObjectId, prediction_id: PredictionID):
        result = await self.collection.update_one({"_id": ml_model_id}, {"$push": {"prediction_IDs": prediction_id}})
        if not result:
            raise NonExistentError("ML Model with the given id does not exist")