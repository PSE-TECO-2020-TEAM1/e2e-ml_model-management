from enum import Enum
from app.models.domain.prediction_key import PredictionKey
from app.core.config import ML_MODEL_COLLECTION_NAME, PREDICTION_ID_COLLECTION_NAME
from app.db.error.non_existent_error import NonExistentError
from app.models.domain.sensor import SensorComponent
from app.models.domain.ml_model import MlModel
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from bson.objectid import ObjectId
import dacite

class MlModelRepository():
    def __init__(self, db: AsyncIOMotorDatabase):
        self.ml_model_collection: AsyncIOMotorCollection = db[ML_MODEL_COLLECTION_NAME]
        self.prediction_key_collection: AsyncIOMotorCollection = db[PREDICTION_ID_COLLECTION_NAME]

    async def get_ml_model(self, ml_model_id: ObjectId) -> MlModel:
        ml_model = await self.ml_model_collection.find_one({"_id": ml_model_id})
        if ml_model is None:
            raise NonExistentError("ML Model with the given id does not exist")
        return dacite.from_dict(data_class=MlModel, data=ml_model, config=dacite.Config(cast=[SensorComponent, Enum]))

    async def get_prediction_key(self, prediction_key_id: ObjectId) -> PredictionKey:
        prediction_key = await self.prediction_key_collection.find_one({"_id": prediction_key_id})
        if not prediction_key:
            raise NonExistentError("A prediction key with the given ID does not exist")
        return dacite.from_dict(data_class=PredictionKey, data=prediction_key)

    async def add_prediction_key(self, prediction_key: PredictionKey) -> ObjectId:
        prediction_key_dict = prediction_key.dict_for_db_insertion()
        result = await self.prediction_key_collection.insert_one(prediction_key_dict)
        if not result:
            raise NonExistentError("Could not create a prediction ID because of a database issue")
        return result.inserted_id