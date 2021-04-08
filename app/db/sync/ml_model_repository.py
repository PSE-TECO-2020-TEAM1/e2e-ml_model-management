from dataclasses import asdict
from app.models.domain.ml_model import MlModel
from pymongo.database import Database
from app.core.config import ML_MODEL_COLLECTION_NAME
from bson.objectid import ObjectId


class MlModelRepository():
    def __init__(self, db: Database):
        self.collection = db[ML_MODEL_COLLECTION_NAME]
    
    def add_ml_model(self, ml_model: MlModel) -> ObjectId:
        self.collection.insert_one(asdict(ml_model))
