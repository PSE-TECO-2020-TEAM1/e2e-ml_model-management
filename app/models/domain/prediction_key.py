from dataclasses import dataclass
from bson.objectid import ObjectId
from app.models.domain.db_doc import DbDocument

@dataclass
class PredictionKey(DbDocument):
    workspace_id: ObjectId
    model_id: ObjectId
    # TODO expiration: int