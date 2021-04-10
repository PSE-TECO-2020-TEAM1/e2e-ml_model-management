from dataclasses import dataclass
from bson.objectid import ObjectId
from app.models.domain.db_doc import DbDoc

@dataclass
class PredictionKey(DbDoc):
    workspace_id: ObjectId
    model_id: ObjectId
    # TODO expiration: int