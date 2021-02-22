from app.prediction.predictor import Predictor
from app.db import db

from app.models.workspace import DataPoint
from app.models.mongo_model import OID
from typing import Any, Dict, List, Tuple


class PredictionManager():
    def __init__(self):
        self.model_id_to_pid: Dict[OID, int] = {}
        self.prediction_id_to_model_id_and_shm: Dict[str, Tuple[OID, Any]] = {} # TODO shared memory type

    async def spawn_predictor(self, prediction_id: OID, model_id: OID):
        model_doc = await db.get().ml_models.find_one({"_id": model_id})
        imputation_object = await 
        predictor = Predictor(window_size=model_doc["windowSize"], sliding_step=model_doc["slidingStep"], imputation_object=)


    def submit_data(self, prediction_id: OID, data_window: Dict[str, List[DataPoint]]):
        pass

    def get_predictions(self, prediction_id: OID) -> List[str]:
        pass