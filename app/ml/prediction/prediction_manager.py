from app.core.config import PREDICTION_PROCESS_SCAN_INTERVAL_IN_SECONDS, PREDICTION_PROCESS_TIMEOUT_IN_SECONDS
from app.db.syncdb import create_sync_db
from app.ml.prediction.data_set_manager import DataSetManager
from app.models.schemas.prediction_data import PredictionData
from app.ml.prediction.predictor import Predictor
import asyncio
from datetime import datetime
from multiprocessing import Semaphore
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Dict, List
from dataclasses import dataclass, field
from bson.objectid import ObjectId

@dataclass
class PredictionUtil():
    process: Process
    semaphore: SemaphoreType
    manager_end: Connection
    last_access: datetime = field(default_factory=datetime.utcnow)


class PredictionManager():
    def __init__(self):
        self.prediction_id_to_util: Dict[ObjectId, PredictionUtil] = {}

    def initiate_clean_up_prediction_process(self):
        """
        Call at the program start to clean up the created processes periodically
        """
        # We cannot call this in the constructor. An event loop has to be set by the FastAPI first.
        asyncio.create_task(self.clean_up_prediction_process())

    async def clean_up_prediction_process(self):
        while True:
            now = datetime.utcnow()
            for key, value in list(self.prediction_id_to_util.items()):
                if (now - value.last_access).seconds > PREDICTION_PROCESS_TIMEOUT_IN_SECONDS:
                    print("Cleaning up a prediction process...")
                    value.process.terminate()
                    del self.prediction_id_to_util[key]
            await asyncio.sleep(PREDICTION_PROCESS_SCAN_INTERVAL_IN_SECONDS)

    def spawn_predictor(self, workspace_id: ObjectId, prediction_id: ObjectId, model_id: ObjectId):
        if prediction_id in self.prediction_id_to_util:
            return
        predictor = Predictor(DataSetManager(workspace_id, model_id), create_db=create_sync_db)
        semaphore = Semaphore(0)
        (manager_end, predictor_end) = Pipe(duplex=True)
        process = Process(target=predictor.init_predictor_process, args=(semaphore, predictor_end))
        process.start()
        # Sanity check, not sure if necessary
        if not process.is_alive():
            predictor_end.close()
            manager_end.close()
            raise OSError("A process cannot be spawned for the prediction.")
        predictor_end.close()
        self.prediction_id_to_util[prediction_id] = PredictionUtil(process=process, semaphore=semaphore, manager_end=manager_end)

    def submit_data(self, prediction_data: PredictionData):
        if prediction_data.predictionId not in self.prediction_id_to_util:
            raise ValueError("There is no prediction process in progress with the given id, start a prediction first")
        util = self.prediction_id_to_util[prediction_data.predictionId]
        util.last_access = datetime.utcnow()
        util.manager_end.send(prediction_data.sample)
        util.semaphore.release()

    def get_prediction_results(self, prediction_id: ObjectId) -> List[str]:
        if prediction_id not in self.prediction_id_to_util:
            raise ValueError("There is no prediction process in progress with the given id, start a prediction first")
        results: List[str] = []
        result_pipe = self.prediction_id_to_util[prediction_id].manager_end
        while result_pipe.poll():
            results += result_pipe.recv()
        return results

    def terminate_prediction_processes(self):
        for key, value in list(self.prediction_id_to_util.items()):
            value.process.terminate()
            del self.prediction_id_to_util[key]

prediction_manager = PredictionManager()
