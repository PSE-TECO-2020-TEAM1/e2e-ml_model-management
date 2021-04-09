from app.models.schemas.sample import SampleInSubmit
from app.ml.prediction.predictor import PredictionResult, Predictor, PredictorEntry
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
    last_access = field(default_factory=datetime.utcnow)


class PredictionManager():
    def __init__(self):
        self.prediction_id_to_util: Dict[str, PredictionUtil] = {}

    def initiate_clean_up_prediction_process(self):
        """
        Call at the program start to clean up the created processes periodically
        """
        # We cannot call this in the constructor. An event loop has to be set by the FastAPI first.
        asyncio.create_task(self.clean_up_prediction_process())

    async def clean_up_prediction_process(self):
        # TODO IMPORTANT at least manually test that this works
        while True:
            now = datetime.utcnow()
            for key, value in list(self.prediction_id_to_util.items()):
                if (now - value.last_access).seconds > 5 * 60:  # Remove if inactive for more than 5 minutes
                    print("cleaning up a process")
                    value.process.terminate()
                    del self.prediction_id_to_util[key]
            await asyncio.sleep(1 * 60)  # Check once per minute

    async def spawn_predictor(self, prediction_id: ObjectId, model_id: ObjectId):
        if prediction_id in self.prediction_id_to_util:
            return
        predictor = Predictor(model_id)
        semaphore = Semaphore(0)
        (manager_end, predictor_end) = Pipe(duplex=True)
        process = Process(target=predictor.init_predictor_process, args=(semaphore, predictor_end))
        # TODO IMPORTANT close processes on app shutdown (main.py)
        process.start()
        # Sanity check, not sure if necessary
        if not process.is_alive():
            predictor_end.close()
            manager_end.close()
            raise OSError("A process cannot be spawned for the prediction.")
        predictor_end.close()
        self.prediction_id_to_util[prediction_id] = PredictionUtil(process=process, semaphore=semaphore, manager_end=manager_end)

    def submit_data(self, prediction_id: str, data: SampleInSubmit):
        util = self.prediction_id_to_util[prediction_id]
        util.last_access = datetime.utcnow()
        util.manager_end.send(data)
        util.semaphore.release()

    def get_prediction_results(self, prediction_id: ObjectId) -> List[PredictionResult]:
        results: List[PredictionResult] = []
        result_pipe = self.prediction_id_to_util[prediction_id].manager_end
        while result_pipe.poll():
            results.append(result_pipe.recv())
        return results

prediction_manager = PredictionManager()
