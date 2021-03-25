import asyncio
from datetime import datetime
from multiprocessing import Semaphore
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Dict, List

from app.models.workspace import DataPointsPerSensor
from app.prediction.predictor import PredictionResult, Predictor, PredictorEntry
from app.models.mongo_model import OID


# TODO killing child process after timeout (join?)

class PredictionUtil():
    def __init__(self, process: Process, semaphore: SemaphoreType, manager_end: Connection):
        self.process = process
        self.semaphore = semaphore
        self.manager_end = manager_end
        self.last_access = datetime.utcnow()

class PredictionManager():
    def __init__(self):
        self.prediction_id_to_util: Dict[str, PredictionUtil] = {}
        self.clean_up_prediction_process_task = asyncio.create_task(self.clean_up_prediction_process())

    async def clean_up_prediction_process(self):
        while True:
            now = datetime.utcnow()
            for key, value in list(self.prediction_id_to_util.items()):
                if (now - value.last_access).seconds > 5 * 60: # remove if inactive for more than 5 minutes
                    print("cleaning up a process")
                    value.process.terminate()
                    del self.prediction_id_to_util[key]
            await asyncio.sleep(1 * 60) # check once per minute

    async def spawn_predictor(self, prediction_id: OID, model_id: OID):
        if prediction_id in self.prediction_id_to_util:
            return

        predictor = Predictor(model_id)
        semaphore = Semaphore(0)
        (manager_end, predictor_end) = Pipe(duplex=True)
        process = Process(target=predictor.init_predictor_process, args=(semaphore, predictor_end))
        # TODO close processes on app shutdown (main.py)
        process.start()
        print("asdasd")
        if not process.is_alive():
            predictor_end.close()
            manager_end.close()
            raise OSError("A process can't be spawned for the prediction.")

        predictor_end.close()
        self.prediction_id_to_util[prediction_id] = PredictionUtil(
            process=process, semaphore=semaphore, manager_end=manager_end)

    def submit_data(self, prediction_id: str, data: List[DataPointsPerSensor], start: int, end: int):
        entry = PredictorEntry(data, start, end)
        util = self.prediction_id_to_util[prediction_id]
        util.last_access = datetime.utcnow()
        util.manager_end.send(entry)
        util.semaphore.release()

    def get_prediction_results(self, prediction_id: OID) -> List[PredictionResult]:
        results: List[PredictionResult] = []
        result_pipe = self.prediction_id_to_util[prediction_id].manager_end
        while result_pipe.poll():
            results.append(result_pipe.recv())
        return results


_prediction_manager = None
def get_prediction_manager():
    global _prediction_manager
    if (_prediction_manager is None):
        _prediction_manager = PredictionManager()
    return _prediction_manager