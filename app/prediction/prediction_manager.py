from app.models.workspace import DataPoint, DataPointsPerSensor
from multiprocessing import Semaphore
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Any, Dict, List

from app.prediction.predictor import Predictor, PredictorEntry
from app.models.mongo_model import OID


# TODO killing child process after timeout (join?)

class PredictionUtil():
    def __init__(self, process: Process, semaphore: SemaphoreType, manager_end: Connection):
        self.process = process
        self.semaphore = semaphore
        self.manager_end = manager_end


class PredictionManager():
    def __init__(self):
        self.prediction_id_to_util: Dict[str, PredictionUtil] = {}

    async def spawn_predictor(self, prediction_id: OID, model_id: OID):
        predictor = Predictor(model_id)

        semaphore = Semaphore(0)
        (manager_end, predictor_end) = Pipe(duplex=True)
        process = Process(target=predictor.init_predictor_process, args=(semaphore, predictor_end))
        # TODO close processes on app shutdown (main.py)
        process.start()
        if not process.is_alive():
            predictor_end.close()
            manager_end.close()
            raise OSError("A process can't be spawned for the prediction.")

        predictor_end.close()
        self.prediction_id_to_util[prediction_id] = PredictionUtil(
            process=process, semaphore=semaphore, manager_end=manager_end)

    def submit_data(self, prediction_id: str, data: List[DataPointsPerSensor], start: int, end: int):
        entry = PredictorEntry(data, start, end)
        self.prediction_id_to_util[prediction_id].manager_end.send(entry)
        self.prediction_id_to_util[prediction_id].semaphore.release()

    def get_prediction_results(self, prediction_id: OID) -> List[str]:
        results: List[str] = []
        result_pipe = self.prediction_id_to_util[prediction_id].manager_end
        while result_pipe.poll():
            results.append(result_pipe.recv())
        return results


prediction_manager = PredictionManager()
