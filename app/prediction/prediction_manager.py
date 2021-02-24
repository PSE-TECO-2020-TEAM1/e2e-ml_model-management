from app.models.ml_model import MlModel
from multiprocessing import Semaphore
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Any, Dict, List

from app.prediction.predictor import Predictor
from app.db import db
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

    async def spawn_predictor(self, prediction_id: OID, model_id: OID, label_code_to_label: Dict[int, str]):
        predictor = Predictor(model_id, label_code_to_label=label_code_to_label)

        semaphore = Semaphore(0)
        (manager_end, predictor_end) = Pipe(duplex=True)
        process = Process(target=predictor.for_process, args=(semaphore, predictor_end))
        #process.daemon = True [extractfeatures predictorda tsfresh process spawnliyo paralel yapmak icin ama daemonic process spawnlayamiyo ondan fastapi on shutdown'da, burada processleri terminateleyip joinleeyn bi method olusturup onu cagirmamiz gerekiyo]
        process.start()
        if not process.is_alive():
            raise OSError("A process can't be spawned for the prediction.")
        #TODO close predictor_end
        self.prediction_id_to_util[prediction_id] = PredictionUtil(
            process=process, semaphore=semaphore, manager_end=manager_end)

    def submit_data(self, prediction_id: str, data_window: Dict[str, List[Any]]):
        self.prediction_id_to_util[prediction_id].manager_end.send(data_window)
        print("----------")
        self.prediction_id_to_util[prediction_id].semaphore.release()
        print(self.prediction_id_to_util[prediction_id].manager_end.recv())
        print("----------")

    def get_predictions(self, prediction_id: OID) -> List[str]:
        results: List[str] = []
        result_pipe = self.prediction_id_to_util[prediction_id].predictor_end
        while result_pipe.poll():
            result_pipe.recv()
        return results


prediction_manager = PredictionManager()
