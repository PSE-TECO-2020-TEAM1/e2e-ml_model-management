from multiprocessing import Process, Semaphore, Pipe
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

from typing import Dict, List

from app.prediction.predictor import Predictor
from app.db import db
from app.models.workspace import DataPoint
from app.models.mongo_model import OID
from app.config import get_settings


# TODO killing child process after timeout (join?)

class PredictionUtil():
    def __init__(self, process: Process, semaphore: Semaphore, shm: SharedMemory, read_end_of_pipe: Connection):
        self.process = process
        self.semaphore = semaphore
        self.shm = shm
        self.read_end_of_pipe = read_end_of_pipe


class PredictionManager():
    def __init__(self):
        self.prediction_id_to_util: Dict[str, PredictionUtil] = {}

    async def spawn_predictor(self, prediction_id: OID, model_id: OID, label_code_to_label: Dict[int, str]):
        model_doc = await db.get().ml_models.find_one({"_id": model_id})
        predictor = Predictor(window_size=model_doc["windowSize"], sliding_step=model_doc["slidingStep"], imputation_id=model_doc["imputer_object"],
                              normalizer_id=model_doc["normalizer_object"], classifier_id=model_doc["classifier_object"], features=model_doc["features"], label_code_to_label=label_code_to_label)
        
        semaphore = Semaphore(0)
        (read_end_of_pipe, write_end_of_pipe) = Pipe(duplex=False)
        shm = SharedMemory("shm_" + prediction_id, create=True, size=get_settings().SIZE_OF_QUEUE_IN_DATA_WINDOWS)

        process = Process(target=predictor.for_process, args=(semaphore), write_end_of_pipe=write_end_of_pipe) # TODO shm
        process.start()
        if not process.is_alive():
            raise OSError("A process can't be spawned for the prediction.")
        self.prediction_id_to_util[prediction_id] = PredictionUtil(process=process, semaphore=semaphore, read_end_of_pipe=read_end_of_pipe) # TODO shm
        write_end_of_pipe.close()
        

    def submit_data(self, prediction_id: OID, data_window: Dict[str, List[DataPoint]]):
        pass

    def get_predictions(self, prediction_id: OID) -> List[str]:
        pass
