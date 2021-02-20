from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


training_executor: ProcessPoolExecutor = None
prediction_executor: ProcessPoolExecutor = None


def create_executors():
    global training_executor, prediction_executor
    training_executor = ProcessPoolExecutor(max_workers=cpu_count())
    prediction_executor = ProcessPoolExecutor(max_workers=cpu_count())
