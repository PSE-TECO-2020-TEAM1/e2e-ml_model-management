from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

class ExecutorManager:

    def __init__(self):
        self.executor: ProcessPoolExecutor

    def start(self):
        self.executor = ProcessPoolExecutor(max_workers=cpu_count())

    def shutdown(self):
        self.executor.shutdown()

    def submit(self, callable):
        return self.executor.submit(callable)

    
training_executor = ExecutorManager()
prediction_executor = ExecutorManager()