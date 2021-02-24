from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

class TrainingPool:

    def __init__(self):
        self.executor: ProcessPoolExecutor

    def start(self):
        self.executor = ProcessPoolExecutor(max_workers=cpu_count())

    def shutdown(self):
        self.executor.shutdown()

    def submit(self, callable, args):
        return self.executor.submit(callable, args)

    
training_pool = TrainingPool()