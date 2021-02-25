from concurrent.futures import ProcessPoolExecutor

class TrainingPool:

    def __init__(self):
        self.executor: ProcessPoolExecutor

    def start(self):
        self.executor = ProcessPoolExecutor()

    def shutdown(self):
        self.executor.shutdown()

    def submit(self, callable, args):
        return self.executor.submit(callable, args)

    
training_pool = TrainingPool()