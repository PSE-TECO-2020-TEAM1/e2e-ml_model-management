from typing import Callable

from pymongo.database import Database
from app.models.domain.training_config import TrainingConfig
from app.ml.training.data_cache_manager import DataCacheManager

class Trainer():

    def __init__(self, config: TrainingConfig, cache_manager: DataCacheManager, db_conn: Callable[[], Database]):
        self.config = config
        self.cache_manager = cache_manager
        self.db_conn = db_conn

    def setup(self):
        db = self.db_conn()
        self.cache_manager.set_database(db)
        self.cache_manager.update_training_data_set()

    def train():
        
        pass