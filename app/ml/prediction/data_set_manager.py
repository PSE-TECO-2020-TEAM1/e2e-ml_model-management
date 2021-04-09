from typing import Dict
from app.models.domain.sensor import Sensor
from app.models.domain.sliding_window import SlidingWindow
from app.db.sync.ml_model_repository import MlModelRepository
from app.db.sync.file_repository import FileRepository
from app.db.sync.workspace_repository import WorkspaceRepository
from app.models.domain.ml_model import MlModel
from bson.objectid import ObjectId
from pymongo.database import Database
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

class DataSetManager():
    def __init__(self, workspace_id: ObjectId):
        self.workspace_id = workspace_id
        self.ml_model: MlModel = None

    def set_db(self, db: Database):
        self.workspace_repository = WorkspaceRepository(db)
        self.file_repository = FileRepository(db)
        self.ml_model_repository = MlModelRepository(db)

    def get_ml_model(self) -> MlModel:
        # We can cache since MlModel is frozen
        if not self.ml_model:
            self.ml_model = self.ml_model_repository.get_ml_model(self.workspace_id)
        return self.ml_model

    def get_workspace_sensors(self) -> Dict[str, Sensor]:
        return self.workspace_repository.get_workspace(self.workspace_id).sensors

    def get_sliding_window(self) -> SlidingWindow:
        ml_model = self.get_ml_model()
        return ml_model.config.feature_extraction_config.sliding_window

    def get_label_encoder(self) -> LabelEncoder:
        ml_model = self.get_ml_model()
        return self.file_repository.get_file(MlModel.deserialize_label_encoder(ml_model.label_encoder_object_file_ID))

    def get_pipeline(self) -> Pipeline:
        ml_model = self.get_ml_model()
        return self.file_repository.get_file(MlModel.deserialize_pipeline(ml_model.pipeline_object_file_ID))