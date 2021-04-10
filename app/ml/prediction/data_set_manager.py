from app.ml.objects.feature.enum import Feature
from typing import Dict, List
from app.models.domain.sensor import Sensor, SensorComponent
from app.models.domain.sliding_window import SlidingWindow
from app.db.syncdb.ml_model_repository import MlModelRepository
from app.db.syncdb.file_repository import FileRepository
from app.db.syncdb.workspace_repository import WorkspaceRepository
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
        # Cache the model here
        if not self.ml_model:
            self.ml_model = self.ml_model_repository.get_ml_model(self.workspace_id)
        return self.ml_model

    def get_workspace_sensors(self) -> Dict[str, Sensor]:
        return self.workspace_repository.get_workspace(self.workspace_id).sensors

    def get_sliding_window(self) -> SlidingWindow:
        ml_model = self.get_ml_model()
        return ml_model.config.sliding_window

    def get_component_features(self) -> Dict[SensorComponent, List[Feature]]:
        ml_model = self.get_ml_model()
        return ml_model.config.get_component_features()

    def get_column_order(self) -> List[str]:
        ml_model = self.get_ml_model()
        return ml_model.column_order

    def get_label_encoder(self) -> LabelEncoder:
        ml_model = self.get_ml_model()
        return self.file_repository.get_file(MlModel.deserialize_label_encoder(ml_model.label_encoder_object_file_ID))

    def get_pipeline(self) -> Pipeline:
        ml_model = self.get_ml_model()
        return self.file_repository.get_file(MlModel.deserialize_pipeline(ml_model.pipeline_object_file_ID))
