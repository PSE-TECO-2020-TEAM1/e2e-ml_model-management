from app.db.asyncdb.workspace_repository import WorkspaceRepository
from app.models.domain.prediction_id import PredictionID
from app.models.domain.ml_model import MlModel
from bson.objectid import ObjectId
from app.models.domain.workspace import Workspace
from app.db.asyncdb.ml_model_repository import MlModelRepository
from app.db.asyncdb import get_async_db

async def workspace_belongs_to_user(workspace_id: ObjectId, user_id: ObjectId) -> bool:
    workspace = await get_workspace(workspace_id)
    return workspace.user_id == user_id

async def get_workspace(workspace_id: ObjectId) -> Workspace:
    workspace = await WorkspaceRepository(get_async_db()).get_workspace(workspace_id)
    return workspace

async def get_ml_model(ml_model_id: ObjectId) -> MlModel:
    model = await MlModelRepository(get_async_db()).get_ml_model(ml_model_id)
    return model

async def add_prediction_id_to_model(ml_model_id: ObjectId, prediction_id: PredictionID):
    await MlModelRepository(get_async_db()).add_prediction_ID(ml_model_id, prediction_id)
