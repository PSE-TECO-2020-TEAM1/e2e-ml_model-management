from app.ml.training.training_state import TrainingState
from app.db.asyncdb.workspace_repository import WorkspaceRepository
from app.models.domain.prediction_key import PredictionKey
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

async def add_workspace(workspace: Workspace):
    await WorkspaceRepository(get_async_db()).add_workspace(workspace)

async def get_ml_model(ml_model_id: ObjectId) -> MlModel:
    model = await MlModelRepository(get_async_db()).get_ml_model(ml_model_id)
    return model

async def delete_ml_model(ml_model_id: ObjectId):
    await MlModelRepository(get_async_db()).delete_ml_model(ml_model_id)

async def get_prediction_key(prediction_id: ObjectId) -> PredictionKey:
    key = await MlModelRepository(get_async_db()).get_prediction_key(prediction_id)
    return key

async def generate_prediction_id_for_model(workspace_id: ObjectId, ml_model_id: ObjectId):
    key = PredictionKey(None, workspace_id=workspace_id, model_id=ml_model_id)
    return await MlModelRepository(get_async_db()).add_prediction_key(key)

async def set_training_state_to_initiated(workspace_id: ObjectId):
    await WorkspaceRepository(get_async_db()).set_training_state(workspace_id, TrainingState.TRAINING_INITIATED)
