from app.models.domain.ml_model import MlModel
from bson.objectid import ObjectId
from app.models.domain.workspace import Workspace
from app.fastapi.services import add_prediction_id_to_model, get_ml_model
from app.fastapi.dependencies.workspace import get_workspace_by_id_from_path
from typing import List
from app.models.schemas.ml_model import MlModelMetadataInResponse
from app.ml.training.training_manager import initiate_new_training
from app.models.schemas.mongo_model import OID
from app.models.schemas.training_config import TrainingConfigInTrain
from app.fastapi.dependencies.auth import extract_user_id, validate_user_of_workspace
from starlette.responses import Response
from fastapi import APIRouter, status
from fastapi.param_functions import Depends
from uuid import uuid4

router = APIRouter(
    prefix="/workspaces",
    dependencies=[Depends(validate_user_of_workspace)]
)

@router.post("/{workspaceId}/train", status_code=status.HTTP_200_OK)
async def post_train(workspaceId: OID, training_config: TrainingConfigInTrain, userId: ObjectId = Depends(extract_user_id)):
    initiate_new_training(workspaceId, training_config)
    return Response(status_code=status.HTTP_200_OK)

@router.get("/{workspaceId}/models", response_model=List[MlModelMetadataInResponse], status_code=status.HTTP_200_OK)
async def get_models(workspace: Workspace = Depends(get_workspace_by_id_from_path)):
    models: List[MlModel] = []
    for ml_model_ref in workspace.trained_ml_model_refs:
        models.append(get_ml_model(ml_model_ref))
    return [MlModelMetadataInResponse(id=model._id, name=model.config.model_name) for model in models]

@router.get("/{workspaceId}/models/{modelId}/generatePredictionId", response_model=str, status_code=status.HTTP_200_OK)
async def get_prediction_id(workspaceId: OID, modelId: OID):
    new_prediction_id = uuid4().hex
    add_prediction_id_to_model(workspaceId, modelId)
    return new_prediction_id
