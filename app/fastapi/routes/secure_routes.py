from app.ml.training.training_manager import initiate_new_training
from app.models.schemas.mongo_model import OID
from app.models.schemas.training_config import TrainingConfigInTrain
from app.fastapi.dependencies.auth import extract_userId
from starlette.responses import Response
from fastapi import APIRouter, status
from fastapi.param_functions import Depends

router = APIRouter()

@router.post("/workspaces/{workspaceId}/train", status_code=status.HTTP_200_OK)
async def post_train(workspaceId: OID, training_config: TrainingConfigInTrain, userId=Depends(extract_userId)):
    initiate_new_training(workspaceId, training_config)
    return Response(status_code=status.HTTP_200_OK)

