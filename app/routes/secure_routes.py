import app.models.requests as requests
import app.models.responses as responses
import jwt
from app.db.mongodb import db
from app.models.mongo_model import OID
from app.models.workspace import Workspace
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.param_functions import Depends


async def extract_userId(auth_header: str = Header(None)) -> str:
    try:
        return jwt.decode(auth_header)["userId"]
    except:
        raise HTTPException(401)

router = APIRouter()


@router.post("/workspaces/createModelWorkspace", status_code=status.HTTP_201_CREATED)
async def create_model_workspace(req: requests.PostCreateWorkspaceReq, user_id: OID = Depends(extract_userId)):
    workspace: Workspace = Workspace(_id=req.workspace_id, user_id=user_id, sensors=req.sensors)
    await db.workspaces.insert_one(workspace.dict())


@router.get("/workspaces/{workspaceId}/models", response_model=responses.GetModelsRes, status_code=status.HTTP_200_OK)
async def get_models(workspaceId: OID, userId: OID = Depends(extract_userId)):
    # TODO
    pass


@router.post("/workspaces/{workspaceId}/train", status_code=status.HTTP_200_OK)
async def post_train(req: requests.PostTrainReq, userId: OID = Depends(extract_userId)):
    # TODO

    pass


@router.get("/workspaces/{workspaceId}/trainingProgress", response_model=responses.GetTrainingProgressRes, status_code=status.HTTP_200_OK)
async def get_training_progress(workspaceId: OID, userId: OID = Depends(extract_userId)):
    # TODO
    pass


@router.get("/workspaces/{workspaceId}/models/{modelId}", response_model=responses.GetModelRes, status_code=status.HTTP_200_OK)
async def get_model(workspaceId: OID, modelId: OID, userId: OID = Depends(extract_userId)):
    # TODO
    pass


@router.get("/workspaces/{workspaceId}/models/{modelId}/predictionId", response_model=responses.GetPredictionIdRes, status_code=status.HTTP_200_OK)
async def get_prediction_id(workspaceId: OID, modelId: OID, userId: OID = Depends(extract_userId)):
    # TODO
    pass
