from app.models.mongo_model import OID
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.param_functions import Depends
import jwt

import app.models.requests as requests
import app.models.responses as responses

async def extract_userId(auth_header: str = Header(None)) -> str:
    try:
        return jwt.decode(auth_header)["userId"]
    except:
        raise HTTPException(401)

router = APIRouter(prefix="/api")

@router.post("/workspaces/createModelWorkspace", status_code=status.HTTP_201_CREATED)
async def create_model_workspace(req: requests.PostCreateWorkspaceReq, userId: OID = Depends(extract_userId)):
    #TODO sample route
    pass

@router.get("/workspaces/{workspaceId}/models", response_model = responses.GetModelsRes, status_code=status.HTTP_200_OK)
async def get_models(workspaceId: OID, userId: OID = Depends(extract_userId)):
    #TODO
    pass

@router.post("/workspaces/{workspaceId}/train", status_code=status.HTTP_200_OK)
async def post_train(req: requests.postTrainReq, userId: OID = Depends(extract_userId)):
    #TODO
    pass

@router.get("/workspaces/{workspaceId}/trainingProgress", response_model = responses.GetTrainingProgressRes, status_code=status.HTTP_200_OK)
async def get_training_progress(workspaceId: OID, userId: OID = Depends(extract_userId)):
    #TODO
    pass

@router.get("/workspaces/{workspaceId}/models/{modelId}", response_model = responses.GetModelRes, status_code=status.HTTP_200_OK)
async def get_model(workspaceId: OID, modelId: OID, userId: OID = Depends(extract_userId)):
    #TODO
    pass

@router.get("/workspaces/{workspaceId}/models/{modelId}/predictionId", response_model = responses.GetPredictionIdRes, status_code=status.HTTP_200_OK)
async def get_prediction_id(workspaceId: OID, modelId: OID, userId: OID = Depends(extract_userId)):
    #TODO
    pass