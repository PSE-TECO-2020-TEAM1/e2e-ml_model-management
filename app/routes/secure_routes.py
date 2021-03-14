from multiprocessing.context import Process
import jwt
import json
import uuid
from starlette.responses import Response
from typing import Dict, List
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.param_functions import Depends
from httpx import AsyncClient
from bson import ObjectId

from app.models.ml_model import MlModel
import app.models.requests as request_models
import app.models.responses as response_models
from app.config import get_settings
from app.training.trainer import Trainer
from app.db import db
from app.models.mongo_model import OID
from app.models.workspace import Workspace, SampleInJson, WorkspaceData

async def extract_userId(Authorization: str = Header(None)) -> ObjectId:
    decoded = jwt.decode(jwt=Authorization.split()[1], key=get_settings().secret_key, algorithms=["HS256"])
    try:
        print(get_settings().secret_key)
        decoded = jwt.decode(jwt=Authorization.split()[1], key=get_settings().secret_key, algorithms=["HS256"])
        print(decoded)
        if "exp" not in decoded:
            raise jwt.ExpiredSignatureError
        return ObjectId(decoded["userId"])
    except:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

router = APIRouter()


@router.post("/workspaces/createModelWorkspace", status_code=status.HTTP_201_CREATED)
async def create_model_workspace(req: request_models.PostCreateWorkspaceReq, user_id=Depends(extract_userId)):
    workspace = Workspace(_id=req.workspaceId, user_id=user_id, sensors=req.sensors)
    await db.get().workspaces.insert_one(workspace.dict(by_alias=True))
    return #TODO how to return empty and not null?

@router.post("/workspaces/{workspaceId}/train", status_code=status.HTTP_200_OK)
async def post_train(workspaceId: OID, req: request_models.PostTrainReq, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)

    if await db.get().workspaces.count_documents({"_id": workspaceId, "userId": userId}, limit=1) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")

    if await db.get().ml_models.count_documents({"workspaceId": workspaceId, "name": req.modelName}, limit=1) == 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="A trained model with the given name already exists for the user")

    if await db.get().workspaces.update_one({"_id": workspaceId, "progress": -1}, {"$set": {"progress": 0}}) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="A training for this workspace is already in progress")

    # __fetch_workspace_data(workspaceId)

    trainer = Trainer(workspaceId, req.modelName, req.windowSize, req.slidingStep,
                      req.features, req.imputation, req.normalizer, req.classifier, req.hyperparameters)

    import time
    start = time.time()
    process = Process(target=trainer.train)
    process.start()
    #process.join()
    #print(time.time() - start)
    return Response(status_code=status.HTTP_200_OK)



@router.get("/workspaces/{workspaceId}/trainingProgress", response_model=response_models.GetTrainingProgressRes, status_code=status.HTTP_200_OK)
async def get_training_progress(workspaceId: OID, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)

    workspace_doc = await db.get().workspaces.find_one({"_id": workspaceId}, {"_id": False, "progress": True})
    if workspace_doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")
    workspace = Workspace(**workspace_doc)
    if workspace.progress == -1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="There is no training in progress for this workspace.")

    return response_models.GetTrainingProgressRes(progress = workspace.progress)


@router.get("/workspaces/{workspaceId}/models", response_model=response_models.GetModelsRes, status_code=status.HTTP_200_OK)
async def get_models(workspaceId: OID, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)

    workspace_doc = await db.get().workspaces.find_one(
        {"_id": workspaceId, "userId": userId}, {"_id": False, "mlModels": True})
    if workspace_doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")
    models = [MlModel(**model) async for model in db.get().ml_models.find({"_id": {"$in": workspace_doc["mlModels"]}})]
    return response_models.GetModelsRes(models=[model.dict(include={"id", "name"}) for model in models])

        
@router.get("/workspaces/{workspaceId}/models/{modelId}", response_model=response_models.GetModelRes, status_code=status.HTTP_200_OK)
async def get_model(workspaceId: OID, modelId: OID, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)
    modelId = ObjectId(modelId)

    if await db.get().workspaces.count_documents({"_id": workspaceId, "userId": userId}, limit=1) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")

    model_doc = await db.get().ml_models.find_one({"_id": modelId, "workspaceId": workspaceId})
    if model_doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No model matched for the user")
    # Response model masks the unwanted fields
    return MlModel(**model_doc).dict()


@router.get("/workspaces/{workspaceId}/models/{modelId}/generatePredictionId", response_model=response_models.GetPredictionIdRes, status_code=status.HTTP_200_OK)
async def get_prediction_id(workspaceId: OID, modelId: OID, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)
    modelId = ObjectId(modelId)

    workspace_doc = await db.get().workspaces.find_one({"_id": workspaceId, "userId": userId}, {"_id": False, "mlModels": True})
    if workspace_doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")

    if modelId not in workspace_doc["mlModels"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No model matched for the user")

    new_prediction_id = uuid.uuid4().hex

    await db.get().workspaces.update_one({"_id": workspaceId}, {"$set": {"predictionIds." + new_prediction_id: modelId}})
    return response_models.GetPredictionIdRes(predictionId=new_prediction_id)

