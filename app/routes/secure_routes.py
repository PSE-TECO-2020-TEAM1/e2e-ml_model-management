from app.training.trainer import Trainer
from typing import Dict, List, Tuple
import jwt
import json
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.param_functions import Depends
from httpx import AsyncClient
from bson import ObjectId

import app.models.requests as request_models
import app.models.responses as response_models
from app.models.ml_model import ML_Model
from app.db import db
from app.process_pool import training_executor, prediction_executor
from app.models.mongo_model import OID
from app.models.workspace import Sample, Workspace, WorkspaceData

async def extract_userId(Authorization: str = Header(None)) -> ObjectId:
    try:
        # TODO ENV key
        decoded = jwt.decode(jwt=Authorization.split()[1], key="sabahci_kahvesi", algorithms=["HS256"])
        if "exp" not in decoded:
            raise jwt.ExpiredSignatureError
        return ObjectId(decoded["userId"])
    except:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

router = APIRouter()


@router.post("/workspaces/createModelWorkspace", status_code=status.HTTP_201_CREATED)
async def create_model_workspace(req: request_models.PostCreateWorkspaceReq, user_id=Depends(extract_userId)):
    workspace = Workspace(id=req.workspace_id, user_id=user_id, sensors=req.sensors)
    workspace_doc = workspace.dict(exclude={"id"})
    workspace_doc["_id"] = workspace.id
    await db.get().workspaces.insert_one(workspace_doc)


@router.post("/workspaces/{workspaceId}/train", status_code=status.HTTP_200_OK)
async def post_train(workspaceId: OID, req: request_models.PostTrainReq, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)

    if await db.get().workspaces.count_documents({"_id": workspaceId, "user_id": userId}, limit=1) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")

    if await db.get().ml_models.count_documents({"workspace_id": workspaceId, "name": req.modelName}, limit=1) == 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="A trained model with the given name already exists for the user")

    if await db.get().workspaces.find_one_and_update({"_id": workspaceId, "progress": -1}, {"$set": {"progress": 0}}) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="A training for this workspace is already in progress")

    # TODO change update workspace
    # __update_workspace_data(workspaceId)

    trainer = Trainer(workspaceId, req.modelName, req.windowSize, req.slidingStep,
                      req.features, req.imputation, req.normalizer, req.classifier, req.hyperparameters)

    import time
    start = time.time()
    # future = training_executor.submit(trainer.train)
    # future.add_done_callback(lambda x: print(time.time() - start))
    trainer.train()
    print(time.time() - start)
    return


async def __update_workspace_data(workspace: Workspace):
    async with AsyncClient() as http_client:
        # TODO return what from this api endpoint murat? change to string from json
        last_modified: int = await http_client.get(url="/api/workspaces/" + workspace.id + "/samples", params={"onlyDate": True})
        if last_modified == workspace.workspace_data.last_modified:
            return

        labels: List[str] = await http_client.get(url="")  # TODO complete api endpoint
        label_to_label_code: Dict[str, int] = {labels[i]: i+1 for i in range(len(labels))}
        label_to_label_code["Other"] = 0
        label_code_to_label: Dict[int, str] = {i+1: labels[i] for i in range(len(labels))}
        label_code_to_label[0] = "Other"
        # TODO complete api endpoint
        samples_response = json.loads(await http_client.get(url="").json())
        samples: List[Sample] = [Sample.parse_raw(sample) for sample in samples_response]
        new_workspace_data = WorkspaceData(
            last_modified=last_modified, label_to_label_code=label_to_label_code, label_code_to_label=label_code_to_label, samples=samples)
        await db.get().workspaces.update_one({"_id": workspace.id}, {
            "$set": {"workspace_data": new_workspace_data.dict()}})


@router.get("/workspaces/{workspaceId}/trainingProgress", response_model=response_models.GetTrainingProgressRes, status_code=status.HTTP_200_OK)
async def get_training_progress(workspaceId: OID, userId=Depends(extract_userId)):
    # TODO
    workspaceId = ObjectId(workspaceId)

    workspace_doc = await db.get().workspaces.find_one({"_id": workspaceId}, {"_id": False, "progress": True})
    if workspace_doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")

    if workspace_doc["progress"] == -1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="There is no training in progress for this workspace.")

    return response_models.GetTrainingProgressRes(progress = workspace_doc["progress"])


@router.get("/workspaces/{workspaceId}/models", response_model=response_models.GetModelsRes, status_code=status.HTTP_200_OK)
async def get_models(workspaceId: OID, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)
    models: List[Tuple[OID, str]] = []

    workspace_doc = await db.get().workspaces.find_one(
        {"_id": workspaceId, "user_id": userId}, {"_id": False, "ml_models": True})
    if workspace_doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")

    model_ids_of_workspace = workspace_doc["ml_models"]

    for model_id in model_ids_of_workspace:
        model_name = (await db.get().ml_models.find_one({"_id": model_id}, {"_id": False, "name": True}))["name"]
        models.append((model_id, model_name))
    return response_models.GetModelsRes(models=models)


@router.get("/workspaces/{workspaceId}/models/{modelId}", response_model=response_models.GetModelRes, status_code=status.HTTP_200_OK)
async def get_model(workspaceId: OID, modelId: OID, userId=Depends(extract_userId)):
    workspaceId = ObjectId(workspaceId)
    modelId = ObjectId(modelId)

    if await db.get().workspaces.count_documents({"_id": workspaceId, "user_id": userId}, limit=1) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No workspace matched for the user")

    model_document = db.get().ml_models.find_one({"_id": modelId, "workspace_id": workspaceId})
    if model_document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No model matched for the user")

    return response_models.GetModelRes(model=ML_Model(**model_document))


@router.get("/workspaces/{workspaceId}/models/{modelId}/predictionId", response_model=response_models.GetPredictionIdRes, status_code=status.HTTP_200_OK)
async def get_prediction_id(workspaceId: OID, modelId: OID, userId=Depends(extract_userId)):
    # TODO
    workspaceId = ObjectId(workspaceId)
    modelId = ObjectId(modelId)
    pass