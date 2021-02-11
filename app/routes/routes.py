from fastapi import APIRouter 
from pydantic import BaseModel
from app.models.requests import (
    TrainReq,
    GetPredictionConfig,
    SubmitDataWindow
)
from app.models.responses import (
    GetParameterRes,
    GetModelsRes,
    TrainingProgress,
    ModelRes,
    PredictionIDRes,
    PredictionConfigRes,
    submitDataWindowRes
)

router = APIRouter(prefix="/api")

@router.get("/Parameters", response_model = GetParameterRes)
async def getPara():
    para: GetParameterRes
    return para

@router.get("/workspaces/{workspaceId}/models", response_model = GetModelsRes)
async def getModel():
    models: GetModelsRes
    return models

@router.post("/workspaces/{workspaceId}/train")
async def getModel(trainReq: TrainReq):
    return True

@router.get("/workspaces/{workspaceId}/trainingProgress", response_model = TrainingProgress)
async def getModel():
    trainingProgress: TrainingProgress
    return trainingProgress

@router.get("/workspaces/{workspaceId}/models/{modelId}", response_model = ModelRes)
async def getModel():
    model: ModelRes
    return model

@router.get("/workspaces/{workspaceId}/models/{modelId}/predictionId", response_model = PredictionIDRes)
async def getModel():
    predictionID: PredictionIDRes
    return predictionID

@router.get("/predictionConfig", response_model = PredictionConfigRes)
async def getModel(predictionConfigreq: GetPredictionConfig):
    predictionConfig: PredictionConfigRes
    return predictionConfig

@router.post("/submitDataWindow", response_model = submitDataWindowRes)
async def getModel(datawindow: SubmitDataWindow):
    datawindow: submitDataWindowRes
    return datawindow