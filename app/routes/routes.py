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

@router.get("/parameters", response_model = GetParameterRes)
async def parameters():
    return


@router.get("/predictionConfig", response_model = PredictionConfigRes)
async def getModel(predictionConfigreq: GetPredictionConfig):
    predictionConfig: PredictionConfigRes
    return predictionConfig

@router.post("/submitDataWindow", response_model = submitDataWindowRes)
async def getModel(datawindow: SubmitDataWindow):
    datawindow: submitDataWindowRes
    return datawindow