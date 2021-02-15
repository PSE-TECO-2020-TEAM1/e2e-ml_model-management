import app.models.requests as requests
import app.models.responses as responses
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


@router.get("/parameters", response_model=responses.GetParameterRes)
async def parameters():
    # TODO
    pass


@router.get("/predictionConfig", response_model=responses.PredictionConfigRes)
async def getModel(predictionConfigreq: requests.GetPredictionConfig):
    # TODO
    pass


@router.post("/submitDataWindow", response_model=responses.submitDataWindowRes)
async def getModel(datawindow: requests.SubmitDataWindow):
    # TODO
    pass
