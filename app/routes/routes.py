from fastapi import APIRouter 
from pydantic import BaseModel

import app.models.requests as requests
import app.models.responses as responses

router = APIRouter()

@router.get("/parameters", response_model = responses.GetParameterRes)
async def parameters():
    #TODO
    pass

@router.get("/predictionConfig", response_model = responses.PredictionConfigRes)
async def getModel(predictionConfigreq: GetPredictionConfig):
    #TODO
    pass

@router.post("/submitDataWindow", response_model = responses.submitDataWindowRes)
async def getModel(datawindow: SubmitDataWindow):
    #TODO
    pass