from fastapi import APIRouter

from app.util.training_parameters import Feature, Imputation, Normalization
import app.models.requests as requests
import app.models.responses as responses

router = APIRouter()


@router.get("/parameters", response_model=responses.GetParametersRes)
async def getParameters():
    response = responses.GetParametersRes(
        features=[f for f in Feature],
        imputers=[i for i in Imputation],
        normalizers=[n for n in Normalization],
        #TODO here xd
        classifier_selections=[]
    ) 
        

@router.get("/predictionConfig", response_model=responses.PredictionConfigRes)
async def getModel(predictionConfigreq: requests.GetPredictionConfig):
    # TODO
    pass


@router.post("/submitDataWindow", response_model=responses.submitDataWindowRes)
async def getModel(datawindow: requests.SubmitDataWindow):
    # TODO
    pass
