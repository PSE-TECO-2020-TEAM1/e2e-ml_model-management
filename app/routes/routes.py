from fastapi import APIRouter, status
from starlette.status import HTTP_200_OK

from app.util.training_parameters import Feature, Imputation, Normalization
import app.models.requests as request_models
import app.models.responses as response_models

from app.util.classifier_config_spaces import get_classifiers_with_hyperparameters

router = APIRouter()


@router.get("/parameters", response_model=response_models.GetParametersRes, status_code=status.HTTP_200_OK)
async def get_parameters():
    response = response_models.GetParametersRes(
        features=[f for f in Feature],
        imputers=[i for i in Imputation],
        normalizers=[n for n in Normalization],
        classifier_selections=get_classifiers_with_hyperparameters()
    )
    return response
        

@router.get("/predictionConfig", response_model=response_models.GetPredictionConfigRes, status_code=HTTP_200_OK)
async def get_prediction_config(req: request_models.GetPredictionConfigReq):
    # TODO
    pass


@router.post("/submitData", )
async def post_submit_data(req: request_models.SubmitDataReq):
    # TODO
    pass
