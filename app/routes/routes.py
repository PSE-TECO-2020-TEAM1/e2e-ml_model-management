from app.models.workspace import Workspace
from app.models.mongo_model import OID
from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from starlette.status import HTTP_200_OK

from app.db import db
from app.util.training_parameters import Feature, Imputation, Normalization
import app.models.requests as request_models
import app.models.responses as response_models
from app.prediction.prediction_manager import prediction_manager
from app.util.classifier_config_spaces import get_classifiers_with_hyperparameters

router = APIRouter()


@router.get("/parameters", response_model=response_models.GetParametersRes, status_code=status.HTTP_200_OK)
async def get_parameters():
    response = response_models.GetParametersRes(
        features=[f for f in Feature],
        imputers=[i for i in Imputation],
        normalizers=[n for n in Normalization],
        # TODO max values of window size and sliding step?
        windowSize=(4, 500),
        slidingStep=(1, 250),
        classifierSelections=get_classifiers_with_hyperparameters()
    )
    return response


@router.get("/predictionConfig", response_model=response_models.GetPredictionConfigRes, status_code=HTTP_200_OK)
async def get_prediction_config(predictionId: str):
    # TODO
    workspace_doc = await db.get().workspaces.find_one({"predictionIds." + predictionId: {"$exists": True}})
    if workspace_doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="The prediction id is not valid")
    workspace = Workspace(**workspace_doc)
    await prediction_manager.spawn_predictor(prediction_id=predictionId, model_id=workspace.predictionIds[predictionId])
    return response_models.GetPredictionConfigRes(sensors=workspace.sensors)


@router.post("/submitData", status_code=status.HTTP_200_OK)
async def post_submit_data(req: request_models.PostSubmitDataReq):
    # TODO here properly with checks etc...
    prediction_manager.submit_data(req.predictionId, req.dataPointCount, req.data)

@router.get("/predictionResults", response_model=response_models.GetPredictionResultsRes, status_code=HTTP_200_OK)
async def get_prediction_results(predictionId: str):
    # TODO here properly with checks etc...
    results = response_models.GetPredictionResultsRes(predictions=prediction_manager.get_prediction_results(predictionId))
    return results