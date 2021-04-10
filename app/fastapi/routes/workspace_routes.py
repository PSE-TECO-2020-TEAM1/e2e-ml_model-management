from app.models.schemas.prediction_config import PredictionIdInResponse
from dataclasses import asdict
from app.ml.training.training_state import TrainingState
from app.models.domain.ml_model import MlModel
from app.models.domain.workspace import Workspace
from app.fastapi.services import generate_prediction_id_for_model, get_ml_model
from app.fastapi.dependencies import get_ml_model_by_id_from_path, get_workspace_by_id_from_path, validate_user_of_workspace
from typing import List
from app.models.schemas.ml_model import MlModelInResponse, MlModelMetadataInResponse, PerformanceMetricsInResponse, SingleMetricInResponse
from app.ml.training.training_manager import initiate_new_training, validate_config
from app.models.schemas.mongo_model import OID
from app.models.schemas.training_config import TrainingConfigInTrain
from app.ml.training.config_parsing import parse_config_for_response
from starlette.responses import Response
from fastapi import APIRouter, status
from fastapi.param_functions import Depends

router = APIRouter(
    prefix="/workspaces",
    dependencies=[Depends(validate_user_of_workspace)]
)


@router.post("/{workspaceId}/train", status_code=status.HTTP_200_OK)
async def post_train(training_config: TrainingConfigInTrain, workspace: Workspace = Depends(get_workspace_by_id_from_path)):
    validate_config(training_config, workspace.sensors)
    initiate_new_training(workspace._id, training_config)
    return Response(status_code=status.HTTP_200_OK)


@router.get("/{workspaceId}/trainingState", response_model=TrainingState, status_code=status.HTTP_200_OK)
async def get_training_progress():
    # TODO
    pass


@router.get("/{workspaceId}/models", response_model=List[MlModelMetadataInResponse], status_code=status.HTTP_200_OK)
async def get_models(workspace: Workspace = Depends(get_workspace_by_id_from_path)):
    models: List[MlModel] = []
    for ml_model_ref in workspace.trained_ml_model_refs:
        models.append(await get_ml_model(ml_model_ref))
    return [MlModelMetadataInResponse(id=model._id, name=model.config.model_name) for model in models]


@router.get("/{workspaceId}/models/{modelId}", response_model=MlModelInResponse, status_code=status.HTTP_200_OK)
async def get_model(ml_model: MlModel = Depends(get_ml_model_by_id_from_path)):
    return MlModelInResponse(config=parse_config_for_response(ml_model.config),
                             labelPerformanceMetrics=[PerformanceMetricsInResponse(
                                 label=label_metric.label,
                                 metrics=[SingleMetricInResponse(
                                     name=metric.name,
                                     score=metric.score
                                 ) for metric in label_metric.metrics]
                             ) for label_metric in ml_model.label_performance_metrics])


@router.get("/{workspaceId}/models/{modelId}/generatePredictionId", response_model=PredictionIdInResponse, status_code=status.HTTP_200_OK)
async def get_prediction_id(workspaceId: OID, modelId: OID):
    prediction_id = await generate_prediction_id_for_model(workspaceId, modelId)
    return PredictionIdInResponse(predictionId=prediction_id)
