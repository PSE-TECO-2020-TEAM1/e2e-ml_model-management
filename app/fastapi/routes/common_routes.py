from starlette.responses import Response
from app.models.domain.sensor import Sensor, make_sensor_component
from app.models.domain.workspace import Workspace
from typing import List
from app.models.schemas.prediction_data import PredictionData
from app.fastapi.services import add_workspace, get_ml_model, get_workspace
from app.models.domain.prediction_key import PredictionKey
from app.models.schemas.sensor import SensorInPredictionConfig, SensorInWorkspace
from app.models.schemas.workspace_config import WorkspaceConfig
from fastapi.param_functions import Depends
from app.fastapi.dependencies import extract_user_id, get_prediction_key_by_id_from_query
from app.models.schemas.mongo_model import OID
from fastapi import APIRouter, status
from app.models.schemas.parameters import ClassifierSelection, ParametersInResponse
from app.ml.objects.util import get_features, get_imputations, get_normalizations, get_classifiers
from app.ml.objects.classification.classifier_config_spaces.util import get_hyperparameters, get_conditions
from app.ml.prediction.prediction_manager import prediction_manager
from app.models.schemas.prediction_config import PredictionConfig

router = APIRouter()


@router.post("/createModelWorkspace", status_code=status.HTTP_201_CREATED)
async def create_model_workspace(workspace_config: WorkspaceConfig, user_id: OID = Depends(extract_user_id)):
    workspace = Workspace(_id=workspace_config.workspaceId,
                          user_id=user_id,
                          sensors={sensor.name: Sensor(
                              sampling_rate=sensor.samplingRate,
                              components=[make_sensor_component(sensor.name, component) for component in sensor.dataFormat]
                          ) for sensor in workspace_config.sensors})
    await add_workspace(workspace)
    return Response(status_code=status.HTTP_201_OK)


@router.get("/parameters", response_model=ParametersInResponse, status_code=status.HTTP_200_OK)
async def get_parameters():
    response = ParametersInResponse(
        features=get_features(),
        imputations=get_imputations(),
        normalizations=get_normalizations(),
        classifierSelections=[ClassifierSelection(
            classifier=classifier,
            hyperparameters=get_hyperparameters(classifier),
            conditions=get_conditions(classifier)
        ) for classifier in get_classifiers()]
    )
    return response


@router.get("/predictionConfig", response_model=PredictionConfig, status_code=status.HTTP_200_OK)
async def get_prediction_config(prediction_key: PredictionKey = Depends(get_prediction_key_by_id_from_query)):
    workspace = await get_workspace(prediction_key.workspace_id)
    parsed_sensors = [SensorInPredictionConfig(name=name, samplingRate=sensor.sampling_rate) for name, sensor in workspace.sensors.items()]
    return PredictionConfig(sensors=parsed_sensors)


@router.get("/startPrediction", status_code=status.HTTP_200_OK)
async def get_start_prediction(prediction_key: PredictionKey = Depends(get_prediction_key_by_id_from_query)):
    prediction_manager.spawn_predictor(prediction_key.workspace_id, prediction_key._id, prediction_key.model_id)
    return Response(status_code=status.HTTP_200_OK)


@router.post("/submitData", status_code=status.HTTP_200_OK)
async def post_submit_data(prediction_data: PredictionData):
    prediction_manager.submit_data(prediction_data)
    return Response(status_code=status.HTTP_200_OK)


@router.get("/predictionResults", response_model=List[str], status_code=status.HTTP_200_OK)
async def get_prediction_results(predictionId: OID):
    return prediction_manager.get_prediction_results(predictionId)
