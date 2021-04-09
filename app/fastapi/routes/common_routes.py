from app.models.schemas.workspace_config import WorkspaceConfig
from fastapi.param_functions import Depends
from app.fastapi.dependencies.auth import extract_user_id
from app.models.schemas.mongo_model import OID
from fastapi import APIRouter, status
from app.models.schemas.parameters import ClassifierSelection, ParametersInResponse
from app.ml.objects.util import get_features, get_imputations, get_normalizations, get_classifiers
from app.ml.objects.classification.classifier_config_spaces.util import get_hyperparameters, get_conditions

router = APIRouter()

@router.post("/createModelWorkspace", status_code=status.HTTP_201_CREATED)
async def create_model_workspace(workspace_config: WorkspaceConfig, user_id: OID = Depends(extract_user_id)):
    #
    return #TODO how to return empty and not null

@router.get("/parameters", response_model=ParametersInResponse, status_code=status.HTTP_200_OK)
async def get_parameters():
    response = ParametersInResponse(
        features=get_features(),
        imputations=get_imputations(),
        normalizations=get_normalizations(),
        classifier_selection=[ClassifierSelection(
            classifier=classifier,
            hyperparameters=get_hyperparameters(classifier),
            conditions=get_conditions(classifier)
        ) for classifier in get_classifiers()]
    )
    return response

