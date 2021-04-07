from app.ml.objects import normalization
from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from starlette.status import HTTP_200_OK
from app.models.schemas.parameters import ClassifierSelection, ParametersInResponse
from app.ml.objects.util import get_features, get_imputations, get_normalizations, get_classifiers
from app.ml.objects.classification.classifier_config_spaces.util import get_hyperparameters, get_conditions

router = APIRouter()

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
    #TODO do the training manager maybe


