from app.models.domain.prediction_key import PredictionKey
from app.models.domain.ml_model import MlModel
from app.fastapi.services import get_ml_model, get_prediction_key, workspace_belongs_to_user
from app.models.schemas.mongo_model import OID
from fastapi.exceptions import HTTPException
from fastapi import Header
from fastapi.param_functions import Depends
import jwt
from starlette import status
from app.core.config import AUTH_SECRET
from bson.objectid import ObjectId
from app.fastapi.services import get_workspace
from app.models.schemas.mongo_model import OID
from app.models.domain.workspace import Workspace


async def extract_user_id(Authorization: str = Header(None)) -> ObjectId:
    try:
        decoded = jwt.decode(jwt=Authorization.split()[1], key=AUTH_SECRET, algorithms=["HS256"])
        if "exp" not in decoded:
            raise jwt.ExpiredSignatureError
        userId = ObjectId(decoded["userId"])
        return userId
    except:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


async def validate_user_of_workspace(workspaceId: OID, user_id=Depends(extract_user_id)):
    if not await workspace_belongs_to_user(workspaceId, user_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="The requested workspace does not belong to the user")


async def get_workspace_by_id_from_path(workspaceId: OID) -> Workspace:
    return await get_workspace(workspaceId)


async def get_ml_model_by_id_from_path(workspaceId: OID, modelId: OID) -> MlModel:
    workspace = await get_workspace(workspaceId)
    if modelId not in workspace.trained_ml_model_refs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="The request model does not exist in the workspace")
    return await get_ml_model(modelId)


async def get_prediction_key_by_id_from_query(predictionId: OID) -> PredictionKey:
    return await get_prediction_key(predictionId)
