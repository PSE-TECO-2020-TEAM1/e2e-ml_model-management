from app.fastapi.services import workspace_belongs_to_user
from app.models.schemas.mongo_model import OID
from fastapi.exceptions import HTTPException
from fastapi import Header
from fastapi.param_functions import Depends
import jwt
from starlette import status
from app.core.config import AUTH_SECRET
from bson.objectid import ObjectId

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