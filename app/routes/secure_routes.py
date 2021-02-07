from app.models.mongo_model import OID
from fastapi import APIRouter, Header, HTTPException
from fastapi.param_functions import Depends
import jwt

import app.models.requests as requests

async def extract_userId(auth_header: str = Header(None)) -> str:
    try:
        return jwt.decode(auth_header)["userId"]
    except:
        raise HTTPException(401)

router = APIRouter(prefix="/api")

@router.post("/workspaces/createModelWorkspace")
async def create_model_workspace(req: requests.PostCreateWorkspaceReq, userId: OID = Depends(extract_userId)):
    #TODO sample route
    pass