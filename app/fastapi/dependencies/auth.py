from fastapi.exceptions import HTTPException
from fastapi import Header
import jwt
from starlette import status
from app.core.config import AUTH_SECRET
from bson.objectid import ObjectId

async def extract_userId(Authorization: str = Header(None)) -> ObjectId:
    try:
        decoded = jwt.decode(jwt=Authorization.split()[1], key=AUTH_SECRET, algorithms=["HS256"])
        if "exp" not in decoded:
            raise jwt.ExpiredSignatureError
        userId = ObjectId(decoded["userId"])
        return userId
    except:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")