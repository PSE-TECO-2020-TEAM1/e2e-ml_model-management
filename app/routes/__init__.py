
from typing import Dict

from bson import ObjectId
from app.routes.routes import router as common_router
from app.routes.secure_routes import router as secure_router
from fastapi import APIRouter 

router: APIRouter = APIRouter(prefix="/api")

router.include_router(common_router)
router.include_router(secure_router)

