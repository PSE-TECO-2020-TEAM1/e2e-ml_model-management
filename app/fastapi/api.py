from app.fastapi.routes import common_routes, secure_routes
from fastapi import APIRouter 

router: APIRouter = APIRouter(prefix="/api")

router.include_router(common_routes.router)
router.include_router(secure_routes.router)

