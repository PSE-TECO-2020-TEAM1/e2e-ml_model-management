from app.fastapi.routes import common_routes, workspace_routes
from fastapi import APIRouter 

router: APIRouter = APIRouter(prefix="/api")

router.include_router(common_routes.router)
router.include_router(workspace_routes.router)
