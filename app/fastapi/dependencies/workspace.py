from app.fastapi.services import get_workspace
from app.models.schemas.mongo_model import OID
from app.models.domain.workspace import Workspace

async def get_workspace_by_id_from_path(workspaceId: OID) -> Workspace:
    return await get_workspace(workspaceId)