from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.config import WORKSPACE_COLLECTION_NAME
from app.models.domain.workspace import Workspace

class WorkspaceRepository():
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[WORKSPACE_COLLECTION_NAME]

    async def add_workspace(self, workspace: Workspace):
        await self.collection.insert_one(workspace)

    async def update_x(self, workspace_id: ObjectId, x):
        # TODO add when necessary
        pass

    async def delete_workspace(self, workspace_id: ObjectId):
        await self.collection.delete_one({"_id": workspace_id})