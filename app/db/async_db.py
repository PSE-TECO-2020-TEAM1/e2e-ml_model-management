from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import Settings, get_settings


class AsyncDb():
    def __init__(self):    
        self.settings: Settings = get_settings()
        self.client: AsyncIOMotorClient
        self.db: AsyncIOMotorDatabase

    def get(self):
        return self.db

    def connect_to_database(self):
        self.client = AsyncIOMotorClient(self.settings.DATABASE_URI, self.settings.DATABASE_PORT)
        self.db = self.client[self.settings.DATABASE_NAME]

    def disconnect_from_database(self):
        self.client.close()
        