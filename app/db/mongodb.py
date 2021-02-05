from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

class MongoManager():
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None

    async def connect_to_database(self, host: str, port: int):
        self.client = AsyncIOMotorClient(host, port)
        self.db = self.client.main_db

    async def close_database_connection(self):
        self.client.close()
