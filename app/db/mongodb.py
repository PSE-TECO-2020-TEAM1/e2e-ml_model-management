from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

client: AsyncIOMotorClient
db: AsyncIOMotorDatabase


async def connect_to_database(client_uri: str, db_name: str):
    client = AsyncIOMotorClient(client_uri)
    db = client[db_name]


async def disconnect_from_database():
    client.close()
