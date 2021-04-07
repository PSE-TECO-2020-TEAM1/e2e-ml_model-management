from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import DATABASE_NAME, DATABASE_URI, DATABASE_PORT

_client = AsyncIOMotorClient(DATABASE_URI, DATABASE_PORT)
_db = _client[DATABASE_NAME]

def get_db() -> AsyncIOMotorDatabase:
    return _db
