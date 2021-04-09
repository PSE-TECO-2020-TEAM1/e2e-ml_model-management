from app.core.config import DATABASE_NAME, DATABASE_URI, DATABASE_PORT
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_async_db = None

def initialize_async_db_connection():
    global _async_db
    _async_client = AsyncIOMotorClient(DATABASE_URI, DATABASE_PORT)
    _async_db = _async_client[DATABASE_NAME]


def get_async_db() -> AsyncIOMotorDatabase:
    if _async_db is None:
        raise RuntimeError("Database connection is not initialized")
    return _async_db
