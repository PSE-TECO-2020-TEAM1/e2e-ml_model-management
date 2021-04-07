from app.core.config import DATABASE_NAME, DATABASE_URI, DATABASE_PORT
from pymongo import MongoClient
from pymongo.database import Database

def create_sync_db() -> Database:
    client = MongoClient(DATABASE_URI, DATABASE_PORT)
    return client[DATABASE_NAME]