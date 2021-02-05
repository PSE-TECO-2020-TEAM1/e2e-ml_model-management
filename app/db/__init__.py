from app.db.mongodb import MongoManager

db = MongoManager()

async def get_database() -> MongoManager:
    return db