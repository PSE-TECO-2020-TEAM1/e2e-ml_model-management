from typing import Dict, Optional
from bson.errors import InvalidId
from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from pydantic.fields import Field
from pydantic.main import BaseConfig
from app.models.workspace import Workspace
from enum import Enum

#TODO delete once understood xd

client = AsyncIOMotorClient("localhost", 27017)
db = client.get_database("mydatabase")
collection = db.get_collection("mycollection")

class OID(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            return ObjectId(str(v))
        except InvalidId:
            raise ValueError("Not a valid ObjectId")

class FruitEnum(str, Enum):
    pear = 'pear'
    banana = 'banana'

class CookingModel(BaseModel):
    fruit: FruitEnum


class MongoModel(BaseModel):
    class Config(BaseConfig):
        json_encoders = {
            ObjectId: lambda oid: str(oid),
        }

class User(MongoModel):
    _id: Optional[OID] = None


async def do_insert(a):
    result = await collection.insert_one(a)
    print('result %s' % repr(result.inserted_id))

async def do_find():
    result: User = await collection.find_one({"_id": ObjectId('601f10a6e6341e9be0496e69')})
    print(type(result))
    print(result["_id"])


import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(do_find())
