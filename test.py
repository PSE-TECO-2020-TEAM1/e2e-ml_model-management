from enum import Enum
from pickle import dumps
from typing import Dict, List, Optional

from bson import Binary
from bson.errors import InvalidId
from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from pydantic.fields import Field
from pydantic.main import BaseConfig

from app.models.workspace import Workspace

#TODO delete once understood xd

client = AsyncIOMotorClient("localhost", 27017)
db = client.get_database("mydatabase")
collection = db.get_collection("mycollection")

async def do_insert(a):
    result = await collection.insert_one(a)
    print('result %s' % repr(result.inserted_id))

async def do_find():
    result = await collection.find_one({"_id": ObjectId('601f10a6e6341e9be0496e69')})
    print(result["_id"])

from enum import Enum

class A(str, Enum):
    elem1 = "elem1"
    elem2 = "elem2"

class B(BaseModel):
    a_dict: Dict[A, float]

class C(BaseModel):
    b_dict: Dict[str, List[B]]

print(C(b_dict = {"1": [{"elem1": 0.97}]}))

import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete()
