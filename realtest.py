import asyncio
from typing import Dict, Union

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
import random

from app.training.trainer import Trainer
from app.util.training_parameters import (Classifier, Feature, Imputation,
                                          Normalization)


def fillData(num):
    data_points = []
    for i in range(num):
        data_points.append({"values": [random.randint(1, 100) for i in range(3)]})
    return data_points


async def runtest():
    uri = 'mongodb://0.0.0.0/'
    client = AsyncIOMotorClient(uri, 27017)
    db = client.test

    workspace_id = ObjectId('666f6f2d6261722d71757578')
    user_id = ObjectId('666f6f2d6261722d71757578')

    DATA_POINTS = 10

    await client.drop_database("test")
    result = await db.samples.insert_many([{
        "label": "blue",
        "timeframes": [{
            "start": 0,
            "end": DATA_POINTS
        }],
        "sensor_data_points": {
            "accelerometer": fillData(DATA_POINTS),
            "gyroscope": fillData(DATA_POINTS)
        }
    } for i in range(10)
        # {
        #     "label": "red",
        #     "timeframes": [{
        #         "start": 0,
        #         "end": DATA_POINTS
        #     }],
        #     "sensor_data_points": {
        #         "accelerometer": fillData(DATA_POINTS),
        #         "gyroscope": fillData(DATA_POINTS)
        #     }
        # }
    ])
    await db.workspaces.insert_one(
        {
            "_id": workspace_id,
            "user_id": user_id,
            "ml_models": [],
            "progress": -1,
            "workspace_data": {
                "label_to_label_code": {"blue": 1, "red": 2},
                "label_code_to_label": {"1": "blue", "2": "red"},
                "sliding_windows": {},
                "samples": result.inserted_ids
            }
        }
    )


loop = asyncio.get_event_loop()
loop.run_until_complete(runtest())
