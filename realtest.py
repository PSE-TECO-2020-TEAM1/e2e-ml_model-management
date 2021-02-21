from typing import Dict, Union

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from app.training.trainer import Trainer
from app.util.training_parameters import (Classifier, Feature, Imputation,
                                          Normalization)


async def runtest():
    uri = 'mongodb://0.0.0.0/'
    client = AsyncIOMotorClient(uri, 27017)
    db = client.test

    workspace_id = ObjectId('666f6f2d6261722d71757578')
    user_id = ObjectId('666f6f2d6261722d71757578')

    await client.drop_database("test")
    await db.workspaces.insert_one(
        {
            "_id": workspace_id,
            "user_id": user_id,
            "ml_models": [],
            "workspace_data": {
                "label_to_label_code": {"blue": 1, "red": 2},
                "label_code_to_label": {"1": "blue", "2": "red"},
                "sliding_windows": {},
                "samples": [
                    {
                        "label": "blue",
                        "timeframes": [{
                            "start": 0,
                            "end": 1
                        }],
                        "sensor_data_points": {
                            "accelerometer": [
                                {
                                    "values": [1, 2, 3]
                                    
                                },
                                {
                                    "values": [31, 42, 53]
                                },
                                {
                                    "values": [31, 42, 53]
                                }
                            ],
                            "gyroscope": [
                                {
                                    "values": [90, 60, 90]      
                                },
                                {
                                    "values": [13, 24, 35]
                                },
                                {
                                    "values": [13, 24, 35]
                                }
                            ]
                        }
                    },
                    {
                        "label": "red",
                        "timeframes": [{
                            "start": 0,
                            "end":2
                        }],
                        "sensor_data_points": {
                            "accelerometer": [
                                {
                                    "values": [90, 60, 90] 
                                },
                                {
                                    "values": [31, 42, 53]
                                },
                                {
                                    "values": [31, 42, 53]
                                }
                            ],
                            "gyroscope": [
                                {
                                    "values": [1, 2, 3] 
                                },
                                {
                                    "values": [13, 24, 35]
                                },
                                {
                                    "values": [13, 24, 35]
                                }
                            ]
                        }
                    }
                ]
            }
        }
    )

import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete(runtest())
