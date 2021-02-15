from app.util.training_parameters import Imputation, Feature, Normalizer, Classifier
from app.training.trainer import Trainer
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

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
            "workspace_data": {
                "label_dict": {"blue": 1, "red": 2},
                "sliding_windows": {},
                "samples": [
                    {
                        "label": "blue",
                        "sensor_data_points": {
                            "accelerometer": [
                                {
                                    "values": [1, 2, 3]
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
                                }
                            ]
                        }
                    },
                    {
                        "label": "red",
                        "sensor_data_points": {
                            "accelerometer": [
                                {
                                    "values": [90, 60, 90] 
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
                                }
                            ]
                        }
                    }
                ]
            }
        }
    )
    trainer1 = Trainer(db, workspace_id, "atalay", 1, 1, [Feature.VARIANCE], Imputation.MEAN_IMPUTATION, Normalizer.NORMALIZER, Classifier.RANDOM_FOREST_CLASSIFIER, None)
    # trainer2 = Trainer(db, workspace_id, "atalay", Imputation.MEAN_IMPUTATION, [Feature.MAXIMUM, Feature.VARIANCE], Normalizer.NORMALIZER, Classifier.RANDOM_FOREST_CLASSIFIER , 1, 1, None)
    await trainer1.train()
    # await trainer2.train()

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(runtest())