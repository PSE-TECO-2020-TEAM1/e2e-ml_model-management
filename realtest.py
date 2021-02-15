from app.util.training_parameters import Imputation, Feature, Normalizer, Classifier
from app.training.trainer import Trainer
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

uri = 'mongodb://0.0.0.0/'
client = AsyncIOMotorClient(uri, 27017)
db = client.test

our_id = ObjectId('666f6f2d6261722d71757578')

db.workspaces.drop()
db.workspaces.insert_one(
    {
        "_id": our_id,
        "user_id": 3000,
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

Trainer(db, our_id, "atalay", Imputation.MEAN_IMPUTATION, [Feature.MIN, Feature.MAX], Normalizer.NORMALIZER, Classifier.RANDOM_FOREST_CLASSIFIER , 1, 1, None).train()