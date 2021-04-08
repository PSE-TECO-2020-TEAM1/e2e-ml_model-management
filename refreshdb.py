from dataclasses import asdict
from app.models.domain.training_state import TrainingState
from app.models.domain.training_data_set import TrainingDataSet
from app.models.domain.sensor import Sensor
from bson.objectid import ObjectId
from pymongo.mongo_client import MongoClient
from app.core.config import DATABASE_NAME, WORKSPACE_COLLECTION_NAME
from app.models.domain.workspace import Workspace

uri = 'mongodb://0.0.0.0/'
client = MongoClient(uri, 27017)

client.drop_database(DATABASE_NAME)

db = client[DATABASE_NAME]
coll = db[WORKSPACE_COLLECTION_NAME]

test_workspace = asdict(Workspace(
    _id=ObjectId("666f6f2d6261722d71757578"),
    user_id=ObjectId("666f6f2d6261722d71757578"),
    sensors={
        "Accelerometer": Sensor(sampling_rate=50, components=["x_Accelerometer", "y_Accelerometer", "z_Accelerometer"]),
        "Gyroscope": Sensor(sampling_rate=50, components=["x_Gyroscpoe", "y_Gyroscope", "z_Gyroscope"])
    },
    training_data_set=TrainingDataSet(last_modified=1),
    training_state=TrainingState()
))
coll.insert_one(test_workspace)
