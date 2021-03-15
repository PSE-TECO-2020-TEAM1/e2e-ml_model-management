import pickle

from bson import ObjectId
from gridfs import GridFS
import random
from pandas.core.frame import DataFrame

from pymongo.mongo_client import MongoClient

uri = 'mongodb://0.0.0.0/'
client = MongoClient(uri, 27017)
db = client.test
fs = GridFS(db)

DATA_POINTS = 1000

def fillData():
    sensors = ["Accelerometer", "Gyroscope"]
    format = ["x", "y", "z"]
    data = []
    for i in range(DATA_POINTS):
        intermediate = {}
        for sensor in sensors:
            for j in format:
                intermediate[sensor+"_"+j] = random.randint(1, 100)
        data.append(intermediate)
    return fs.put(pickle.dumps(DataFrame(data)))


def runtest():
    workspace_id = ObjectId('666f6f2d6261722d71757578')
    user_id = ObjectId('666f6f2d6261722d71757578')
    client.drop_database("test")
    samples_list1 = [{
        "label": "red",
        "sensorDataPoints": fillData()
    }]

    samples_list2 =[{
        "label": "blue",
        "sensorDataPoints": fillData()
    }]
    
    db.workspaces.insert_one(
        {
            "_id": workspace_id,
            "userId": user_id,
            "mlModels": [],
            "progress": -1,
            "sensors": [{"name": "Accelerometer", "samplingRate": 2, "dataFormat": ["x", "y", "z"]},
                        {"name": "Gyroscope", "samplingRate": 3, "dataFormat": ["x", "y", "z"]}],
            "workspaceData": {
                "labelToLabelCode": {"blue": "1", "red": "2"},
                "labelCodeToLabel": {"1": "blue", "2": "red"},
                "slidingWindows": {},
                "samples": samples_list1 + samples_list2
            }
        }
    )

    workspace_id2 = ObjectId('666f6f2d6261722d71757579')
    user_id = ObjectId('666f6f2d6261722d71757578')

    db.workspaces.insert_one(
        {
            "_id": workspace_id2,
            "userId": user_id,
            "mlModels": [],
            "progress": -1,
            "sensors": [{"name": "Accelerometer", "samplingRate": 2, "dataFormat": ["x", "y", "z"]},
                        {"name": "Gyroscope", "samplingRate": 3, "dataFormat": ["x", "y", "z"]}],
            "workspaceData": {
                "labelToLabelCode": {"blue": 1, "red": 2},
                "labelCodeToLabel": {"1": "blue", "2": "red"},
                "slidingWindows": {},
                "samples": samples_list1 + samples_list2
            }
        }
    )

runtest()
