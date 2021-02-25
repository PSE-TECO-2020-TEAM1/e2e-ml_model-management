import pickle

from bson import ObjectId
from gridfs import GridFS
import random

from pymongo.mongo_client import MongoClient

uri = 'mongodb://0.0.0.0/'
client = MongoClient(uri, 27017)
db = client.test
fs = GridFS(db)

def fillData(num):
    data_points = []
    for i in range(num):
        data_points.append([random.randint(1, 100) for i in range(3)])
    return data_points

DATA_POINTS = 1000

def fillDataPoints():
    sensors = ["accelerometer", "gyroscope"]
    object = {}
    for sensor in sensors:
        object[sensor] = fillData(DATA_POINTS)
    return fs.put(pickle.dumps(object))

def runtest():
    workspace_id = ObjectId('666f6f2d6261722d71757578')
    user_id = ObjectId('666f6f2d6261722d71757578')
    client.drop_database("test")

    samples_list1 = [{
        "label": "red",
        "dataPointCount": DATA_POINTS,
        "timeframes": [{
            "start": -555555,
            "end": 999999
        }],
        "sensorDataPoints": fillDataPoints()
    }]

    samples_list2 =[{
        "label": "blue",
        "dataPointCount": DATA_POINTS,
        "timeframes": [{
            "start": -555555,
            "end": 999999
        }],
        "sensorDataPoints": fillDataPoints()
    }]
    
    db.workspaces.insert_one(
        {
            "_id": workspace_id,
            "userId": user_id,
            "mlModels": [],
            "progress": -1,
            "sensors": [{"name": "Accelerometer", "samplingRate": 50},
                        {"name": "Gyroscope", "samplingRate": 75}],
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
            "sensors": [{"name": "Accelerometer", "samplingRate": 50},
                        {"name": "Gyroscope", "samplingRate": 75}],
            "workspaceData": {
                "labelToLabelCode": {"blue": 1, "red": 2},
                "labelCodeToLabel": {"1": "blue", "2": "red"},
                "slidingWindows": {},
                "samples": samples_list1 + samples_list2
            }
        }
    )

runtest()
