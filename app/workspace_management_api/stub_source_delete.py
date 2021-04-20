from app.workspace_management_api.sample_model import DataPoint, DataPointsPerSensor, SampleFromWorkspace, Timeframe
from typing import List
from app.workspace_management_api.data_source import ExternalDataSource
from bson.objectid import ObjectId
import random

class WorkspaceDataSource(ExternalDataSource):

    @staticmethod
    def last_modified(user_id: ObjectId, workspace_id: ObjectId) -> int:
        return 2

    @staticmethod
    def fetch_samples(user_id: ObjectId, workspace_id: ObjectId) -> List[SampleFromWorkspace]:
        samples = []
        for i in range(50):
            samples.append(random_sample("green"))
            samples.append(random_sample("blue"))
        print("Samples created")
        return samples

def random_sample(label: str):
    params = {}
    params["label"] = label
    params["start"] = 0
    params["end"] = 5000
    params["timeFrames"] = [Timeframe(0, 5000)]
    params["sensorDataPoints"] = [random_data_points_per_sensor("Accelerometer"), random_data_points_per_sensor("Gyroscope")]
    return SampleFromWorkspace(**params)

def random_data_points_per_sensor(sensor_name: str):
    params = {}
    params["sensorName"] = sensor_name
    params["dataPoints"] = [DataPoint(data=[random.randint(1, 100) for j in range(3)], timestamp=i*50) for i in range(100)]
    return DataPointsPerSensor(**params)