from app.models.workspace import Sensor
from bson.objectid import ObjectId
import pytest

#TODO rename when there are more tests
@pytest.fixture
def first_create_workspace_request():
    return {
        "workspaceId": str(ObjectId()),
        "sensors": [
            Sensor(name="Accelerometer", samplingRate=50, dataFormat=["x", "y", "z"])
        ]
    }


#TODO rename when there are more tests
class MockSingleBasicSample():
    def json():
        sample = {}

        return [sample]

def test_train(client, monkeypatch):
    monkeypatch.patch()