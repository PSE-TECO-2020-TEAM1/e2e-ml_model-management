import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session", autouse=True)
def setup():
    

@pytest.fixture(scope="module")
def client():
    return TestClient(app)
