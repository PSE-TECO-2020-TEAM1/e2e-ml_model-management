from app.ml.training.config_parsing import parse_config_for_response, parse_config_for_training
from tests.stubs.models.schemas.training_config import get_training_config_in_response_stub_5_1, get_training_config_in_train_stub_5_1
from tests.stubs.models.domain.ml_model import get_training_config_stub_5_1
import pytest

@pytest.fixture
def training_config():
    return get_training_config_stub_5_1()

@pytest.fixture
def training_config_in_train():
    return get_training_config_in_train_stub_5_1()

@pytest.fixture
def training_config_in_response():
    return get_training_config_in_response_stub_5_1()

def test_parse_config_for_training(training_config, training_config_in_train):
    assert parse_config_for_training(training_config_in_train) == training_config

def test_parse_config_for_response(training_config, training_config_in_response):
    assert parse_config_for_response(training_config) == training_config_in_response
