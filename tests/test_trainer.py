import mongomock
import responses
import pytest

from app.training.trainer import Trainer

#TODO rename when there are more tests
class MockSingleBasicSample():
    def json():
        sample = {}

        return [sample]

@mongomock.patch()
@responses.activate
def test_train():
    
    trainer = Trainer()