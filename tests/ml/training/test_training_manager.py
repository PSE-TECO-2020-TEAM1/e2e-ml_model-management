from app.ml.objects.normalization.enum import Normalization
from app.ml.objects.imputation.enum import Imputation
from app.ml.objects.feature.enum import Feature
from app.models.schemas.training_config import PerComponentConfigInTrain
from app.ml.training.training_manager import validate_config_and_parse_hyperparameters
from tests.stubs.models.schemas.training_config import get_training_config_in_train_stub_5_1
from tests.stubs.models.domain.workspace import get_workspace_stub
import pytest
from unittest import mock


@pytest.fixture
def valid_training_config():
    return get_training_config_in_train_stub_5_1()


@pytest.fixture
def workspace_sensors():
    return get_workspace_stub().sensors

@mock.patch("app.ml.training.training_manager.validate_and_parse_hyperparameters")
def test_validate_config_and_parse_hyperparameters_with_valid_config(mock, valid_training_config, workspace_sensors):
    validate_config_and_parse_hyperparameters(valid_training_config, workspace_sensors)
    mock.assert_called()

def test_validate_config_and_parse_hyperparameters_with_windowSize_less_than_4(valid_training_config, workspace_sensors):
    valid_training_config.windowSize = 2
    with pytest.raises(ValueError, match="Window size must be greater than or equal to 4"):
        validate_config_and_parse_hyperparameters(valid_training_config, workspace_sensors)


def test_validate_config_and_parse_hyperparameters_with_slidingStep_less_than_1(valid_training_config, workspace_sensors):
    valid_training_config.slidingStep = 0
    with pytest.raises(ValueError, match="Sliding step must be greater than or equal to 1"):
        validate_config_and_parse_hyperparameters(valid_training_config, workspace_sensors)


def test_validate_config_and_parse_hyperparameters_with_slidingStep_greater_than_windowSize(valid_training_config, workspace_sensors):
    valid_training_config.windowSize = 8
    valid_training_config.slidingStep = 9
    with pytest.raises(ValueError, match="Sliding step cannot excess the window size"):
        validate_config_and_parse_hyperparameters(valid_training_config, workspace_sensors)

def test_validate_config_and_parse_hyperparameters_with_invalid_sensor(valid_training_config, workspace_sensors):
    valid_training_config.perComponentConfigs.append(PerComponentConfigInTrain(
        sensor="Magnetometer",
        component="x",
        features=[Feature.MAXIMUM],
        imputation=Imputation.MEAN_IMPUTATION,
        normalizer=Normalization.NORMALIZER
    ))
    with pytest.raises(ValueError, match=" is not a sensor of this workspace"):
        validate_config_and_parse_hyperparameters(valid_training_config, workspace_sensors)

def test_validate_config_and_parse_hyperparameters_with_invalid_sensor_component(valid_training_config, workspace_sensors):
    valid_training_config.perComponentConfigs.append(PerComponentConfigInTrain(
        sensor="Accelerometer",
        component="a",
        features=[Feature.MAXIMUM],
        imputation=Imputation.MEAN_IMPUTATION,
        normalizer=Normalization.NORMALIZER
    ))
    with pytest.raises(ValueError, match=" is not a valid component of "):
        validate_config_and_parse_hyperparameters(valid_training_config, workspace_sensors)

# invalid classifier and hyperparameters are tested in the test cases of validate_and_parse_hyperparameters