from app.models.workspace import Sensor
from bson.objectid import ObjectId
import pytest
import json

def test_hyperparameters(client):
    expected = json.loads(
    """{
        "features": [
            "minimum",
            "maximum",
            "variance",
            "abs_energy",
            "mean",
            "median",
            "skewness",
            "kurtosis"
        ],
        "imputers": [
            "mean_imputation",
            "zero_interpolation",
            "linear_interpolation",
            "quadratic_interpolation",
            "cubic_interpolation",
            "moving_average_imputation",
            "last_observation_carried_forward_imputation"
        ],
        "normalizers": [
            "min_max_scaler",
            "normalizer",
            "quantile_transformer",
            "robust_scaler",
            "standard_scaler"
        ],
        "windowSize": [
            4,
            500
        ],
        "slidingStep": [
            1,
            250
        ],
        "classifierSelections": [
            {
                "classifier": "kneighbors_classifier",
                "hyperparameters": {
                    "n_neighbors": {
                        "lower": 1,
                        "upper": 100,
                        "default_value": 1,
                        "type": "UniformIntegerHyperparameter"
                    },
                    "p": {
                        "choices": [
                            1,
                            2
                        ],
                        "default_value": 2,
                        "type": "CategoricalHyperparameter"
                    },
                    "weights": {
                        "choices": [
                            "uniform",
                            "distance"
                        ],
                        "default_value": "uniform",
                        "type": "CategoricalHyperparameter"
                    }
                },
                "conditions": []
            },
            {
                "classifier": "mlp_classifier",
                "hyperparameters": {
                    "activation": {
                        "choices": [
                            "tanh",
                            "relu"
                        ],
                        "default_value": "relu",
                        "type": "CategoricalHyperparameter"
                    },
                    "alpha": {
                        "lower": 1e-07,
                        "upper": 0.1,
                        "default_value": 0.0001,
                        "type": "UniformFloatHyperparameter"
                    },
                    "batch_size": {
                        "value": "auto",
                        "type": "Constant"
                    },
                    "beta_1": {
                        "value": 0.9,
                        "type": "Constant"
                    },
                    "beta_2": {
                        "value": 0.999,
                        "type": "Constant"
                    },
                    "early_stopping": {
                        "choices": [
                            "valid",
                            "train"
                        ],
                        "default_value": "valid",
                        "type": "CategoricalHyperparameter"
                    },
                    "epsilon": {
                        "value": 1e-08,
                        "type": "Constant"
                    },
                    "hidden_layer_depth": {
                        "lower": 1,
                        "upper": 3,
                        "default_value": 1,
                        "type": "UniformIntegerHyperparameter"
                    },
                    "learning_rate_init": {
                        "lower": 0.0001,
                        "upper": 0.5,
                        "default_value": 0.001,
                        "type": "UniformFloatHyperparameter"
                    },
                    "n_iter_no_change": {
                        "value": 32,
                        "type": "Constant"
                    },
                    "num_nodes_per_layer": {
                        "lower": 16,
                        "upper": 264,
                        "default_value": 32,
                        "type": "UniformIntegerHyperparameter"
                    },
                    "shuffle": {
                        "value": "True",
                        "type": "Constant"
                    },
                    "solver": {
                        "value": "adam",
                        "type": "Constant"
                    },
                    "tol": {
                        "value": 0.0001,
                        "type": "Constant"
                    },
                    "validation_fraction": {
                        "value": 0.1,
                        "type": "Constant"
                    }
                },
                "conditions": [
                    "validation_fraction | early_stopping in {'valid'}"
                ]
            },
            {
                "classifier": "random_forest_classifier",
                "hyperparameters": {
                    "bootstrap": {
                        "choices": [
                            "True",
                            "False"
                        ],
                        "default_value": "True",
                        "type": "CategoricalHyperparameter"
                    },
                    "criterion": {
                        "choices": [
                            "gini",
                            "entropy"
                        ],
                        "default_value": "gini",
                        "type": "CategoricalHyperparameter"
                    },
                    "max_depth": {
                        "value": "None",
                        "type": "Constant"
                    },
                    "max_features": {
                        "lower": 0.0,
                        "upper": 1.0,
                        "default_value": 0.5,
                        "type": "UniformFloatHyperparameter"
                    },
                    "max_leaf_nodes": {
                        "value": "None",
                        "type": "Constant"
                    },
                    "min_impurity_decrease": {
                        "value": 0.0,
                        "type": "Constant"
                    },
                    "min_samples_leaf": {
                        "lower": 1,
                        "upper": 20,
                        "default_value": 1,
                        "type": "UniformIntegerHyperparameter"
                    },
                    "min_samples_split": {
                        "lower": 2,
                        "upper": 20,
                        "default_value": 2,
                        "type": "UniformIntegerHyperparameter"
                    },
                    "min_weight_fraction_leaf": {
                        "value": 0.0,
                        "type": "Constant"
                    }
                },
                "conditions": []
            },
            {
                "classifier": "svc_classifier",
                "hyperparameters": {
                    "C": {
                        "lower": 0.03125,
                        "upper": 32768.0,
                        "default_value": 1.0,
                        "type": "UniformFloatHyperparameter"
                    },
                    "gamma": {
                        "lower": 3.0517578125e-05,
                        "upper": 8.0,
                        "default_value": 0.1,
                        "type": "UniformFloatHyperparameter"
                    },
                    "kernel": {
                        "choices": [
                            "rbf",
                            "poly",
                            "sigmoid"
                        ],
                        "default_value": "rbf",
                        "type": "CategoricalHyperparameter"
                    },
                    "max_iter": {
                        "value": -1,
                        "type": "Constant"
                    },
                    "shrinking": {
                        "choices": [
                            "True",
                            "False"
                        ],
                        "default_value": "True",
                        "type": "CategoricalHyperparameter"
                    },
                    "tol": {
                        "lower": 1e-05,
                        "upper": 0.1,
                        "default_value": 0.001,
                        "type": "UniformFloatHyperparameter"
                    },
                    "coef0": {
                        "lower": -1.0,
                        "upper": 1.0,
                        "default_value": 0.0,
                        "type": "UniformFloatHyperparameter"
                    },
                    "degree": {
                        "lower": 2,
                        "upper": 5,
                        "default_value": 3,
                        "type": "UniformIntegerHyperparameter"
                    }
                },
                "conditions": [
                    "coef0 | kernel in {'poly', 'sigmoid'}",
                    "degree | kernel == 'poly'"
                ]
            }
        ]
    }""")
    json_response = client.get("/api/parameters").json()
    assert json_response == expected

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