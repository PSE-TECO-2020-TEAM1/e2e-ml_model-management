from app.models.schemas.prediction_data import DataPointInPredict, DataPointsPerSensorInPredict, PredictionData, SampleInPredict
from tests.stubs.workspace_management_api.sample_model import get_sample_from_workspace_stub_1, get_sample_from_workspace_stub_2


def get_sample_in_predict_stub_1():
    sample_from_workspace_stub_1 = get_sample_from_workspace_stub_1()
    return SampleInPredict(
        start=sample_from_workspace_stub_1.start,
        end=sample_from_workspace_stub_1.end,
        sensorDataPoints=[
            DataPointsPerSensorInPredict(
                sensor=data_points_per_sensor.sensorName,
                dataPoints=[
                    DataPointInPredict(
                        data=data_point.data,
                        timestamp=data_point.timestamp
                    )
                    for data_point in data_points_per_sensor.dataPoints
                ]
            )
            for data_points_per_sensor in sample_from_workspace_stub_1.sensorDataPoints
        ]
    )

def get_prediction_data_stub_1():
    return PredictionData(
        predictionId="607473b0a017269ac167142d",
        sample=get_sample_in_predict_stub_1()
    )


def get_sample_in_predict_stub_2():
    sample_from_workspace_stub_2 = get_sample_from_workspace_stub_2()
    return SampleInPredict(
        start=sample_from_workspace_stub_2.start,
        end=sample_from_workspace_stub_2.end,
        sensorDataPoints=[
            DataPointsPerSensorInPredict(
                sensor=data_points_per_sensor.sensorName,
                dataPoints=[
                    DataPointInPredict(
                        data=data_point.data,
                        timestamp=data_point.timestamp
                    )
                    for data_point in data_points_per_sensor.dataPoints
                ]
            )
            for data_points_per_sensor in sample_from_workspace_stub_2.sensorDataPoints
        ]
    )


def get_prediction_data_stub_2():
    return PredictionData(
        predictionId="607473d763f8263c211369a9",
        sample=get_sample_in_predict_stub_2()
    )
