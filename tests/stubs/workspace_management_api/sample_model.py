from app.workspace_management_api.sample_model import DataPoint, SampleFromWorkspace, Timeframe, DataPointsPerSensor

# TODO replace values with better ones


def get_sample_from_workspace_stub_1():
    return SampleFromWorkspace(
        label="Shake",
        start=1617981582100,
        end=161798158200,
        timeFrames=[Timeframe(1617981582100, 161798158200)],
        sensorDataPoints=[DataPointsPerSensor(
            sensor="Accelerometer",
            dataPoints=[
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582100),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582117),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582133),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582150),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582167),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582183),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582200),

            ]
        ), DataPointsPerSensor(
            sensor="Gyroscope",
            dataPoints=[
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582120),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582100),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582140),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582160),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582180),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582200)
            ]
        )]
    )


def get_sample_from_workspace_stub_2():
    return SampleFromWorkspace(
        label="Rotate",
        start=1617981592050,
        end=161798159250,
        timeFrames=[Timeframe(1617981592070, 161798159120), Timeframe(1617981592160, 161798159220)],
        sensorDataPoints=[DataPointsPerSensor(
            sensor="Accelerometer",
            dataPoints=[
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582100),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582117),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582133),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582150),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582167),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582183),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582200),

            ]
        ), DataPointsPerSensor(
            sensor="Gyroscope",
            dataPoints=[
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582120),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582100),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582140),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582160),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582180),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582200)
            ]
        )]
    )
