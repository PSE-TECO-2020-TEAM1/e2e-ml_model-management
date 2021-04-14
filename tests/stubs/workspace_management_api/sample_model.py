from app.workspace_management_api.sample_model import DataPoint, SampleFromWorkspace, Timeframe, DataPointsPerSensor

# TODO replace values with better ones


def get_sample_from_workspace_stub_1():
    return SampleFromWorkspace(
        label="Shake",
        start=1617981582100,
        end=1617981582200,
        timeFrames=[Timeframe(1617981582100, 1617981582200)],
        sensorDataPoints=[DataPointsPerSensor(
            sensorName="Accelerometer",
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
            sensorName="Gyroscope",
            dataPoints=[
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582100),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582120),
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
        start=1617981582010,
        end=1617981582260,
        timeFrames=[Timeframe(1617981582050, 1617981582080),  Timeframe(1617981582100, 1617981582200),
                    Timeframe(1617981582210, 1617981582230)],
        sensorDataPoints=[DataPointsPerSensor(
            sensorName="Accelerometer",
            dataPoints=[
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582100),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582117),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582133),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582150),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582167),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582183),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582200),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582217)
            ]
        ), DataPointsPerSensor(
            sensorName="Gyroscope",
            dataPoints=[
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582100),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582120),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582140),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582160),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582180),
                DataPoint(data=[1.0, 1.0, 1.0], timestamp=1617981582200)
            ]
        )]
    )
