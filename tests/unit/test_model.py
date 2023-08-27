from datetime import date, timedelta

import pandas as pd
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from pyspark.sql import SparkSession

from dbx_demand_forecast.model import DistributedModel
from dbx_demand_forecast.schema import ForecastSchema, SplitSchema
from tests.utils import assert_pyspark_df_equal

group_columns = ["store", "item"]
time_column = "date"
target_column = "sales"
test_size = 7
steps = 2


def test_model(spark: SparkSession):
    df_split = spark.createDataFrame(
        pd.DataFrame(
            {
                "store": 1,
                "item": 1,
                "date": [date(2022, 8, 8) + timedelta(i) for i in range(35)],
                "sales": [
                    2522.0,
                    2403.0,
                    2689.0,
                    2505.0,
                    2887.0,
                    2729.0,
                    2668.0,
                    2673.0,
                    2631.0,
                    2714.0,
                    2562.0,
                    2780.0,
                    2802.0,
                    2784.0,
                    2552.0,
                    2532.0,
                    2490.0,
                    2473.0,
                    2743.0,
                    2753.0,
                    2560.0,
                    2642.0,
                    2514.0,
                    2545.0,
                    2775.0,
                    2910.0,
                    2796.0,
                    2675.0,
                    2681.0,
                    2450.0,
                    2635.0,
                    2534.0,
                    2886.0,
                    2742.0,
                    2696.0,
                ],
                "split:": (["train"] * 28 + ["test"] * 7),
            }
        ),
        schema=SplitSchema,
    )
    df_train = df_split.filter(df_split.split == "train")

    df_expected_forecast_on_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": "ExponentialSmoothing",
                "store": 1,
                "item": 1,
                "date": [date(2022, 9, 5) + timedelta(i) for i in range(7)],
                "sales": map(float, range(7)),
            }
        ),
        schema=ForecastSchema,
    )
    df_expected_forecast = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": "ExponentialSmoothing",
                "store": 1,
                "item": 1,
                "date": [date(2022, 9, 12) + timedelta(i) for i in range(2)],
                "sales": map(float, range(2)),
            }
        ),
        schema=ForecastSchema,
    )

    exponential_smoothing = DistributedModel(
        group_columns,
        time_column,
        target_column,
        ExponentialSmoothing,
        {"seasonal_periods": 7},
    )
    df_forecast_on_test = exponential_smoothing.fit_predict(
        df_train,
        ForecastSchema,
        test_size,
    )
    df_forecast = exponential_smoothing.fit_predict(
        df_split, ForecastSchema, steps
    )

    assert_pyspark_df_equal(
        df_expected_forecast_on_test.drop("sales"),
        df_forecast_on_test.drop("sales"),
    )
    assert_pyspark_df_equal(
        df_expected_forecast.drop("sales"), df_forecast.drop("sales")
    )
