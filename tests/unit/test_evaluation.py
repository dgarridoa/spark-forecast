from datetime import date, timedelta

import pandas as pd
from pyspark.sql import Row, SparkSession

from dbx_demand_forecast.evaluation import Evaluation, mape
from dbx_demand_forecast.schema import (
    ForecastSchema,
    MetricsSchema,
    SplitSchema,
)
from tests.utils import assert_pyspark_df_equal

group_columns = ["store", "item"]
time_column = "date"
target_column = "sales"
metrics = ["mae", "rmse", mape(epsilon=0)]
model_selection_metric = "mae"


def test_evaluation(spark: SparkSession):
    df_split = spark.createDataFrame(
        pd.DataFrame(
            {
                "store": 1,
                "item": 1,
                "date": [date(2022, 9, 2) + timedelta(i) for i in range(10)],
                "sales": map(float, range(1, 11)),
                "split": ["train"] * 8 + ["test"] * 2,
            }
        ),
        schema=SplitSchema,
    )
    df_test = df_split.filter(df_split.split == "test")
    df_forecast_on_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["ExponentialSmoothing"] * 2 + ["Prophet"] * 2,
                "store": [1] * 4,
                "item": [1] * 4,
                "date": [date(2022, 9, 10), date(2022, 9, 11)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    df_all_models_forecast = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["ExponentialSmoothing"] * 2 + ["Prophet"] * 2,
                "store": [1] * 4,
                "item": [1] * 4,
                "date": [date(2022, 9, 12), date(2022, 9, 13)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )

    df_expected_metrics = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["ExponentialSmoothing"] * 3 + ["Prophet"] * 3,
                "store": [1] * 6,
                "item": [1] * 6,
                "metric": ["rmse", "mae", "mape", "rmse", "mae", "mape"],
                "value": [1.750714, 1.75, 0.184444, 0.608276, 0.6, 0.062778],
            }
        ),
        schema=MetricsSchema,
    )
    df_expected_best_models = spark.createDataFrame(
        [
            Row(
                model="Prophet",
                store=1,
                item=1,
                metric="mae",
                value=0.600,
            )
        ],
        schema=MetricsSchema,
    )
    df_expected_forecast = spark.createDataFrame(
        [
            Row(
                model="Prophet",
                store=1,
                item=1,
                date=date(2022, 9, 12),
                sales=8.5,
            ),
            Row(
                model="Prophet",
                store=1,
                item=1,
                date=date(2022, 9, 13),
                sales=10.7,
            ),
        ],
        schema=ForecastSchema,
    )

    evaluation = Evaluation(
        group_columns,
        time_column,
        target_column,
        metrics,
        model_selection_metric,
    )
    df_metrics = evaluation.get_metrics(
        df_test, df_forecast_on_test, MetricsSchema
    )
    df_best_models = evaluation.get_best_models(df_metrics)
    df_forecast = evaluation.get_forecast(
        df_all_models_forecast, df_best_models
    )

    assert_pyspark_df_equal(df_expected_metrics, df_metrics)
    assert_pyspark_df_equal(df_expected_best_models, df_best_models)
    assert_pyspark_df_equal(df_expected_forecast, df_forecast)
