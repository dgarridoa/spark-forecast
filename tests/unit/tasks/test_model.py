import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pyspark.sql.functions as F
import pytest
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from pyspark.sql import SparkSession

from spark_forecast.params import ModelParams
from spark_forecast.schema import ForecastSchema, SplitSchema
from spark_forecast.tasks.model import ModelTask
from spark_forecast.utils import read_delta_table, write_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "tz": "America/Santiago",
    "execution_date": "2018-12-31",
    "database": "default",
    "group_columns": ["store", "item"],
    "time_column": "date",
    "target_column": "sales",
    "model_cls": "ExponentialSmoothing",
    "model_params": {"seasonal_periods": 7},
    "test_size": 5,
    "steps": 2,
    "freq": "1D",
}
params = ModelParams.model_validate(conf)


def create_split_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "store": 1,
                "item": 1,
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "sales": map(float, range(1, 31)),
                "split": (
                    ["train"] * (30 - params.test_size)
                    + ["test"] * params.test_size
                ),
            }
        ),
        schema=SplitSchema,
    )
    write_delta_table(spark, df, SplitSchema, params.database, "split")


@pytest.fixture(scope="module", autouse=True)
def launch_model_task(spark: SparkSession):
    logging.info(f"Launching {ModelTask.__name__}")
    create_split_table(spark)
    task = ModelTask(params)
    task.launch(spark)
    logging.info(f"Launching the {ModelTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_forecast_on_test(spark: SparkSession):
    df = read_delta_table(
        spark,
        params.database,
        "forecast_on_test",
    ).filter(F.col("model") == params.model_cls)
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": params.model_cls,
                "store": 1,
                "item": 1,
                "date": [
                    date(2018, 12, 26) + timedelta(i)
                    for i in range(params.test_size)
                ],
                "sales": map(float, range(26, 31)),
            }
        ),
        schema=ForecastSchema,
    )
    assert_pyspark_df_equal(df_test.drop("sales"), df.drop("sales"))


def test_all_models_forecast(spark: SparkSession):
    df = read_delta_table(
        spark,
        params.database,
        "all_models_forecast",
    ).filter(F.col("model") == params.model_cls)
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ExponentialSmoothing.__name__,
                "store": 1,
                "item": 1,
                "date": [
                    date(2018, 12, 31) + timedelta(i)
                    for i in range(params.steps)
                ],
                "sales": map(float, range(31, 33)),
            }
        ),
        schema=ForecastSchema,
    )
    assert_pyspark_df_equal(df_test.drop("sales"), df.drop("sales"))
