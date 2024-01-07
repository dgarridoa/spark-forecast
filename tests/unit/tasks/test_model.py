import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pyspark.sql.functions as F
import pytest
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from pyspark.sql import SparkSession

from spark_forecast.schema import ForecastSchema, SplitSchema
from spark_forecast.tasks.model import ModelTask
from spark_forecast.utils import read_delta_table, write_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "input": {
        "database": "default",
        "table": "split",
    },
    "output": {
        "forecast_on_test": {
            "database": "default",
            "table": "forecast_on_test",
        },
        "all_models_forecast": {
            "database": "default",
            "table": "all_models_forecast",
        },
    },
    "group_columns": ["store", "item"],
    "time_column": "date",
    "target_column": "sales",
    "model_params": {"seasonal_periods": 7},
    "test_size": 5,
    "steps": 2,
    "execution_date": "2018-12-31",
    "freq": "1D",
}


def create_split_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "store": 1,
                "item": 1,
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "sales": map(float, range(1, 31)),
                "split": (
                    ["train"] * (30 - conf["test_size"])
                    + ["test"] * conf["test_size"]
                ),
            }
        ),
        schema=SplitSchema,
    )
    write_delta_table(
        spark,
        df,
        SplitSchema,
        conf["input"]["database"],
        conf["input"]["table"],
    )


@pytest.fixture(scope="module", autouse=True)
def launch_model_task(spark: SparkSession):
    logging.info(f"Launching {ModelTask.__name__}")
    create_split_table(spark)
    task = ModelTask(spark, conf, ExponentialSmoothing)
    task.launch()
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
        conf["output"]["forecast_on_test"]["database"],
        conf["output"]["forecast_on_test"]["table"],
    ).filter(F.col("model") == "ExponentialSmoothing")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ExponentialSmoothing.__name__,
                "store": 1,
                "item": 1,
                "date": [
                    date(2018, 12, 26) + timedelta(i)
                    for i in range(conf["test_size"])
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
        conf["output"]["all_models_forecast"]["database"],
        conf["output"]["all_models_forecast"]["table"],
    ).filter(F.col("model") == "ExponentialSmoothing")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ExponentialSmoothing.__name__,
                "store": 1,
                "item": 1,
                "date": [
                    date(2018, 12, 31) + timedelta(i)
                    for i in range(conf["steps"])
                ],
                "sales": map(float, range(31, 33)),
            }
        ),
        schema=ForecastSchema,
    )
    assert_pyspark_df_equal(df_test.drop("sales"), df.drop("sales"))
