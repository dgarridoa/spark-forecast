import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from spark_forecast.params import SplitParams
from spark_forecast.schema import InputSchema, SplitSchema
from spark_forecast.tasks.split import SplitTask
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
    "time_delta": 727,
    "test_size": 5,
    "freq": "1D",
}
params = SplitParams.model_validate(conf)


def create_input_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "store": 1,
                "item": 1,
                "sales": map(float, range(1, 31)),
            }
        ),
        schema=InputSchema,
    )
    write_delta_table(spark, df, InputSchema, params.database, "input")


@pytest.fixture(scope="module", autouse=True)
def launch_split_task(spark: SparkSession):
    logging.info(f"Launching {SplitTask.__name__}")
    create_input_table(spark)
    ingestion_task = SplitTask(params)
    ingestion_task.launch(spark)
    logging.info(f"Launching the {SplitTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_split(spark: SparkSession):
    df = read_delta_table(spark, params.database, "split")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "store": 1,
                "item": 1,
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "sales": map(float, range(1, 31)),
                "split:": (
                    ["train"] * (30 - conf["test_size"])
                    + ["test"] * conf["test_size"]
                ),
            }
        ),
        schema=SplitSchema,
    )
    assert_pyspark_df_equal(df_test, df)
