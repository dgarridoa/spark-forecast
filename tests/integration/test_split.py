import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from spark_forecast.schema import InputSchema, SplitSchema
from spark_forecast.tasks.split import SplitTask
from spark_forecast.utils import read_delta_table, write_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "input": {
        "database": "default",
        "table": "input",
    },
    "output": {
        "database": "default",
        "table": "split",
    },
    "group_columns": ["store", "item"],
    "time_column": "date",
    "target_column": "sales",
    "test_size": 5,
    "execution_date": "2018-12-31",
    "time_delta": 727,
    "freq": "1D",
}


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
    write_delta_table(
        spark,
        df,
        InputSchema,
        conf["input"]["database"],
        conf["input"]["table"],
    )


@pytest.fixture(scope="session", autouse=True)
def launch_split_task(spark: SparkSession):
    logging.info(f"Launching {SplitTask.__name__}")
    create_input_table(spark)
    ingestion_task = SplitTask(spark, conf)
    ingestion_task.launch()
    logging.info(f"Launching the {SplitTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_split(spark: SparkSession):
    df = read_delta_table(
        spark, conf["output"]["database"], conf["output"]["table"]
    )
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
