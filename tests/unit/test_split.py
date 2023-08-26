import logging
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from dbx_demand_forecast.schema import InputSchema, SplitSchema
from dbx_demand_forecast.tasks.split import SplitTask
from dbx_demand_forecast.utils import read_delta_table, write_delta_table
from tests.unit.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "experiment": "/Shared/dbx_demand_forecast/dev_demand_forecast",
    "input": {
        "database": "default",
        "table": "input",
    },
    "output": {
        "database": "default",
        "table": "split",
    },
    "test_size": 5,
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
    experiment = mlflow.get_experiment_by_name(conf["experiment"])
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
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "store": 1,
                "item": 1,
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
