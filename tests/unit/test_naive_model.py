import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from dbx_demand_forecast.schema import ForecastSchema, SplitSchema
from dbx_demand_forecast.tasks.naive_model import NaiveModelTask
from dbx_demand_forecast.utils import read_delta_table, write_delta_table
from tests.unit.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "experiment": "/Shared/dbx_demand_forecast/dev_demand_forecast",
    "input": {
        "path": "",
        "database": "default",
        "table": "split",
    },
    "output": {
        "forecast_on_test": {
            "path": "",
            "database": "default",
            "table": "forecast_on_test",
        },
        "all_models_forecast": {
            "path": "",
            "database": "default",
            "table": "all_models_forecast",
        },
    },
    "group_columns": ["store", "item"],
    "time_column": "date",
    "target_column": "sales",
    "model_params": {"K": 1},
    "test_size": 5,
    "steps": 2,
}


def update_conf(spark: SparkSession):
    warehouse_dir = spark.conf.get("spark.hive.metastore.warehouse.dir")
    conf["input"]["path"] = os.path.join(warehouse_dir, "split")
    conf["output"]["forecast_on_test"]["path"] = os.path.join(
        warehouse_dir, "forecast_on_test"
    )
    conf["output"]["all_models_forecast"]["path"] = os.path.join(
        warehouse_dir, "all_models_forecast"
    )


def create_split_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "store": 1,
                "item": 1,
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
        conf["input"]["path"],
        SplitSchema,
        conf["input"]["database"],
        conf["input"]["table"],
    )


@pytest.fixture(scope="session", autouse=True)
def launch_naive_model_task(spark: SparkSession):
    logging.info(f"Launching {NaiveModelTask.__name__}")
    update_conf(spark)
    create_split_table(spark)
    ingestion_task = NaiveModelTask(spark, conf)
    ingestion_task.launch()
    logging.info(f"Launching the {NaiveModelTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(conf["experiment"])
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_forecast_on_test(spark: SparkSession):
    df = read_delta_table(
        spark, path=conf["output"]["forecast_on_test"]["path"]
    )
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": NaiveModelTask.__name__,
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
        spark, path=conf["output"]["all_models_forecast"]["path"]
    )
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": NaiveModelTask.__name__,
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
