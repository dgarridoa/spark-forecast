import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from dbx_demand_forecast.schema import SalesSchema
from dbx_demand_forecast.tasks.ingest import IngestionTask
from dbx_demand_forecast.utils import read_delta_table

conf = {
    "env": "default",
    "experiment": "/Shared/dbx_demand_forecast/dev_demand_forecast",
    "input": {"path": "", "sep": ","},
    "output": {
        "path": "",
        "database": "default",
        "table": "input",
    },
}


def update_conf(spark: SparkSession):
    warehouse_dir = spark.conf.get("spark.hive.metastore.warehouse.dir")
    conf["input"]["path"] = os.path.join(warehouse_dir, "sales")
    conf["output"]["path"] = os.path.join(warehouse_dir, "input")


def create_sales_csv(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "store": 1,
                "item": 1,
                "sales": map(float, range(1, 31)),
            }
        ),
        schema=SalesSchema,
    )
    df.write.csv(
        conf["input"]["path"],
        header=True,
        sep=conf["input"]["sep"],
        mode="overwrite",
    )


@pytest.fixture(scope="session", autouse=True)
def launch_ingestion_task(spark: SparkSession):
    logging.info(f"Launching {IngestionTask.__name__}")
    update_conf(spark)
    create_sales_csv(spark)
    ingestion_task = IngestionTask(spark, conf)
    ingestion_task.launch()
    logging.info(f"Launching the {IngestionTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(conf["experiment"])
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_input(spark: SparkSession):
    df = read_delta_table(spark, path=conf["output"]["path"])
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "store": 1,
                "item": 1,
                "sales": map(float, range(1, 31)),
            }
        ),
        schema=SalesSchema,
    )

    count_mismatch = df_test.join(df, how="anti").count()
    assert df.count() == df_test.count() and count_mismatch == 0
