import logging

import mlflow
import pytest
from pyspark.sql import SparkSession

from dbx_demand_forecast.schema import SalesSchema
from dbx_demand_forecast.tasks.ingest import IngestionTask
from dbx_demand_forecast.utils import read_csv, read_delta_table

conf = {
    "env": "default",
    "experiment": "/Shared/dbx_demand_forecast/dev_demand_forecast",
    "input": {"path": "tests/unit/data/sales", "sep": ","},
    "output": {
        "path": "tests/unit/data/input",
        "database": "default",
        "table": "input",
    },
}


@pytest.fixture(scope="session", autouse=True)
def launch_ingestion_task(spark: SparkSession):
    logging.info(f"Launching {IngestionTask.__name__}")
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
    df_test = read_csv(
        spark, conf["input"]["path"], conf["input"]["sep"], schema=SalesSchema
    )
    count_mismatch = df_test.join(df, how="anti").count()
    assert df.count() == df_test.count() and count_mismatch == 0
