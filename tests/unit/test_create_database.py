import logging

import mlflow
import pytest
from pyspark.sql import SparkSession

from dbx_demand_forecast.tasks.create_database import CreateDataBaseTask

conf = {
    "env": "dev",
    "experiment": "/Shared/dbx_demand_forecast/dev_demand_forecast",
    "database": "dev",
}


@pytest.fixture(scope="session", autouse=True)
def launch_create_database_task(spark: SparkSession):
    logging.info(f"Launching {CreateDataBaseTask.__name__}")
    task = CreateDataBaseTask(spark, conf)
    task.launch()
    logging.info(f"Launching the {CreateDataBaseTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(conf["experiment"])
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_create_database(spark: SparkSession):
    assert spark.sql(f"SHOW DATABASES LIKE '{conf['database']}'").count() == 1
