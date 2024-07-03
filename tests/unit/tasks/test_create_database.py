import logging
import os

import mlflow
import pytest
from pyspark.sql import SparkSession

from spark_forecast.params import CommonParams
from spark_forecast.tasks.create_database import CreateDataBaseTask

conf = {
    "env": "dev",
    "tz": "America/Santiago",
    "database": "dev",
}
params = CommonParams.model_validate(conf)


@pytest.fixture(scope="module", autouse=True)
def launch_create_database_task(spark: SparkSession):
    logging.info(f"Launching {CreateDataBaseTask.__name__}")
    task = CreateDataBaseTask(params)
    task.launch(spark)
    logging.info(f"Launching the {CreateDataBaseTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_create_database(spark: SparkSession):
    assert spark.catalog.databaseExists(params.database)
