import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from spark_forecast.params import IngestionParams
from spark_forecast.schema import SalesSchema
from spark_forecast.tasks.ingest import IngestionTask
from spark_forecast.utils import read_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "tz": "America/Santiago",
    "execution_date": "2018-12-31",
    "database": "default",
    "path": "",
    "sep": ",",
    "stores": [1],
}
params = IngestionParams.model_validate(conf)


def update_conf(spark: SparkSession):
    warehouse_dir = spark.conf.get("spark.hive.metastore.warehouse.dir") or ""
    params.path = os.path.join(warehouse_dir, "sales")


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
        params.path,
        header=True,
        sep=params.sep,
        mode="overwrite",
    )


@pytest.fixture(scope="module", autouse=True)
def launch_ingestion_task(spark: SparkSession):
    logging.info(f"Launching {IngestionTask.__name__}")
    update_conf(spark)
    create_sales_csv(spark)
    ingestion_task = IngestionTask(params)
    ingestion_task.launch(spark)
    logging.info(f"Launching the {IngestionTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_input(spark: SparkSession):
    df = read_delta_table(spark, params.database, "input")
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
    assert_pyspark_df_equal(df_test, df)
