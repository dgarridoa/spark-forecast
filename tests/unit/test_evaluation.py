import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from dbx_demand_forecast.schema import (
    ForecastSchema,
    MetricsSchema,
    SplitSchema,
)
from dbx_demand_forecast.tasks.evaluation import EvaluationTask
from dbx_demand_forecast.utils import read_delta_table, write_delta_table
from tests.unit.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "experiment": "/Shared/dbx_demand_forecast/dev_demand_forecast",
    "input": {
        "split": {
            "path": "",
            "database": "default",
            "table": "split",
        },
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
    "output": {
        "metrics": {
            "path": "",
            "database": "default",
            "table": "metrics",
        },
        "best_models": {
            "path": "",
            "database": "default",
            "table": "best_models",
        },
        "forecast": {
            "path": "",
            "database": "default",
            "table": "forecast",
        },
    },
    "group_columns": ["store", "item"],
    "time_column": "date",
    "target_column": "sales",
    "metrics": ["rmse", "mae"],
    "model_selection_metric": "mae",
}


def update_conf(spark: SparkSession):
    warehouse_dir = spark.conf.get("spark.hive.metastore.warehouse.dir")
    conf["input"]["split"]["path"] = os.path.join(warehouse_dir, "split")
    conf["input"]["forecast_on_test"]["path"] = os.path.join(
        warehouse_dir, "forecast_on_test"
    )
    conf["input"]["all_models_forecast"]["path"] = os.path.join(
        warehouse_dir, "all_models_forecast"
    )
    conf["output"]["metrics"]["path"] = os.path.join(warehouse_dir, "metrics")
    conf["output"]["best_models"]["path"] = os.path.join(
        warehouse_dir, "best_models"
    )
    conf["output"]["forecast"]["path"] = os.path.join(
        warehouse_dir, "forecast"
    )


def create_split_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(10)],
                "store": 1,
                "item": 1,
                "sales": map(float, range(1, 11)),
                "split": ["train"] * 8 + ["test"] * 2,
            }
        ),
        schema=SplitSchema,
    )

    write_delta_table(
        spark,
        df,
        conf["input"]["split"]["path"],
        SplitSchema,
        conf["input"]["split"]["database"],
        conf["input"]["split"]["table"],
    )


def create_forecast_on_test_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["NaiveModel"] * 2 + ["RandomForest"] * 2,
                "store": [1] * 4,
                "item": [1] * 4,
                "date": [date(2018, 12, 9), date(2018, 12, 10)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    write_delta_table(
        spark,
        df,
        conf["input"]["forecast_on_test"]["path"],
        ForecastSchema,
        conf["input"]["forecast_on_test"]["database"],
        conf["input"]["forecast_on_test"]["table"],
    )


def create_all_models_forecast_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["NaiveModel"] * 2 + ["RandomForest"] * 2,
                "store": [1] * 4,
                "item": [1] * 4,
                "date": [date(2018, 12, 9), date(2018, 12, 10)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    write_delta_table(
        spark,
        df,
        conf["input"]["all_models_forecast"]["path"],
        ForecastSchema,
        conf["input"]["all_models_forecast"]["database"],
        conf["input"]["all_models_forecast"]["table"],
    )


@pytest.fixture(scope="session", autouse=True)
def launch_evaluation_task(spark: SparkSession):
    logging.info(f"Launching {EvaluationTask.__name__}")
    update_conf(spark)
    create_split_table(spark)
    create_forecast_on_test_table(spark)
    create_all_models_forecast_table(spark)
    evaluation_task = EvaluationTask(spark, conf)
    evaluation_task.launch()
    logging.info(f"Launching the {EvaluationTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(conf["experiment"])
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_metrics(spark: SparkSession):
    df = read_delta_table(spark, path=conf["output"]["metrics"]["path"])
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["NaiveModel"] * 2 + ["RandomForest"] * 2,
                "store": [1] * 4,
                "item": [1] * 4,
                "metric": ["rmse", "mae", "rmse", "mae"],
                "value": [1.750714, 1.75, 0.608276, 0.6],
            }
        ),
        schema=MetricsSchema,
    )
    count_mismatch = df_test.join(df, how="anti").count()
    assert df.count() == df_test.count()
    assert count_mismatch == 0


def test_best_models(spark: SparkSession):
    df = read_delta_table(spark, path=conf["output"]["best_models"]["path"])
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["RandomForest"],
                "store": [1],
                "item": [1],
                "metric": ["mae"],
                "value": [0.600],
            }
        ),
        schema=MetricsSchema,
    )
    assert_pyspark_df_equal(df_test, df)


def test_forecast(spark: SparkSession):
    df = read_delta_table(spark, path=conf["output"]["forecast"]["path"])
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["RandomForest"] * 2,
                "store": [1] * 2,
                "item": [1] * 2,
                "date": [date(2018, 12, 9), date(2018, 12, 10)],
                "sales": [8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    assert_pyspark_df_equal(df_test, df)
