import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from spark_forecast.schema import ForecastSchema, MetricsSchema, SplitSchema
from spark_forecast.tasks.evaluation import EvaluationTask
from spark_forecast.utils import read_delta_table, write_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "input": {
        "split": {
            "database": "default",
            "table": "split",
        },
        "forecast_on_test": {
            "database": "default",
            "table": "forecast_on_test",
        },
        "all_models_forecast": {
            "database": "default",
            "table": "all_models_forecast",
        },
    },
    "output": {
        "metrics": {
            "database": "default",
            "table": "metrics",
        },
        "best_models": {
            "database": "default",
            "table": "best_models",
        },
        "forecast": {
            "database": "default",
            "table": "forecast",
        },
    },
    "group_columns": ["store", "item"],
    "time_column": "date",
    "target_column": "sales",
    "metrics": ["rmse", "mae"],
    "model_selection_metric": "mae",
    "execution_date": "2018-12-31",
    "freq": "1D",
}


def create_split_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "store": 1,
                "item": 1,
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(10)],
                "sales": map(float, range(1, 11)),
                "split": ["train"] * 8 + ["test"] * 2,
            }
        ),
        schema=SplitSchema,
    )

    write_delta_table(
        spark,
        df,
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
        ForecastSchema,
        conf["input"]["forecast_on_test"]["database"],
        conf["input"]["forecast_on_test"]["table"],
        ["model"],
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
        ForecastSchema,
        conf["input"]["all_models_forecast"]["database"],
        conf["input"]["all_models_forecast"]["table"],
        ["model"],
    )


@pytest.fixture(scope="module", autouse=True)
def launch_evaluation_task(spark: SparkSession):
    logging.info(f"Launching {EvaluationTask.__name__}")
    create_split_table(spark)
    create_forecast_on_test_table(spark)
    create_all_models_forecast_table(spark)
    evaluation_task = EvaluationTask(spark, conf)
    evaluation_task.launch()
    logging.info(f"Launching the {EvaluationTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_metrics(spark: SparkSession):
    df = read_delta_table(
        spark,
        conf["output"]["metrics"]["database"],
        conf["output"]["metrics"]["table"],
    )
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
    assert_pyspark_df_equal(df_test, df)


def test_best_models(spark: SparkSession):
    df = read_delta_table(
        spark,
        conf["output"]["best_models"]["database"],
        conf["output"]["best_models"]["table"],
    )
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
    df = read_delta_table(
        spark,
        conf["output"]["forecast"]["database"],
        conf["output"]["forecast"]["table"],
    )
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
