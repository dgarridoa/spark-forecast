import logging
import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from spark_forecast.params import EvaluationParams
from spark_forecast.schema import ForecastSchema, MetricsSchema, SplitSchema
from spark_forecast.tasks.evaluation import EvaluationTask
from spark_forecast.utils import read_delta_table, write_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "tz": "America/Santiago",
    "execution_date": "2018-12-31",
    "database": "default",
    "group_columns": ["store", "item"],
    "time_column": "date",
    "target_column": "sales",
    "metrics": ["rmse", "mae"],
    "model_selection_metric": "mae",
    "freq": "1D",
}
params = EvaluationParams.model_validate(conf)


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

    write_delta_table(spark, df, SplitSchema, params.database, "split")


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
        params.database,
        "forecast_on_test",
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
        params.database,
        "all_models_forecast",
        ["model"],
    )


@pytest.fixture(scope="module", autouse=True)
def launch_evaluation_task(spark: SparkSession):
    logging.info(f"Launching {EvaluationTask.__name__}")
    create_split_table(spark)
    create_forecast_on_test_table(spark)
    create_all_models_forecast_table(spark)
    evaluation_task = EvaluationTask(params)
    evaluation_task.launch(spark)
    logging.info(f"Launching the {EvaluationTask.__name__} - done")


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_metrics(spark: SparkSession):
    df = read_delta_table(spark, params.database, "metrics")
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
    assert_pyspark_df_equal(df_test, df, 2)


def test_best_models(spark: SparkSession):
    df = read_delta_table(spark, params.database, "best_models")
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
    assert_pyspark_df_equal(df_test, df, 2)


def test_forecast(spark: SparkSession):
    df = read_delta_table(spark, params.database, "forecast")
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
