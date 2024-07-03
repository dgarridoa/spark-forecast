import os
from typing import Optional

import mlflow
import pandas as pd
from darts.timeseries import TimeSeries
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType


def read_csv(
    spark: SparkSession,
    path: str,
    sep: str,
    schema: StructType,
) -> DataFrame:
    return spark.read.csv(path, sep=sep, header=True, schema=schema)


def read_delta_table(
    spark: SparkSession,
    database: Optional[str] = None,
    table: Optional[str] = None,
) -> DataFrame:
    return spark.read.table(f"{database}.{table}")


def write_delta_table(
    spark: SparkSession,
    df: DataFrame,
    schema: StructType,
    database: str,
    table: str,
    mode: str = "overwrite",
    partition_cols: Optional[list[str]] = None,
) -> None:
    if partition_cols:
        (
            DeltaTable.createIfNotExists(spark)
            .tableName(f"{database}.{table}")
            .addColumns(schema)
            .partitionedBy(partition_cols)
            .execute()
        )
        df.write.format("delta").partitionBy(*partition_cols).mode(
            mode
        ).option("partitionOverwriteMode", "dynamic").saveAsTable(
            f"{database}.{table}"
        )
    else:
        (
            DeltaTable.createIfNotExists(spark)
            .tableName(f"{database}.{table}")
            .addColumns(schema)
            .execute()
        )
        df.write.format("delta").saveAsTable(f"{database}.{table}", mode=mode)


def extract_timeseries_from_pandas_dataframe(
    df: pd.DataFrame, time_column: str, target_column: str, freq: str = "D"
) -> TimeSeries:
    serie = pd.Series(
        df[target_column].values, index=pd.to_datetime(df[time_column])
    )
    time_serie = TimeSeries.from_series(serie, freq=freq)
    return time_serie


def set_mlflow_experiment() -> None:
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not experiment_name:
        raise ValueError(
            "environment variable MLFLOW_EXPERIMENT_NAME is unset"
        )
    mlflow.set_experiment(experiment_name)
