from typing import Optional

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
    path: Optional[str] = None,
) -> DataFrame:
    if path:
        return spark.read.format("delta").load(path)
    return spark.read.table(f"{database}.{table}")


def write_delta_table(
    spark: SparkSession,
    df: DataFrame,
    path: str,
    schema: StructType,
    database: str,
    table: str,
) -> None:
    (
        DeltaTable.createIfNotExists(spark)
        .tableName(f"{database}.{table}")
        .addColumns(schema)
        .location(path)
        .execute()
    )
    df.write.format("delta").mode("overwrite").save(path)


def extract_timeseries_from_pandas_dataframe(
    df: pd.DataFrame, time_column: str, target_column: str
) -> TimeSeries:
    serie = pd.Series(
        df[target_column].values, index=pd.to_datetime(df[time_column])
    )
    time_serie = TimeSeries.from_series(serie, freq="D")
    return time_serie
