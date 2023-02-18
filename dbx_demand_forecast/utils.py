from typing import Optional

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
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {database}")
    (
        DeltaTable.createIfNotExists(spark)
        .tableName(f"{database}.{table}")
        .addColumns(schema)
        .location(path)
        .execute()
    )
    df.write.format("delta").mode("overwrite").save(path)
