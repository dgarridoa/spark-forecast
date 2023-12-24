from typing import Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType


def assert_pyspark_df_equal(
    expected_df: DataFrame,
    actual_df: DataFrame,
    round: Optional[int] = None,
    ignore_order: bool = True,
) -> None:
    assert (
        expected_df.dtypes == actual_df.dtypes
    ), "DataFrame schemas are different"

    if round is not None:
        for col in (
            field.name
            for field in expected_df.schema
            if isinstance(field.dataType, NumericType)
        ):
            expected_df = expected_df.withColumn(col, F.round(col, 2))
            actual_df = actual_df.withColumn(col, F.round(col, 2))

    if not ignore_order:
        expected_df = expected_df.withColumn(
            "_index", F.monotonically_increasing_id()
        )
        actual_df = actual_df.withColumn(
            "_index", F.monotonically_increasing_id()
        )

    assert (
        expected_df.count() == actual_df.count()
    ), "DataFrame row counts are different"

    def are_dfs_equal(df1, df2):
        if ignore_order:
            df1 = df1.orderBy([F.col(c).asc_nulls_last() for c in df1.columns])
            df2 = df2.orderBy([F.col(c).asc_nulls_last() for c in df2.columns])
        return (
            df1.subtract(df2).count() == 0 and df2.subtract(df1).count() == 0
        )

    assert are_dfs_equal(
        expected_df, actual_df
    ), "DataFrames contain different data"
