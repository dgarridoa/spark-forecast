from datetime import date, timedelta

import pandas as pd
import pytest
from darts.timeseries import TimeSeries
from pyspark.sql import DataFrame, SparkSession

from spark_forecast.utils import (
    extract_timeseries_from_pandas_dataframe,
    get_table_info,
    read_delta_table,
    write_delta_table,
)
from tests.utils import assert_pyspark_df_equal


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "dt": date(2018, 12, 1),
                "date": [date(2018, 12, 1) + timedelta(i) for i in range(30)],
                "store": 1,
                "item": 1,
                "sales": map(float, range(1, 31)),
            }
        ),
    )


def test_write_delta_table(spark: SparkSession, df: DataFrame):
    schema = df.schema
    database = "default"

    table = "internal_table"
    write_delta_table(spark, df, schema, database, table)
    assert get_table_info(spark, database, table)["Type"] == "MANAGED"
    delta_table_df = read_delta_table(spark, database, table)
    assert_pyspark_df_equal(df, delta_table_df)

    table = "internal_table_with_partitions"
    write_delta_table(spark, df, schema, database, table, "overwrite", ["dt"])
    assert get_table_info(spark, database, table)["Type"] == "MANAGED"
    delta_table_df = read_delta_table(spark, database, table)
    assert_pyspark_df_equal(df, delta_table_df)


def test_extract_timeseries_from_pandas_dataframe():
    expected_series = pd.Series(
        range(3),
        index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
    )
    expected_time_series = TimeSeries.from_series(expected_series, freq="D")

    df = pd.DataFrame(
        {
            "time": [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)],
            "target": range(3),
        }
    )
    time_series = extract_timeseries_from_pandas_dataframe(
        df, "time", "target", freq="D"
    )
    print(expected_time_series)
    print(time_series)
    assert expected_time_series == time_series
