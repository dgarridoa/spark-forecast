from datetime import date, timedelta

import pandas as pd
from pyspark.sql import Row, SparkSession

from spark_forecast.schema import SalesSchema, SplitSchema
from spark_forecast.split import Split
from tests.utils import assert_pyspark_df_equal

execution_date = date(2022, 9, 12)
time_delta = 2 * 52 * 7
group_columns = ["store", "item"]
time_column = "date"
target_column = "sales"
test_size = 7


def test_split(spark: SparkSession):
    df_sales = spark.createDataFrame(
        [
            Row(
                date=date(2022, 8, 29),
                store=1,
                item=1,
                sales=1.0,
            ),
            Row(
                date=date(2022, 9, 10),
                store=1,
                item=1,
                sales=7.0,
            ),
        ],
        schema=SalesSchema,
    )

    df_expected_split = spark.createDataFrame(
        pd.DataFrame(
            {
                "store": 1,
                "item": 1,
                "date": [date(2022, 8, 29) + timedelta(i) for i in range(14)],
                "sales": [1.0] + [0.0] * 11 + [7.0, 0.0],
                "split:": (["train"] * 7 + ["test"] * 7),
            }
        ),
        schema=SplitSchema,
    )

    split = Split(
        group_columns,
        time_column,
        target_column,
        test_size,
        execution_date,
        time_delta,
    )
    df_split = split.transform(df_sales, SplitSchema)

    assert_pyspark_df_equal(df_expected_split, df_split)
