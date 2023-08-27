from datetime import date, timedelta

import pandas as pd
from poc_forecast_cm.schema import SalesSchema, SplitSchema
from poc_forecast_cm.split import Split
from pyspark.sql import Row, SparkSession

from tests.unit.utils import assert_pyspark_df_equal

execution_date = date(2022, 9, 12)
time_delta = 2 * 52 * 7
group_columns = ["location_id"]
time_column = "tran_start_dt"
target_column = "q_trx"
test_size = 14


def test_split(spark: SparkSession):
    df_sales = spark.createDataFrame(
        [
            Row(
                location_id="T104",
                tran_start_dt=date(2022, 8, 29),
                q_trx=1.0,
            ),
            Row(
                location_id="T104",
                tran_start_dt=date(2022, 9, 10),
                q_trx=7.0,
            ),
        ],
        schema=SalesSchema,
    )

    df_expected_split = spark.createDataFrame(
        pd.DataFrame(
            {
                "location_id": "T104",
                "tran_start_dt": [
                    date(2022, 8, 29) + timedelta(i) for i in range(14)
                ],
                "q_trx": [1.0] + [0.0] * 11 + [7.0, 0.0],
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
    df_split = split.transform(df_sales)

    assert_pyspark_df_equal(df_expected_split, df_split)
