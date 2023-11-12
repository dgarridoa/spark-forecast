import copy
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import pyspark.sql.functions as F
from darts.timeseries import TimeSeries
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType


class Split:
    def __init__(
        self,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        test_size: int = 14,
        execution_date: Optional[date] = None,
        time_delta: int = 2 * 52 * 7 - 1,
        freq: str = "1D",
    ):
        self.group_columns = group_columns
        self.time_column = time_column
        self.target_column = target_column
        self.test_size = test_size
        self.freq = freq

        if execution_date is None:
            execution_date = date.today() - timedelta(
                days=date.today().weekday()
            )
        self.execution_date = execution_date
        self.time_delta = time_delta
        self.end_date = self.execution_date - pd.Timedelta(days=1)
        self.start_date = self.execution_date - self.time_delta * pd.Timedelta(
            self.freq
        )
        self.end_train_date = self.end_date - self.test_size * pd.Timedelta(
            self.freq
        )

    def add_dummy_date(self, df: pd.DataFrame) -> None:
        dummy_row = copy.deepcopy(df.iloc[0])
        dummy_row[self.time_column] = self.execution_date
        dummy_row[self.target_column] = 0
        df.loc[len(df)] = dummy_row

    def fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        self.add_dummy_date(df)
        serie = pd.Series(
            df.loc[:, self.target_column].values,
            index=pd.to_datetime(df.loc[:, self.time_column]),
        )
        time_serie = TimeSeries.from_series(
            serie, fill_missing_dates=True, fillna_value=0, freq=self.freq
        )
        serie = time_serie.pd_series().iloc[:-1]

        group_columns_values = df.loc[0, self.group_columns].to_dict()
        df = pd.DataFrame(
            {
                **group_columns_values,
                self.time_column: serie.index,
                self.target_column: serie.values,
            }
        )
        return df

    def transform(self, df: DataFrame, split_schema: StructType) -> DataFrame:
        df = df.filter(
            F.col(self.time_column).between(self.start_date, self.end_date)
        )

        df = df.groupBy(*self.group_columns).applyInPandas(
            lambda pdf: self.fill_missing_dates(pdf), schema=df.schema
        )

        df_split = df.withColumn(
            "split",
            F.when(
                F.col(self.time_column) > self.end_train_date, "test"
            ).otherwise("train"),
        ).select(*split_schema.fieldNames())
        return df_split
