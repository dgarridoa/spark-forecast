from typing import Optional, Protocol, Type, TypeVar, runtime_checkable

import pandas as pd
from darts.timeseries import TimeSeries
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType

from dbx_demand_forecast.utils import extract_timeseries_from_pandas_dataframe


@runtime_checkable
class ModelProtocol(Protocol):
    def __init__(self):
        ...

    def fit(self, series: TimeSeries) -> Optional["ModelProtocol"]:
        ...

    def predict(self, n: int) -> TimeSeries:
        ...


T = TypeVar("T", bound=ModelProtocol)


class Model:
    def __init__(
        self,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        model_cls: Type[T],
        model_params: dict = {},
        freq: str = "1D",
    ):
        self.group_columns = group_columns
        self.time_column = time_column
        self.target_column = target_column
        self.model = model_cls(**model_params)
        self.freq = freq

    def set_group_columns_values(self, df: pd.DataFrame) -> None:
        self.group_columns_values = df.loc[0, self.group_columns].to_dict()

    def fit(self, df_train: pd.DataFrame) -> None:
        self.set_group_columns_values(df_train)
        y_train = extract_timeseries_from_pandas_dataframe(
            df_train, self.time_column, self.target_column, self.freq
        )

        self.model.fit(y_train)

    def predict(self, steps: int) -> pd.DataFrame:
        y_predict = self.model.predict(steps).pd_series()
        df = pd.DataFrame(
            {
                "model": self.model.__class__.__name__,
                **self.group_columns_values,
                self.time_column: y_predict.index,
                self.target_column: y_predict.values,
            }
        )
        return df

    @staticmethod
    def fit_predict(
        df_train: pd.DataFrame,
        steps: int,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        model_cls: Type[ModelProtocol],
        model_params: dict = {},
        freq: str = "1D",
    ) -> pd.DataFrame:
        model = Model(
            group_columns,
            time_column,
            target_column,
            model_cls,
            model_params,
            freq,
        )
        model.fit(df_train)
        df_predict = model.predict(steps)
        return df_predict


class DistributedModel:
    def __init__(
        self,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        model_cls: Type[ModelProtocol],
        model_params: dict = {},
        freq: str = "1D",
    ):
        self.group_columns = group_columns
        self.time_column = time_column
        self.target_column = target_column
        self.model_cls = model_cls
        self.model_params = model_params
        self.freq = freq

    def fit_predict(
        self, df_train: DataFrame, forecast_schema: StructType, steps: int
    ) -> DataFrame:
        df_predict = df_train.groupBy(self.group_columns).applyInPandas(
            lambda pdf: Model.fit_predict(
                df_train=pdf,
                steps=steps,
                group_columns=self.group_columns,
                time_column=self.time_column,
                target_column=self.target_column,
                model_cls=self.model_cls,
                model_params=self.model_params,
                freq=self.freq,
            ),
            schema=forecast_schema,
        )
        return df_predict
