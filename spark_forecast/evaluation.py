from collections.abc import Callable
from typing import Union

import darts.metrics
import pandas as pd
import pyspark.sql.functions as F
from darts.timeseries import TimeSeries
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType

from spark_forecast.utils import extract_timeseries_from_pandas_dataframe

MetricType = Callable[[TimeSeries, TimeSeries], float]


def mape(epsilon: float = 0.0) -> MetricType:
    def mape(y_true: TimeSeries, y_pred: TimeSeries) -> float:
        return (
            abs(y_true.pd_series() - y_pred.pd_series())
            / (y_true.pd_series() + epsilon)
        ).mean()

    return mape


class Metrics:
    def __init__(
        self,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        freq: str = "1D",
    ):
        self.group_columns = ["model"] + group_columns
        self.time_column = time_column
        self.target_column = target_column
        self.freq = freq

    def set_group_columns(self, df: pd.DataFrame) -> None:
        self.group_columns_values = df.loc[0, self.group_columns].to_dict()

    def evaluate(
        self,
        df: pd.DataFrame,
        metric: MetricType,
    ) -> pd.DataFrame:
        self.set_group_columns(df)
        y = extract_timeseries_from_pandas_dataframe(
            df, self.time_column, self.target_column, self.freq
        )
        y_pred = extract_timeseries_from_pandas_dataframe(
            df, self.time_column, "prediction", self.freq
        )
        metric_value = metric(y, y_pred)
        df_metric = pd.DataFrame(
            [
                {
                    **self.group_columns_values,
                    "metric": metric.__name__,
                    "value": metric_value,
                }
            ]
        )
        return df_metric

    @staticmethod
    def get_metrics(
        metrics: list[MetricType],
        df: pd.DataFrame,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        freq: str = "1D",
    ) -> pd.DataFrame:
        df_metrics = pd.concat(
            [
                Metrics(
                    group_columns, time_column, target_column, freq
                ).evaluate(df, metric)
                for metric in metrics
            ]
        )
        return df_metrics


class Evaluation:
    def __init__(
        self,
        group_columns: list[str],
        time_column: str,
        target_column: str,
        metrics: list[Union[str, MetricType]],
        model_selection_metric: str,
        freq: str = "1D",
    ):
        self.group_columns = group_columns
        self.time_column = time_column
        self.target_column = target_column
        self.metrics = metrics
        self.model_selection_metric = model_selection_metric
        self.freq = freq

    def get_metrics(
        self,
        df_test: DataFrame,
        df_forecast_on_test: DataFrame,
        metrics_schema: StructType,
    ) -> DataFrame:
        df = df_test.join(
            df_forecast_on_test.withColumnRenamed(
                self.target_column, "prediction"
            ),
            on=[*self.group_columns, self.time_column],
            how="inner",
        )

        metrics = [
            getattr(darts.metrics, metric)
            if isinstance(metric, str)
            else metric
            for metric in self.metrics
        ]
        df_metrics = df.groupBy(["model", *self.group_columns]).applyInPandas(
            lambda pdf: Metrics.get_metrics(
                df=pdf,
                metrics=metrics,
                group_columns=self.group_columns,
                time_column=self.time_column,
                target_column=self.target_column,
                freq=self.freq,
            ),
            schema=metrics_schema,
        )
        return df_metrics

    def get_best_models(self, df_metrics: DataFrame) -> DataFrame:
        df_best_models = (
            df_metrics.filter(F.col("metric") == self.model_selection_metric)
            .groupBy(self.group_columns)
            .agg(F.min(F.struct("value", "model")).alias("arr"))
            .withColumn("model", F.col("arr.model"))
            .withColumn("value", F.col("arr.value"))
            .drop("arr")
            .withColumn("metric", F.lit(self.model_selection_metric))
            .select(*df_metrics.columns)
        )
        return df_best_models

    def get_forecast(
        self, df_all_models_forecast: DataFrame, df_best_models: DataFrame
    ) -> DataFrame:
        df_forecast = df_all_models_forecast.join(
            df_best_models.drop("metric", "value"),
            on=["model"] + self.group_columns,
            how="inner",
        )
        return df_forecast
