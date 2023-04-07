from collections.abc import Callable

import darts.metrics
import mlflow
import pandas as pd
import pyspark.sql.functions as F
from darts.timeseries import TimeSeries
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType

from dbx_demand_forecast.common import Task
from dbx_demand_forecast.schema import ForecastSchema, MetricsSchema
from dbx_demand_forecast.utils import (
    extract_timeseries_from_pandas_dataframe,
    read_delta_table,
    write_delta_table,
)


class Metrics:
    def __init__(
        self,
        group_columns: list[str],
        time_column: str,
        target_column: str,
    ):
        self.group_columns = ["model"] + group_columns
        self.time_column = time_column
        self.target_column = target_column

    def set_group_columns(self, df: pd.DataFrame) -> None:
        self.group_columns_values = df.loc[0, self.group_columns].to_dict()

    def evaluate(
        self,
        df: pd.DataFrame,
        metric: Callable[[TimeSeries, TimeSeries], float],
    ) -> pd.DataFrame:
        self.set_group_columns(df)
        y = extract_timeseries_from_pandas_dataframe(df, self.time_column, "y")
        y_pred = extract_timeseries_from_pandas_dataframe(
            df, self.time_column, "y_pred"
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
        metrics: list[Callable[[TimeSeries, TimeSeries], float]],
        df: pd.DataFrame,
        group_columns: list[str],
        time_column: str,
        target_column: str,
    ) -> pd.DataFrame:
        df_metrics = pd.concat(
            [
                Metrics(group_columns, time_column, target_column).evaluate(
                    df, metric
                )
                for metric in metrics
            ]
        )
        return df_metrics


class EvaluationTask(Task):
    def _read_delta_table(self, input_dict: dict[str, str]) -> DataFrame:
        df = read_delta_table(self.spark, path=input_dict["path"])
        return df

    def _write_delta_table(
        self, df: DataFrame, schema: StructType, output_dict: dict[str, str]
    ) -> None:
        write_delta_table(
            self.spark,
            df,
            output_dict["path"],
            schema,
            output_dict["database"],
            output_dict["table"],
        )

    def get_metrics(
        self, df_test: DataFrame, df_forecast_on_test: DataFrame
    ) -> DataFrame:
        df = df_test.withColumnRenamed(self.conf["target_column"], "y").join(
            df_forecast_on_test.withColumnRenamed(
                self.conf["target_column"], "y_pred"
            ),
            on=["store", "item", "date"],
            how="left",
        )

        metrics = [
            getattr(darts.metrics, metric) for metric in self.conf["metrics"]
        ]
        group_columns: list[str] = self.conf["group_columns"]
        time_column: str = self.conf["time_column"]
        target_column: str = self.conf["target_column"]
        df_metrics = df.groupBy(["model", "store", "item"]).applyInPandas(
            lambda pdf: Metrics.get_metrics(
                df=pdf,
                metrics=metrics,
                group_columns=group_columns,
                time_column=time_column,
                target_column=target_column,
            ),
            schema=MetricsSchema,
        )
        return df_metrics

    def get_best_models(self, df_metrics: DataFrame) -> DataFrame:
        df_best_models = (
            df_metrics.filter(
                F.col("metric") == self.conf["model_selection_metric"]
            )
            .groupBy(self.conf["group_columns"])
            .agg(F.min(F.struct("value", "model")).alias("arr"))
            .withColumn("model", F.col("arr.model"))
            .withColumn("value", F.col("arr.value"))
            .drop("arr")
            .withColumn("metric", F.lit(self.conf["model_selection_metric"]))
            .select(*MetricsSchema.fieldNames())
        )
        return df_best_models

    def get_forecast(
        self, df_all_models_forecast: DataFrame, df_best_models: DataFrame
    ) -> DataFrame:
        df_forecast = df_all_models_forecast.join(
            df_best_models.drop("metric", "value"),
            on=["model"] + self.conf["group_columns"],
            how="inner",
        )
        return df_forecast

    def launch(self):
        self.logger.info(f"Launching {self.__class__.__name__}")

        mlflow.set_experiment(self.conf["experiment"])
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.conf)

            df_test = self._read_delta_table(
                self.conf["input"]["split"]
            ).filter(F.col("split") == "test")
            df_forecast_on_test = self._read_delta_table(
                self.conf["input"]["forecast_on_test"]
            )
            df_metrics = self.get_metrics(df_test, df_forecast_on_test)
            df_best_models = self.get_best_models(df_metrics).cache()

            mlflow.log_metric(
                self.conf["model_selection_metric"],
                df_best_models.agg(F.mean("value")).collect()[0][0],
            )

        df_all_models_forecast = self._read_delta_table(
            self.conf["input"]["all_models_forecast"]
        )
        df_forecast = self.get_forecast(df_all_models_forecast, df_best_models)

        self._write_delta_table(
            df_metrics, MetricsSchema, self.conf["output"]["metrics"]
        )
        self._write_delta_table(
            df_best_models, MetricsSchema, self.conf["output"]["best_models"]
        )
        self._write_delta_table(
            df_forecast, ForecastSchema, self.conf["output"]["forecast"]
        )


def entrypoint():
    task = EvaluationTask()
    task.launch()


if __name__ == "__main__":
    entrypoint()
