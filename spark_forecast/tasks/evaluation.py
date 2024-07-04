import mlflow
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType

from spark_forecast.evaluation import Evaluation
from spark_forecast.params import EvaluationParams, Params, read_config
from spark_forecast.schema import ForecastSchema, MetricsSchema
from spark_forecast.utils import (
    read_delta_table,
    set_mlflow_experiment,
    write_delta_table,
)


class EvaluationTask:
    def __init__(
        self,
        params: EvaluationParams,
    ) -> None:
        self.params = params

    def read(self, spark: SparkSession, table_name: str) -> DataFrame:
        df = read_delta_table(spark, self.params.database, table_name)
        return df

    def write(
        self,
        spark: SparkSession,
        df: DataFrame,
        schema: StructType,
        table_name: str,
    ) -> None:
        write_delta_table(spark, df, schema, self.params.database, table_name)

    def launch(self, spark: SparkSession):
        evaluation = Evaluation(
            group_columns=self.params.group_columns,
            time_column=self.params.time_column,
            target_column=self.params.target_column,
            metrics=self.params.metrics,
            model_selection_metric=self.params.model_selection_metric,
            freq=self.params.freq,
        )

        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.params.__dict__)

            df_test = self.read(spark, "split").filter(
                F.col("split") == "test"
            )
            df_forecast_on_test = self.read(spark, "forecast_on_test")
            df_metrics = evaluation.get_metrics(
                df_test, df_forecast_on_test, MetricsSchema
            )
            self.write(spark, df_metrics, MetricsSchema, "metrics")

            df_metrics = self.read(spark, "metrics")
            df_best_models = evaluation.get_best_models(df_metrics).cache()
            mlflow.log_metric(
                self.params.model_selection_metric,
                df_best_models.agg(F.mean("value")).collect()[0][0],
            )
            self.write(spark, df_best_models, MetricsSchema, "best_models")

        df_all_models_forecast = self.read(spark, "all_models_forecast")
        df_best_models = self.read(spark, "best_models")
        df_forecast = evaluation.get_forecast(
            df_all_models_forecast, df_best_models
        )
        self.write(spark, df_forecast, ForecastSchema, "forecast")


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = EvaluationTask(params.evaluation)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
