import argparse
import sys
from typing import TypeVar

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from spark_forecast.model import DistributedModel, ModelProtocol
from spark_forecast.params import ModelParams, Params, read_config
from spark_forecast.schema import ForecastSchema
from spark_forecast.utils import (
    read_delta_table,
    set_mlflow_experiment,
    write_delta_table,
)

T = TypeVar("T", bound=ModelProtocol)


class ModelTask:
    def __init__(
        self,
        params: ModelParams,
    ) -> None:
        self.params = params

    def read(self, spark: SparkSession) -> DataFrame:
        df = read_delta_table(spark, self.params.database, "split")
        return df

    def write(
        self, spark: SparkSession, df: DataFrame, table_name: str
    ) -> None:
        write_delta_table(
            spark,
            df,
            ForecastSchema,
            self.params.database,
            table_name,
            partition_cols=["model"],
        )

    def fit_predict(self, df_train: DataFrame, steps: int) -> DataFrame:
        distributed_model = DistributedModel(
            group_columns=self.params.group_columns,
            time_column=self.params.time_column,
            target_column=self.params.target_column,
            model_cls=self.params.model_cls,
            model_params=self.params.model_params,
            freq=self.params.freq,
        )
        df_predict = distributed_model.fit_predict(
            df_train=df_train,
            forecast_schema=ForecastSchema,
            steps=steps,
        )
        return df_predict

    def launch(self, spark: SparkSession):
        run_name = f"{self.__class__.__name__}[{self.params.model_cls}]"

        set_mlflow_experiment()
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(self.params.__dict__)

        df = self.read(spark)
        num_partitions = (
            df.select(*self.params.group_columns).distinct().cache().count()
        )
        df = df.repartition(num_partitions, *self.params.group_columns).cache()
        df_train = df.filter(df["split"] == "train")

        df_forecast_on_test = self.fit_predict(df_train, self.params.test_size)
        df_forecast = self.fit_predict(df, self.params.steps)

        self.write(spark, df_forecast_on_test, "forecast_on_test")
        self.write(spark, df_forecast, "all_models_forecast")


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="ExponentialSmoothing")
    model_name = parser.parse_known_args(sys.argv[1:])[0].model_name

    task = ModelTask(params.models[model_name])
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
