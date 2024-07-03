import os

import mlflow
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from spark_forecast.params import IngestionParams, Params, read_config
from spark_forecast.schema import InputSchema, SalesSchema
from spark_forecast.utils import (
    read_csv,
    set_mlflow_experiment,
    write_delta_table,
)


class IngestionTask:
    def __init__(self, params: IngestionParams):
        self.params = params

    def read(self, spark: SparkSession) -> DataFrame:
        file_path = os.getenv("WORKSPACE_FILE_PATH")
        if file_path:
            self.params.path = "file:{file_path}/{relative_path}".format(
                file_path=file_path, relative_path=self.params.path
            )
        df = read_csv(
            spark,
            self.params.path,
            self.params.sep,
            SalesSchema,
        )

        if self.params.stores:
            df = df.filter(F.col("store").isin(self.params.stores))
        return df

    def write(self, spark: SparkSession, df: DataFrame) -> None:
        write_delta_table(
            spark, df, InputSchema, self.params.database, "input"
        )

    def launch(self, spark: SparkSession):
        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.params.__dict__)

        df = self.read(spark)
        self.write(spark, df)


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = IngestionTask(params.ingestion)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
