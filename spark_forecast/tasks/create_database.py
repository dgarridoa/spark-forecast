import mlflow
from pyspark.sql import SparkSession

from spark_forecast.params import (
    CommonParams,
    Params,
    read_config,
)
from spark_forecast.utils import set_mlflow_experiment


class CreateDataBaseTask:
    def __init__(self, params: CommonParams):
        self.params = params

    def launch(self, spark: SparkSession):
        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.params.__dict__)

        spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.params.database}")


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = CreateDataBaseTask(params.split)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
