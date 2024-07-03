import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from spark_forecast.params import Params, SplitParams, read_config
from spark_forecast.schema import SplitSchema
from spark_forecast.split import Split
from spark_forecast.utils import (
    read_delta_table,
    set_mlflow_experiment,
    write_delta_table,
)


class SplitTask:
    def __init__(
        self,
        params: SplitParams,
    ) -> None:
        self.params = params

    def read(self, spark: SparkSession) -> DataFrame:
        df = read_delta_table(
            spark,
            self.params.database,
            "input",
        )
        return df

    def write(self, spark: SparkSession, df: DataFrame) -> None:
        write_delta_table(
            spark,
            df,
            SplitSchema,
            self.params.database,
            "split",
        )

    def transform(self, df: DataFrame) -> DataFrame:
        split = Split(
            group_columns=self.params.group_columns,
            time_column=self.params.time_column,
            target_column=self.params.target_column,
            execution_date=self.params.execution_date,
            time_delta=self.params.time_delta,
            test_size=self.params.test_size,
            freq=self.params.freq,
        )
        df_split = split.transform(df, SplitSchema)
        return df_split

    def launch(self, spark: SparkSession):
        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.params.__dict__)

        df = self.read(spark)
        df_split = self.transform(df)
        self.write(spark, df_split)


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = SplitTask(params.split)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
